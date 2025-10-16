"""
Modified from guided-diffusion/scripts/image_sample.py

本脚本用于对输入真实图像进行：
1) DDIM 反演（x0 -> xT）
2) 重建采样（xT -> x'0）
3) 二次重构（x'0 -> x''0）
4) 计算三种误差：
   - D1 = |x0 - x'0|
   - D2 = |x'0 - x''0|
   - DiffRecon = D1 - D2  （差分重构）
（可选）绝对值设置：
  - --abs_d1 / --abs_d2 控制 D1、D2 是否取绝对值（默认取绝对值以复现实验设定）
  - --abs_final_diff 控制 DiffRecon 是否对 (D1 - D2) 再取绝对值（默认不取）
可用于做消融：绝对值版强调“强度”，非绝对值版保留“方向性”信息。
并将重建图与误差图保存到磁盘。

注意：
- .pt 权重文件仅包含参数，模型结构由 guided_diffusion 代码创建。
- 对 LSUN-Bedroom(Real) 必须使用 lsun_bedroom.pt（在LSUN-Bedroom上训练的ADM权重）。
- 反演与重建均调用 diffusion 对象内置的 `ddim_reverse_sample_loop` 与 `ddim_sample_loop` 工具函数，无需自行实现。
"""

import argparse
import os
import torch

import sys
import cv2

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th
import torch.distributed as dist

# Helpers for distributed info and barrier
def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def _maybe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

# guided_diffusion 子包中的工具与日志器：
# - dist_util: 设备/分布式初始化、加载权重的辅助函数
# - logger: 简单日志记录器
from guided_diffusion import dist_util, logger

# 数据加载器（带“反演/重建”任务所需的预处理）
from guided_diffusion.image_datasets import load_data_for_reverse

# 脚本工具：
# - NUM_CLASSES: 若为类条件模型（如ImageNet）时的类别数
# - model_and_diffusion_defaults: 返回模型与扩散过程的默认配置
# - create_model_and_diffusion: 根据配置创建 UNet 模型与 Diffusion 对象
# - add_dict_to_argparser/args_to_dict: 将配置与命令行参数互转
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    将输入图像张量统一到 (N, 3, image_size, image_size)。

    处理逻辑：
    1) 若为单张 (3, H, W)，先扩成 batch 维度；
    2) 若宽高不等，中心裁剪为正方形；
    3) 若尺寸与 image_size 不一致，使用双三次插值到指定分辨率。

    这样可以确保后续进入扩散模型的张量尺寸正确。
    """
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def count_images_in_dir(root: str) -> int:
    """
    递归统计 root 目录下的图像文件数量（支持常见后缀）。
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                total += 1
    return total

# def to_u8_vis(t: th.Tensor) -> th.Tensor:
#     """
#     将任意范围（包含可能为负）的张量可视化为 uint8。
#     - 若存在负值：按对称范围 [-amax, amax] 线性映射到 [0,255]（0.5 为 0）
#     - 若非负：按最大值线性映射到 [0,255]（兼容原先 |·|∈[0,2] 的 255/2 缩放）
#     形状保持 (N,C,H,W)；返回 (N,H,W,C) uint8。
#     """
#     # (N,C,H,W)
#     t = t.clone().detach()
#     n,c,h,w = t.shape
#     t_min = t.amin(dim=(1,2,3), keepdim=True)
#     t_max = t.amax(dim=(1,2,3), keepdim=True)
#     has_neg = (t_min < 0).any()
#     if has_neg:
#         amax = th.maximum(-t_min, t_max)  # 每个样本的对称上界
#         amax = th.clamp(amax, min=1e-8)
#         x01 = (t / (2*amax)) + 0.5
#     else:
#         denom = th.clamp(t_max, min=1e-8)
#         x01 = t / denom
#     x255 = (x01.clamp(0,1) * 255.0).round().to(th.uint8)
#     return x255.permute(0,2,3,1).contiguous()


def to_u8_vis(t: th.Tensor, min_val: float = None, max_val: float = None) -> th.Tensor:
    """
    将张量可视化为 uint8（返回形状 (N,H,W,C)）。
    可选两种放缩模式：
      - 固定放缩（传入 min_val, max_val）：按固定区间 [min_val, max_val] 线性映射到 [0,255]
      - 自适应放缩（min_val/max_val 为 None）：逐样本自适应到 [0,255]
        * 若样本包含负值：对称到 [-amax, amax] 再映射
        * 若样本全非负：按该样本最大值映射
    参数：
      t: (N,C,H,W) 的张量
      min_val, max_val: 固定放缩下的全局下/上界；两者需同时提供
    """
    t = t.clone().detach()
    # 固定放缩：使用统一区间
    if (min_val is not None) and (max_val is not None):
        denom = max_val - min_val
        # 防止除零
        if abs(denom) < 1e-8:
            denom = 1e-8
        x01 = (t - min_val) / denom
        x255 = (x01.clamp(0, 1) * 255.0).round().to(th.uint8)
        return x255.permute(0, 2, 3, 1).contiguous()

    # 自适应放缩：逐样本归一化
    n, c, h, w = t.shape
    t_min = t.amin(dim=(1, 2, 3), keepdim=True)
    t_max = t.amax(dim=(1, 2, 3), keepdim=True)
    has_neg = (t_min < 0).any()

    if has_neg:
        amax = th.maximum(-t_min, t_max)  # 每个样本的对称上界
        amax = th.clamp(amax, min=1e-8)
        x01 = (t / (2 * amax)) + 0.5
    else:
        denom = th.clamp(t_max, min=1e-8)
        x01 = t / denom

    x255 = (x01.clamp(0, 1) * 255.0).round().to(th.uint8)
    return x255.permute(0, 2, 3, 1).contiguous()


def main():
    # 解析命令行参数（包含数据路径、输出路径、模型配置等）
    args = create_argparser().parse_args()
    is_dist_env = False

    # 若未显式指定，自动将 num_samples 设为 images_dir 下图片总数
    if getattr(args, "num_samples", -1) is None or args.num_samples <= 0:
        auto_count = count_images_in_dir(args.images_dir)
        # 若目录为空，保持为 0；否则设为统计值
        args.num_samples = max(0, int(auto_count))

    # 初始化分布式/设备
    dist_util.setup_dist(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    # 标记是否处于分布式环境，并取得 rank/world
    is_dist_env = dist.is_available() and dist.is_initialized()
    rank, world = _dist_info()

    # 配置日志输出目录（将日志写到重建图的输出目录）
    if rank == 0:
        logger.configure(dir=args.recons_dir)
        logger.log(f"[auto] num_samples = {args.num_samples} (from images_dir='{args.images_dir}')")

    # 创建重建输出与DIRE输出的目录
    os.makedirs(args.recons_dir, exist_ok=True)
    os.makedirs(args.dire_dir, exist_ok=True)

    # 新增：第二次重建单独目录
    if hasattr(args, "recons2_dir") and args.recons2_dir:
        os.makedirs(args.recons2_dir, exist_ok=True)
    else:
        args.recons2_dir = os.path.join(args.recons_dir, "..", "recons2")
        os.makedirs(args.recons2_dir, exist_ok=True)

    # 新增输出目录：第二次误差与差分重构
    if hasattr(args, "dire2_dir") and args.dire2_dir:
        os.makedirs(args.dire2_dir, exist_ok=True)
    else:
        args.dire2_dir = os.path.join(args.recons_dir, "..", "dire2")
        os.makedirs(args.dire2_dir, exist_ok=True)

    if hasattr(args, "diff_dir") and args.diff_dir:
        os.makedirs(args.diff_dir, exist_ok=True)
    else:
        args.diff_dir = os.path.join(args.recons_dir, "..", "diffrecon")
        os.makedirs(args.diff_dir, exist_ok=True)

    # 打印当前运行参数，方便复现实验（仅主进程）
    if rank == 0:
        logger.log(str(args))

    # === 自动确定固定放缩范围（基于是否取绝对值） ===
    if getattr(args, "vis_mode", "adaptive") == "fixed":
        # D1/D2：若取绝对值则 [0,2]，否则 [-2,2]
        vis_min_d1, vis_max_d1 = (0.0, 2.0) if getattr(args, "abs_d1", True) else (-2.0, 2.0)
        vis_min_d2, vis_max_d2 = (0.0, 2.0) if getattr(args, "abs_d2", True) else (-2.0, 2.0)

        # DiffRecon：区间随 D1/D2 的取值域与最终是否再 abs 而变
        # 若 D1/D2 都是 [0,2]，则差 [-2,2]；若都是 [-2,2]，则差 [-4,4]
        base = 2.0 if (getattr(args, "abs_d1", True) and getattr(args, "abs_d2", True)) else 4.0
        if getattr(args, "abs_final_diff", False):
            vis_min_diff, vis_max_diff = 0.0, base
        else:
            vis_min_diff, vis_max_diff = -base, base

        if rank == 0:
            logger.log(f"[vis-range] D1=({vis_min_d1},{vis_max_d1}), D2=({vis_min_d2},{vis_max_d2}), Diff=({vis_min_diff},{vis_max_diff})")
    else:
        vis_min_d1 = vis_max_d1 = vis_min_d2 = vis_max_d2 = vis_min_diff = vis_max_diff = None

    # === 1) 创建模型与扩散过程（模型结构在代码里，参数随后从 .pt 加载） ===
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 从 args.model_path 加载权重；.pt 一般为 state_dict 或 {'model': state_dict}
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))

    # 将模型移动到设备（GPU/CPU）
    model.to(dist_util.dev())
    logger.log("have created model and diffusion")

    # 可选：使用半精度以节省显存（部分环境若不稳定可关闭）
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # === 2) 数据加载器：读取原始图像，并预处理到 [-1,1]、指定分辨率 ===
    # 注意：class_cond=False 时不加载类别；LSUN-Bedroom 使用无条件模型，应设为 False
    data = load_data_for_reverse(
        data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond
    )
    logger.log("have created data loader")

    # === 3) 主循环：对数据集中图像批量进行 反演->重建->DIRE ===
    logger.log("computing recons & DIRE ...")
    have_finished_images = 0

    # 这里以 num_samples 作为截至条件；若希望“读到数据耗尽”，可传一个足够大的数。
    while have_finished_images < args.num_samples:
        # 简化逻辑：每个进程仅处理自己这一批次上限
        remaining = args.num_samples - have_finished_images
        batch_size = min(args.batch_size, remaining)
        if batch_size <= 0:
            break

        all_images = []
        all_labels = []

        # 从数据加载器取一个 batch（imgs 是 [-1,1] 归一化后的张量）
        imgs, out_dicts, paths = next(data)
        imgs = imgs[:batch_size]
        paths = paths[:batch_size]

        # 移动到设备
        imgs = imgs.to(dist_util.dev())

        # 若是类条件模型，这里会随机生成类别标签供模型条件控制；
        # 对 LSUN-Bedroom 无条件模型，应设 class_cond=False，此分支不生效。
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes

        # 选择“反演”函数：DDIM 反演（确定性，将 x0 映射到 xT）
        reverse_fn = diffusion.ddim_reverse_sample_loop

        # 保证图像尺寸与 image_size 一致
        imgs = reshape_image(imgs, args.image_size)

        # === 3.1 DDIM 反演：x0 -> xT ===
        # 将原图 imgs 作为“起点”，走 DDIM 反演得到 latent（可视作 x_T）
        latent = reverse_fn(
            model,
            imgs.shape,
            noise=imgs,                 # 关键：把当前图当作 x0 放入反演
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            real_step=args.real_step,
        )

        # === 3.2 重建采样：xT -> x'0 ===
        # 根据参数选择：随机扩散 p_sample_loop 或 DDIM 采样 ddim_sample_loop
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop

        # 以 latent 作为初始噪声，从 xT 去噪回到重建图像 recons（x'0）
        recons = sample_fn(
            model,
            imgs.shape,
            noise=latent,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            real_step=args.real_step,
        )

        # === 3.3 D1/D2/DiffRecon with optional absolute ===
        d1_raw = imgs - recons          # 可能为负
        # 为了保证定义顺序，先计算二次反演重建
        # === 3.3' 二次反演与重建：x'0 -> x''0 ===
        latent2 = reverse_fn(
            model,
            recons.shape,
            noise=recons,                 # 将第一次重建作为“原图”做反演
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            real_step=args.real_step,
        )
        recons2 = sample_fn(
            model,
            recons.shape,
            noise=latent2,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            real_step=args.real_step,
        )
        d2_raw = recons - recons2       # 可能为负
        # 组件级原始差值（未取绝对值）
        diff_comp_raw = d1_raw - d2_raw  # 完全原始：未对分量做 abs

        dire  = th.abs(d1_raw) if args.abs_d1 else d1_raw     # D1（可能已取绝对）
        dire2 = th.abs(d2_raw) if args.abs_d2 else d2_raw     # D2（可能已取绝对）
        # 在对最终 DiffRecon 取绝对值前，先保留 pre-abs 版本（这里已包含 D1/D2 是否取绝对值的设定）
        diff_pre_abs = dire - dire2
        diff_recon = th.abs(diff_pre_abs) if getattr(args, "abs_final_diff", False) else diff_pre_abs

        # === 3.4 将张量转换为可保存的 uint8 图像 ===
        # 模型输出为 [-1,1]，先映射回 [0,255] 后保存
        recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
        recons = recons.permute(0, 2, 3, 1)  # (N,C,H,W) -> (N,H,W,C)
        recons = recons.contiguous()

        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.contiguous()

        # DIRE 可视化：支持固定或自适应放缩
        if getattr(args, "vis_mode", "adaptive") == "fixed":
            dire_u8  = to_u8_vis(dire,  vis_min_d1,  vis_max_d1)
            dire2_u8 = to_u8_vis(dire2, vis_min_d2,  vis_max_d2)
            diff_u8  = to_u8_vis(diff_recon, vis_min_diff, vis_max_diff)
        else:
            dire_u8     = to_u8_vis(dire)
            dire2_u8    = to_u8_vis(dire2)
            diff_u8     = to_u8_vis(diff_recon)

        # 第二次重建
        recons2_u8 = ((recons2 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        recons2_u8 = recons2_u8.permute(0, 2, 3, 1).contiguous()
        recons2_u8 = recons2_u8.cpu().numpy()

        # 仅根据当前进程本批次样本数推进进度
        have_finished_images += recons.shape[0]

        # === 3.6 保存当前进程生成的 recons 与 dire ===
        recons = recons.cpu().numpy()
        for i in range(len(recons)):
            if args.rank_subdir:
                rank_dir = f"rk{rank}"
            else:
                rank_dir = None

            if args.has_subfolder:
                subdir = paths[i].split("/")[-2]
                recons_save_dir  = os.path.join(args.recons_dir,  subdir)
                recons2_save_dir = os.path.join(args.recons2_dir, subdir)
                dire_save_dir    = os.path.join(args.dire_dir,    subdir)
            else:
                recons_save_dir  = args.recons_dir
                recons2_save_dir = args.recons2_dir
                dire_save_dir    = args.dire_dir

            if rank_dir is not None:
                recons_save_dir  = os.path.join(recons_save_dir,  rank_dir)
                recons2_save_dir = os.path.join(recons2_save_dir, rank_dir)
                dire_save_dir    = os.path.join(dire_save_dir,    rank_dir)

            os.makedirs(recons_save_dir, exist_ok=True)
            os.makedirs(recons2_save_dir, exist_ok=True)
            os.makedirs(dire_save_dir, exist_ok=True)

            fn_save = os.path.basename(paths[i])

            # 保存 D1 可视化
            cv2.imwrite(
                f"{dire_save_dir}/{fn_save}",
                cv2.cvtColor(dire_u8[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            # 保存第一次重建到 recons 目录
            cv2.imwrite(
                f"{recons_save_dir}/{fn_save}",
                cv2.cvtColor(recons[i].astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            # 保存第二次重建到 recons2 目录（文件名不变）
            cv2.imwrite(
                f"{recons2_save_dir}/{fn_save}",
                cv2.cvtColor(recons2_u8[i], cv2.COLOR_RGB2BGR),
            )

            # 保存第二次误差与差分重构
            if args.has_subfolder:
                dire2_save_dir = os.path.join(args.dire2_dir, paths[i].split("/")[-2])
                diff_save_dir  = os.path.join(args.diff_dir,  paths[i].split("/")[-2])
            else:
                dire2_save_dir = args.dire2_dir
                diff_save_dir  = args.diff_dir

            if rank_dir is not None:
                dire2_save_dir = os.path.join(dire2_save_dir, rank_dir)
                diff_save_dir  = os.path.join(diff_save_dir,  rank_dir)

            os.makedirs(dire2_save_dir, exist_ok=True)
            os.makedirs(diff_save_dir, exist_ok=True)

            cv2.imwrite(
                f"{dire2_save_dir}/{fn_save}",
                cv2.cvtColor(dire2_u8[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                f"{diff_save_dir}/{fn_save}",
                cv2.cvtColor(diff_u8[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR),
            )

            if getattr(args, "save_npy", False):
                base = os.path.splitext(fn_save)[0]
                # 保存到与 PNG 一致的目录结构
                np.save(os.path.join(dire_save_dir,  base + "_D1.npy"),      dire[i].cpu().float().numpy())
                np.save(os.path.join(dire2_save_dir, base + "_D2.npy"),      dire2[i].cpu().float().numpy())
                np.save(os.path.join(diff_save_dir,  base + "_DiffRecon.npy"), diff_recon[i].cpu().float().numpy())
                # 如果对应层做了绝对值，则额外保存“未取绝对值”的原始张量
                if getattr(args, "abs_d1", False):
                    np.save(os.path.join(dire_save_dir,  base + "_D1_raw.npy"), d1_raw[i].cpu().float().numpy())
                if getattr(args, "abs_d2", False):
                    np.save(os.path.join(dire2_save_dir, base + "_D2_raw.npy"), d2_raw[i].cpu().float().numpy())
        if rank == 0:
            logger.log(f"have finished {have_finished_images} samples")

    # 等待所有进程结束，打印完成日志
    _maybe_barrier()
    if rank == 0:
        logger.log("finish computing recons & DIRE!")


def create_argparser():
    """
    组装命令行参数解析器。默认值可以被命令行覆盖。

    你需要特别关注的参数：
    - images_dir: 输入真实图像目录
    - recons_dir / dire_dir: 输出目录
    - model_path: 预训练扩散模型权重（如 models/lsun_bedroom.pt）
    - image_size: 模型分辨率（LSUN-Bedroom 常用 256）
    - batch_size: 批大小
    - use_ddim: True 则使用 DDIM 采样（与反演一致性更强）
    - class_cond: 是否类条件（LSUN-Bedroom 应为 False）
    - num_samples: 处理上限（可传大数表示“直到数据耗尽”）
    """
    defaults = dict(
        images_dir="/data2/wangzd/dataset/DiffusionForensics/images",  # 输入图像根目录
        recons_dir="/data2/wangzd/dataset/DiffusionForensics/recons",  # 重建图输出目录
        recons2_dir="/data2/wangzd/dataset/DiffusionForensics/recons2",  # 第二次重建输出目录（与第一次分开）
        dire_dir="/data2/wangzd/dataset/DiffusionForensics/dire",      # DIRE图输出目录
        dire2_dir="/data2/wangzd/dataset/DiffusionForensics/dire2",    # 第二次误差输出
        diff_dir="/data2/wangzd/dataset/DiffusionForensics/diffrecon", # 差分重构输出
        clip_denoised=True,   # 是否对去噪后的图像进行裁剪（避免越界）
        num_samples=-1,       # 处理的图像总数上限；建议运行时显式传入正数或大数
        batch_size=16,        # 批大小
        use_ddim=False,       # True: 使用 DDIM 采样；False: 使用随机 p_sample_loop
        model_path="",        # .pt 权重路径（必须在运行时指定）
        real_step=0,          # 可选：控制从某个时间步开始/结束（不同实现可能有用）
        continue_reverse=False,  # 预留参数（当前脚本未使用）
        has_subfolder=False,  # 输入是否带子文件夹（如类别名），若是则按子目录保存输出
        save_npy=False,       # 是否额外保存原始差值为 .npy
        abs_d1=True,          # D1 是否取绝对值（默认 True：与原论文一致）
        abs_d2=True,          # D2 是否取绝对值（默认 True）
        abs_final_diff=False, # DiffRecon 是否对 (D1-D2) 再取绝对值（默认 False）
        rank_subdir=False,    # 是否为每个 rank 建立单独子目录（默认 False）
        vis_mode="adaptive",   # 'adaptive' 或 'fixed'：可视化放缩方式
    )
    # 合并 guided_diffusion 的默认模型/扩散配置（如 image_size, learn_sigma 等）
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
