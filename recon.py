"""
Modified from guided-diffusion/scripts/image_sample.py

本脚本针对**真实图像的两次重建**，并**在重建阶段直接产出非绝对值差分**用于后续指标计算与可视化（避免离线重复预处理与数值误差）：

流程：
1) DDIM 反演：x0 → x_T
2) 第一次重建：x_T → x'0
3) 二次重建：x'0 → x''0
4) 直接计算并保存（非绝对值，保持方向性）：
   - dire  = x0  - x'0     ∈ [-2, 2]
   - dire2 = x'0 - x''0    ∈ [-2, 2]

存储策略：
- 高频使用（便于下游 metrics 快速加载）：
  - dire / dire2 分别保存为 `.npy`（float32），默认目录 `../dire` 与 `../dire2`，可通过 `--dire_dir / --dire2_dir` 指定。
- 归档/追溯（占空间较大、低频）：
  - recons / recons2（以及可选 latent/latent2、元信息 path 等）打包保存为压缩 `.npz`（float32）到 `../floats`，可通过 `--floats_dir` 指定。
- 本脚本**不再输出 PNG/JPG 图像**；可视化应在后续指标脚本中进行，并统一使用固定区间映射（如 |·| 指标用 [0,2]→[0,255]；有符号图用 [-2,2]→[0,255]），避免逐图自适应拉伸造成不可比性。

说明：
- `.pt` 权重文件仅包含参数，模型结构由 guided_diffusion 代码创建。
- 对 LSUN-Bedroom(Real) 需使用在该数据集训练的 ADM 权重。
- 反演与重建均调用 diffusion 对象的 `ddim_reverse_sample_loop` / `ddim_sample_loop` 实现。

主要新增/常用参数：
- `--save_dire_npy`：是否保存 dire/dire2 的 `.npy`（默认 True）
- `--dire_dir / --dire2_dir`：dire/dire2 输出目录
- `--save_npz` / `--floats_dir`：是否与存放 `.npz` 的目录
- `--save_latent`：是否将反演得到的 latent 一并保存到 `.npz`
"""

import argparse
import os
import torch


import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th
import torch.distributed as dist

# NOTE: New flags to control saving

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


# def count_images_in_dir(root: str) -> int:
#     """
#     递归统计 root 目录下的图像文件数量（支持常见后缀）。
#     """
#     exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
#     total = 0
#     for dirpath, _, filenames in os.walk(root):
#         for fn in filenames:
#             if fn.lower().endswith(exts):
#                 total += 1
#     return total




def main():
    # 解析命令行参数（包含数据路径、输出路径、模型配置等）
    args = create_argparser().parse_args()
    # New runtime flags are parsed in create_argparser()

    # # 若未显式指定，自动将 num_samples 设为 images_dir 下图片总数
    # if getattr(args, "num_samples", -1) is None or args.num_samples <= 0:
    #     auto_count = count_images_in_dir(args.images_dir)
    #     # 若目录为空，保持为 0；否则设为统计值
    #     args.num_samples = max(0, int(auto_count))
    # 检查是否存在 num_samples 参数
    if not hasattr(args, "num_samples"):
        raise AttributeError("参数缺失：请在命令行中指定 --num_samples，或在代码中添加该字段。")

    # 初始化分布式/设备
    dist_util.setup_dist(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    # 标记是否处于分布式环境，并取得 rank/world
    # is_dist_env = dist.is_available() and dist.is_initialized()
    rank, world = _dist_info()

    # 配置日志输出目录（将日志写到重建图的输出目录）
    if rank == 0:
        logger.configure(dir=args.recons_dir)
        logger.log(f"[auto] num_samples = {args.num_samples} (from images_dir='{args.images_dir}')")

    # 创建重建输出目录
    os.makedirs(args.recons_dir, exist_ok=True)
    if hasattr(args, "recons2_dir") and args.recons2_dir:
        os.makedirs(args.recons2_dir, exist_ok=True)
    else:
        args.recons2_dir = os.path.join(args.recons_dir, "..", "recons2")
        os.makedirs(args.recons2_dir, exist_ok=True)

    # New: directory for saving raw floats ([-1,1]) as .npz
    if getattr(args, "save_npz", False):
        if hasattr(args, "floats_dir") and args.floats_dir:
            os.makedirs(args.floats_dir, exist_ok=True)
        else:
            args.floats_dir = os.path.join(args.recons_dir, "..", "floats")
            os.makedirs(args.floats_dir, exist_ok=True)

    # New: directories for saving dire/dire2 as .npy
    if getattr(args, "save_dire_npy", True):
        if hasattr(args, "dire_dir") and args.dire_dir:
            os.makedirs(args.dire_dir, exist_ok=True)
        else:
            args.dire_dir = os.path.join(args.recons_dir, "..", "dire")
            os.makedirs(args.dire_dir, exist_ok=True)
        if hasattr(args, "dire2_dir") and args.dire2_dir:
            os.makedirs(args.dire2_dir, exist_ok=True)
        else:
            args.dire2_dir = os.path.join(args.recons2_dir, "..", "dire2")
            os.makedirs(args.dire2_dir, exist_ok=True)

    # 打印当前运行参数，方便复现实验（仅主进程）
    if rank == 0:
        logger.log(str(args))

    # === 1) 创建模型与扩散过程（模型结构在代码里，参数随后从 .pt 加载） ===
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # model：神经网络（通常是 U-Net）；diffusion：扩散过程，不学习参数

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
    logger.log("computing reconstructions ...")
    have_finished_images = 0

    # 这里以 num_samples 作为截至条件；若希望“读到数据耗尽”，可传一个足够大的数。
    while have_finished_images < args.num_samples:
        # 简化逻辑：每个进程仅处理自己这一批次上限
        remaining = args.num_samples - have_finished_images
        batch_size = min(args.batch_size, remaining)
        if batch_size <= 0:
            break

        # 从数据加载器取一个 batch（imgs 是 [-1,1] 归一化后的张量）
        try:
            imgs, out_dicts, paths = next(data)
        except StopIteration:
            break
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
        # 根据参数选择：use_ddim=False--随机扩散 p_sample_loop 或 use_ddim=True--DDIM 采样 ddim_sample_loop
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

        # === 3.4 Compute raw (non-abs) differences for fast later use ===
        # dire = x0 - x'0, dire2 = x'0 - x''0, all in [-2,2]
        imgs_float   = imgs.detach().clone()      # [-1,1]
        recons_float = recons.detach().clone()    # [-1,1]
        recons2_float= recons2.detach().clone()   # [-1,1]
        dire_batch  = (imgs_float - recons_float).detach()
        dire2_batch = (recons_float - recons2_float).detach()

        # Optionally keep latents if requested
        latent_float  = latent.detach().clone() if getattr(args, "save_latent", False) else None
        latent2_float = latent2.detach().clone() if getattr(args, "save_latent", False) else None

        # 仅根据当前进程本批次样本数推进进度
        have_finished_images += recons.shape[0]

        # === 3.6 保存当前进程生成的 recons ===
        # 保存当前 batch 的结果
        bs = recons.size(0)
        for i in range(bs):

            if args.rank_subdir:
                rank_dir = f"rk{rank}"
            else:
                rank_dir = None
            # Floats (npz) root
            floats_root = args.floats_dir if getattr(args, "save_npz", False) else None
            fn_save = os.path.basename(paths[i])

            # Floats directory (npz)
            if getattr(args, "save_npz", False):
                if args.has_subfolder:
                    floats_save_dir = os.path.join(floats_root, paths[i].split("/")[-2])
                else:
                    floats_save_dir = floats_root
                if rank_dir is not None:
                    floats_save_dir = os.path.join(floats_save_dir, rank_dir)
                os.makedirs(floats_save_dir, exist_ok=True)
                base = os.path.splitext(fn_save)[0]
                out_npz = os.path.join(floats_save_dir, base + ".npz")
                # Prepare numpy arrays in float32 for numerical fidelity
                pkg = {
                    "recons":  recons_float[i].cpu().numpy().astype(np.float32),
                    "recons2": recons2_float[i].cpu().numpy().astype(np.float32),
                    "path":    paths[i],
                    "image_size": np.int32(args.image_size),
                    "real_step":  np.int32(args.real_step),
                    "use_ddim":   np.bool_(args.use_ddim),
                }
                if getattr(args, "save_latent", False):
                    pkg["latent"]  = latent_float[i].cpu().numpy().astype(np.float32)
                    pkg["latent2"] = latent2_float[i].cpu().numpy().astype(np.float32)
                np.savez_compressed(out_npz, **pkg)

            # Save dire/dire2 as .npy (non-abs), high-frequency use
            if getattr(args, "save_dire_npy", True):
                # Build per-sample output directories mirroring subfolders/ranks
                if args.has_subfolder:
                    dire_save_dir = os.path.join(args.dire_dir,  paths[i].split("/")[-2])
                    dire2_save_dir= os.path.join(args.dire2_dir, paths[i].split("/")[-2])
                else:
                    dire_save_dir = args.dire_dir
                    dire2_save_dir= args.dire2_dir
                if rank_dir is not None:
                    dire_save_dir  = os.path.join(dire_save_dir,  rank_dir)
                    dire2_save_dir = os.path.join(dire2_save_dir, rank_dir)
                os.makedirs(dire_save_dir, exist_ok=True)
                os.makedirs(dire2_save_dir, exist_ok=True)

                base = os.path.splitext(fn_save)[0]
                # 文件名与源文件保持一致，仅目录不同，便于一一对应
                out_dire_npy  = os.path.join(dire_save_dir,  base + ".npy")
                out_dire2_npy = os.path.join(dire2_save_dir, base + ".npy")

                # Use float32 for .npy for fast loading; keep raw sign (no abs)
                np.save(out_dire_npy,  dire_batch[i].cpu().numpy().astype(np.float32))
                np.save(out_dire2_npy, dire2_batch[i].cpu().numpy().astype(np.float32))

        if rank == 0:
            pct = 100.0 * have_finished_images / max(1, int(args.num_samples))
            logger.log(f"have finished {have_finished_images} / {args.num_samples} samples ({pct:.1f}%)")

    # 等待所有进程结束，打印完成日志
    _maybe_barrier()
    if rank == 0:
        logger.log("finish computing reconstructions!")


def create_argparser():
    """
    组装命令行参数解析器。默认值可以被命令行覆盖。

    你需要特别关注的参数：
    - images_dir: 输入真实图像目录
    - recons_dir / recons2_dir: 输出目录
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
        clip_denoised=True,   # 是否对去噪后的图像进行裁剪（避免越界）
        num_samples=-1,       # 处理的图像总数上限；建议运行时显式传入正数或大数
        batch_size=16,        # 批大小
        use_ddim=False,       # True: 使用 DDIM 采样；False: 使用随机 p_sample_loop
        model_path="",        # .pt 权重路径（必须在运行时指定）
        real_step=0,          # 可选：控制从某个时间步开始/结束（不同实现可能有用）
        continue_reverse=False,  # 预留参数（当前脚本未使用）
        has_subfolder=False,  # 输入是否带子文件夹（如类别名），若是则按子目录保存输出
        rank_subdir=False,    # 是否为每个 rank 建立单独子目录（默认 False）
        # Only keep float saving controls
        save_npz=True,        # default: save float-domain arrays for offline processing
        floats_dir="/data2/wangzd/dataset/DiffusionForensics/floats",
        save_latent=False,    # optionally persist latents
        save_dire_npy=True,   # save raw (non-abs) dire/dire2 as .npy for fast later use
        dire_dir="/data2/wangzd/dataset/DiffusionForensics/dire",
        dire2_dir="/data2/wangzd/dataset/DiffusionForensics/dire2",
    )
    # 合并 guided_diffusion 的默认模型/扩散配置（如 image_size, learn_sigma 等）
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
