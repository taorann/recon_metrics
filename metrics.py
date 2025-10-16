#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算由 recon.py 生成的 dire/dire2 的指标（D1、D2、DiffRecon）。

输入：
  - dire_dir:   包含 dire 的 .npy（与源文件同名），float16，形状为 (3,H,W) 或 (C,H,W)
  - dire2_dir:  包含 dire2 的 .npy（与源文件同名），float16，形状为 (3,H,W)

输出：
  - metrics_vis/
      DiffRecon/*.png
  - metrics_floats/*.npz   （可选，D1/D2/DiffRecon 的原始浮点值）

总体流程：
 1) 扫描 dire_dir 下的所有 .npy 文件
 2) 通过相对路径镜像严格匹配 dire2_dir 下对应的 dire2 文件，若不存在则报错
 3) 加载 dire 和 dire2，转为 CHW 格式
 4) 计算 D1、D2（根据 abs_diff 是否取绝对值），计算 DiffRecon（根据 abs_final_diff 是否取绝对值）
 5) 保存 D1/D2/DiffRecon 的浮点 npz 文件（非必需）
 6) 对 DiffRecon 使用固定区间线性映射，保存 PNG 可视化（本脚本不保存 D1/D2 PNG）
 7) 打印处理进度，每 50 个样本一次
"""

import os
import re
import argparse
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image

import json
import shutil
import datetime as _dt


def str2bool(s: str) -> bool:
    """安全地将字符串解析为布尔值。支持 True/False, 1/0, yes/no, t/f 等。"""
    return str(s).lower() in {"1", "true", "t", "yes", "y"}


# // 递归列出 root 下所有 .npy 文件（稳定排序）
def list_npy_files(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".npy"):
                out.append(os.path.join(dp, fn))
    out.sort()
    return out


# // 若目录不存在则创建（幂等）
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def fixed_map_to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """线性映射 arr（C,H,W）或（H,W）到 uint8，使用固定的 [vmin, vmax]。"""
    # clip 到固定区间，避免异常值影响
    arr = np.clip(arr, vmin, vmax)
    # 线性缩放到 0-255 范围
    scale = 255.0 / (vmax - vmin)
    out = (arr - vmin) * scale
    # 转换为 uint8 类型，方便保存为图像
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 固定区间是为了样本间可比，避免每图自适应拉伸
    return out


# 自适应映射：每个样本根据自身范围映射到 uint8
def adaptive_map_to_uint8(arr: np.ndarray) -> np.ndarray:
    """自适应将 arr 映射到 uint8，按当前样本范围自动缩放。若含负值则对称到 [-amax, amax]。"""
    amax = np.max(np.abs(arr))
    # 若全为零，直接映射为 0
    if amax == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    # 若有负值，则对称映射到 [-amax, amax]
    if np.any(arr < 0):
        vmin, vmax = -amax, amax
    else:
        vmin, vmax = 0.0, amax
    return fixed_map_to_uint8(arr, vmin, vmax)

def to_chw(arr: np.ndarray) -> np.ndarray:
    """接受 (3,H,W) 或 (H,W,3) 或 (H,W)；返回颜色的 (3,H,W)，灰度的 (1,H,W)。"""
    if arr.ndim == 3:  # 可能是 CHW 或 HWC
        if arr.shape[0] in (1, 3):
            # 识别为 CHW 格式，直接返回
            return arr  # 返回 CHW
        elif arr.shape[2] in (1, 3):
            # 识别为 HWC 格式，转置为 CHW
            return arr.transpose(2, 0, 1)  # 返回 CHW
        else:
            raise ValueError(f"未知的 3D 布局: {arr.shape}")
    elif arr.ndim == 2:
        # 灰度图，增加通道维度，变为 (1,H,W)
        return arr[None, ...]  # 返回 CHW
    else:
        raise ValueError(f"不支持的数组形状 {arr.shape}")


def save_png_from_tensor(t: np.ndarray, path: str):
    """
    t: CHW uint8，C==1 或 3
    这里是数值数组最终被转换为图像并保存的地方
    """
    # 仅支持 CHW 格式，且通道数为 1 或 3；1 通道保存为 L，3 通道保存为 RGB；>3 通道只取前三个
    if t.ndim != 3:
        raise ValueError("保存时期望 CHW 格式。")
    if t.shape[0] == 1:
        img = t[0]
        im = Image.fromarray(img, mode="L")
    elif t.shape[0] == 3:
        img = t.transpose(1, 2, 0)
        im = Image.fromarray(img, mode="RGB")
    else:
        # 如果通道数 >3，只取前三个
        img = t[:3].transpose(1, 2, 0)
        im = Image.fromarray(img, mode="RGB")
    im.save(path)


# 严格镜像模式：必须保证 dire_dir 与 dire2_dir 目录结构完全一致且文件同名；否则报错
# 例如：classA/img001.npy → classA/img001.npy
def derive_partner_path(dire_path: str, dire_dir: str, dire2_dir: str) -> str:
    """
    仅使用“相对路径镜像”来定位 dire2：
      rel = relpath(dire_path, dire_dir) → partner = join(dire2_dir, rel)
    若该路径不存在，立即报错。
    """
    # 1) 相对路径镜像
    rel = os.path.relpath(dire_path, dire_dir)
    cand = os.path.join(dire2_dir, rel)
    if os.path.exists(cand):
        return cand
    else:
        raise RuntimeError(f"[错误] 未找到配对的 dire2 文件（严格镜像模式）: {cand}")


def _save_metrics_snapshot(args, out_root: str):
    # 可通过 --save_snapshot/--snapshot_dir 控制是否保存快照及存储目录
    """
    保存 metrics 运行快照：
    - args.json：参数
    - env.txt：关键环境变量
    - compute_metrics.py：脚本副本
    """
    if not getattr(args, "save_snapshot", True):
        return
    snap_dir = getattr(args, "snapshot_dir", "")
    if not snap_dir:
        snap_dir = os.path.join(out_root, "metrics_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    try:
        # 保存命令行参数 JSON
        with open(os.path.join(snap_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump({k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v)
                       for k, v in vars(args).items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[snapshot] 写 args.json 失败: {e}")
    try:
        # 保存关键环境变量及时间戳
        keys = ["CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(snap_dir, "env.txt"), "w", encoding="utf-8") as f:
            f.write(f"time: {stamp}\n")
            for k in keys:
                f.write(f"{k}={os.environ.get(k, '')}\n")
    except Exception as e:
        print(f"[snapshot] 写 env.txt 失败: {e}")
    try:
        # 复制当前脚本文件
        src = os.path.abspath(__file__)
        dst = os.path.join(snap_dir, "compute_metrics.py")
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"[snapshot] 复制 compute_metrics.py 失败: {e}")


def main():
    # // 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--dire_dir", required=True, type=str,
                        help="包含由 recon.py 生成的 dire 的 .npy 的目录（与源文件同名）")  # dire 文件目录
    parser.add_argument("--dire2_dir", required=True, type=str,
                        help="包含由 recon.py 生成的 dire2 的 .npy 的目录（与源文件同名）")  # dire2 文件目录
    parser.add_argument("--out_root", required=True, type=str,
                        help="指标输出根目录（将创建 metrics_vis/，metrics_floats/）")  # 输出根目录
    # 下面的 bool 类型参数，使用安全的字符串转布尔类型函数
    parser.add_argument("--abs_diff", type=str2bool, default=True,
                        help="是否对 D1/D2 使用绝对值（默认 True），即 D1=|dire|，范围[0,2]；否则保留符号，范围[-2,2]。支持 True/False/1/0/yes/no")  # 是否对 D1/D2 取绝对值
    parser.add_argument("--abs_final_diff", type=str2bool, default=False,
                        help="对 DiffRecon 使用绝对值（默认 False），决定 DiffRecon 映射区间。支持 True/False/1/0/yes/no")  # 是否对 DiffRecon 取绝对值
    parser.add_argument("--save_vis", type=str2bool, default=True,
                        help="保存 PNG 可视化。支持 True/False/1/0/yes/no")  # 是否保存 PNG 图像
    parser.add_argument("--save_floats", type=str2bool, default=True,
                        help="保存 D1/D2/DiffRecon 的 .npz 浮点数据。支持 True/False/1/0/yes/no")  # 是否保存浮点 npz 文件
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 个样本（0=全部）")  # 处理样本数量限制
    parser.add_argument("--save_snapshot", type=str2bool, default=True, help="是否保存运行快照（脚本+参数+环境）。支持 True/False/1/0/yes/no")  # 是否保存快照
    parser.add_argument("--snapshot_dir", type=str, default="", help="快照输出目录（留空则用 out_root 下的 run_snapshots_metrics）")  # 快照目录
    parser.add_argument("--vis_mode", type=str, default="adaptive",
                        choices=["fixed", "adaptive"],
                        help="可视化映射模式：'fixed' 为固定范围映射（默认），'adaptive' 为每样本自适应映射")
    args = parser.parse_args()

    # 准备输出目录
    vis_root = os.path.join(args.out_root, "metrics_vis")
    f32_root = os.path.join(args.out_root, "metrics_floats")
    if args.save_vis:
        # 仅创建 DiffRecon 目录，因为现在只保存 DiffRecon 的 PNG 图像
        ensure_dir(os.path.join(vis_root, "DiffRecon"))
    if args.save_floats:
        ensure_dir(f32_root)

    _save_metrics_snapshot(args, args.out_root)

    # 枚举文件
    dire_list = list_npy_files(args.dire_dir)
    # limit=0 表示全部处理，否则只取前 limit 个
    if args.limit and args.limit > 0:
        dire_list = dire_list[: args.limit]

    if len(dire_list) == 0:
        raise RuntimeError(f"dire_dir={args.dire_dir} 中未找到 .npy 文件")

    # 全局范围统计初始化（DiffRecon 的 min/max）
    global_min, global_max = float('inf'), float('-inf')

    # // 主处理循环：逐文件配对→加载→计算→保存
    for k, dire_path in enumerate(dire_list, 1):
        # 查找配对的 dire2，严格镜像模式，不存在则报错
        dire2_path = derive_partner_path(dire_path, args.dire_dir, args.dire2_dir)

        # 以 float32 计算更稳妥；再转回 float16 存 npz 降低体积
        dire = np.load(dire_path).astype(np.float32)
        dire2 = np.load(dire2_path).astype(np.float32)
        # 确保通道顺序一致以避免误用
        dire = to_chw(dire)
        dire2 = to_chw(dire2)

        # 计算 D1, D2（abs_diff=True 时取绝对值，范围[0,2]；否则保留符号，范围[-2,2]）
        D1 = np.abs(dire) if args.abs_diff else dire
        D2 = np.abs(dire2) if args.abs_diff else dire2

        # DiffRecon
        diff = D1 - D2
        if args.abs_final_diff:
            diff = np.abs(diff)

        # === 统计当前样本与全局范围 ===
        cur_min, cur_max = float(diff.min()), float(diff.max())
        global_min = min(global_min, cur_min)
        global_max = max(global_max, cur_max)
        if k == 1 or k % 50 == 0 or k == len(dire_list):
            print(f"[DEBUG] diff range of sample {k}: [{cur_min:.4f}, {cur_max:.4f}]")

        # 保存浮点数据
        # base 与源文件名一致（不依赖 _dire 后缀）
        base = os.path.splitext(os.path.basename(dire_path))[0]  # 与源文件名保持一致
        if args.save_floats:
            # 仅保存浮点原值，未做 [0,255] 映射；便于统计/阈值分析
            out_npz = os.path.join(f32_root, base + ".npz")
            np.savez_compressed(out_npz,
                                D1=D1.astype(np.float16),
                                D2=D2.astype(np.float16),
                                DiffRecon=diff.astype(np.float16))

        # 仅保存 DiffRecon 的 PNG 可视化
        if args.save_vis:
            # DiffRecon 映射范围自动选择（更精确的分支）：
            # 1) 若最终取绝对值（abs_final_diff=True）：
            #    - 若 abs_diff=True：diff = ||dire| - |dire2|| ∈ [0, 2]  → 用 [0,2]
            #    - 若 abs_diff=False：diff = |dire - dire2|     ∈ [0, 4]  → 用 [0,4]
            # 2) 若最终不取绝对值（abs_final_diff=False）：
            #    - 若 abs_diff=True： diff = |dire| - |dire2|   ∈ [-2, 2] → 用 [-2,2]
            #    - 若 abs_diff=False：diff = dire - dire2       ∈ [-4, 4] → 用 [-4,4]
            if args.vis_mode == "adaptive":
                # 自适应映射（按当前样本范围自动缩放，若含负值则对称到 [-amax, amax]）
                df_u8 = adaptive_map_to_uint8(diff)
            else:
                if args.abs_final_diff:
                    if args.abs_diff:
                        df_u8 = fixed_map_to_uint8(diff, 0.0, 2.0)  # [0,2]
                    else:
                        df_u8 = fixed_map_to_uint8(diff, 0.0, 4.0)  # [0,4]
                elif args.abs_diff:
                    df_u8 = fixed_map_to_uint8(diff, -2.0, 2.0)     # [-2,2]
                else:
                    df_u8 = fixed_map_to_uint8(diff, -4.0, 4.0)     # [-4,4]

            # 确保 CHW uint8 格式
            df_u8 = to_chw(df_u8)

            # 这里只保存 DiffRecon 的 PNG 图像，不再保存 D1/D2
            save_png_from_tensor(df_u8, os.path.join(vis_root, "DiffRecon", base + ".png"))

        # 每 50 个样本打印一次
        if k % 50 == 0:
            print(f"[{k}/{len(dire_list)}] 已处理")

    print(f"[GLOBAL] 全部样本 DiffRecon 范围统计: min={global_min:.4f}, max={global_max:.4f}")
    # 输出根路径
    print(f"完成。PNG 文件 -> {vis_root} | 浮点数据 -> {f32_root}")


if __name__ == "__main__":
    main()