#!/usr/bin/env bash
# ============================================================
# 固定配置版：运行 compute_metrics.py
# 适配 Zoe 的目录结构（dire / dire2 分别为 recon 结果）
# ============================================================

set -euo pipefail

#############################
# 配置区（按需修改）
#############################

# Python 解释器
# PYTHON_BIN="python3"

# 数据路径（你自己的）
DIRE_ROOT="/root/autodl-tmp/data/our_no_no/dire"
DIRE2_ROOT="/root/autodl-tmp/data/our_no_no/dire2"

# 输出根目录（自动分子目录保存）
OUT_ROOT="/root/autodl-tmp/data/history/adaptive/diff_abs_no"

# 是否取绝对值与保存选项
ABS_DIFF="True"
ABS_FINAL_DIFF="False"
SAVE_VIS="True"
SAVE_FLOATS="True"
LIMIT="0"

# 是否保存快照
SAVE_SNAPSHOT="True"
SNAPSHOT_DIR=""

#############################
# 循环跑 train/val/test × {real, adm}
#############################
for SPLIT in test; do
  for DOMAIN in ddpm; do
    DIRE_DIR="${DIRE_ROOT}/${SPLIT}/lsun_bedroom/${DOMAIN}"
    DIRE2_DIR="${DIRE2_ROOT}/${SPLIT}/lsun_bedroom/${DOMAIN}"
    OUT_DIR="${OUT_ROOT}/${SPLIT}/lsun_adm/${DOMAIN}"

    echo ">>> Running ${SPLIT}/${DOMAIN} ..."
    python compute_metrics.py \
      --dire_dir "$DIRE_DIR" \
      --dire2_dir "$DIRE2_DIR" \
      --out_root "$OUT_DIR" \
      --abs_diff "$ABS_DIFF" \
      --abs_final_diff "$ABS_FINAL_DIFF" \
      --save_vis "$SAVE_VIS" \
      --save_floats "$SAVE_FLOATS" \
      --limit "$LIMIT" \
      --save_snapshot "$SAVE_SNAPSHOT" \
      --snapshot_dir "$SNAPSHOT_DIR" \
      --vis_mode "adaptive"
  done
done

echo "[DONE] 所有 split-domain 计算完成，结果在 $OUT_ROOT/"