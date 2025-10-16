#!/usr/bin/env bash
set -euo pipefail

# =========================
# 配置区（按需修改）
# =========================
IMG_DIR="/root/autodl-tmp/data/train/real"

OUT_R="/root/autodl-tmp/data/recons/real"
OUT_R2="/root/autodl-tmp/data/recons2/real"
OUT_D1="/root/autodl-tmp/data/dire/real"
OUT_D2="/root/autodl-tmp/data/dire2/real"
OUT_DIFF="/root/autodl-tmp/data/diffrecon/real"

MODEL_PATH="/root/autodl-tmp/models/lsun_bedroom.pt"

CHUNK=2000          # 每块图像数
BATCH=64            # 固定 batch size（不要自适应）
TIMESTEPS="ddim20"  # DDIM 步数
USE_DDIM=True       # True=ddim采样；False=随机p_sample_loop

# 多卡：留空默认启用全部可见GPU；或者显式指定，比如 "0,1,2"
: "${CUDA_VISIBLE_DEVICES:=0,1,2}"

# PyTorch 显存碎片缓解
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# =========================
# 准备
# =========================
mkdir -p "$OUT_R" "$OUT_R2" "$OUT_D1" "$OUT_D2" "$OUT_DIFF"

# 临时工作区（基于输入目录名做个稳定标识，避免多次跑冲突）
RUN_TAG=$(echo -n "$IMG_DIR" | sha1sum | cut -d' ' -f1)
RUN_TMP="/tmp/recon_run_${RUN_TAG}"
SHARD_DIR="${RUN_TMP}/shards"
DONE_DIR="${RUN_TMP}/done"
TODO_BASE="${RUN_TMP}/todos"
LIST_FILE="${RUN_TMP}/all_images.txt"

mkdir -p "$RUN_TMP" "$SHARD_DIR" "$DONE_DIR" "$TODO_BASE"

echo "==[INFO]== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "==[INFO]== TEMP RUN DIR: ${RUN_TMP}"

# 生成全名单
echo "==[STEP]== 扫描图像..."
find "$IMG_DIR" -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) \
  -print | sort > "$LIST_FILE"

TOTAL=$(wc -l < "$LIST_FILE" || echo 0)
if [[ "$TOTAL" -eq 0 ]]; then
  echo "!! 没找到任何图片：$IMG_DIR"
  exit 1
fi

# 切块
echo "==[STEP]== 切块，每块 ${CHUNK} 张..."
# 先清空旧分块（不删已完成标记）
find "$SHARD_DIR" -mindepth 1 -maxdepth 1 -type f -name 'shard_*' -exec rm -f {} +
split -l "$CHUNK" "$LIST_FILE" "${SHARD_DIR}/shard_"

mapfile -t SHARDS < <(ls -1 "${SHARD_DIR}"/shard_* 2>/dev/null | sort || true)
NUM_SHARDS=${#SHARDS[@]}
if [[ "$NUM_SHARDS" -eq 0 ]]; then
  echo "!! 切块失败：没有生成任何 shard_ 文件"
  exit 1
fi

echo "==[INFO]== 总图片数: $TOTAL, 分块数: $NUM_SHARDS, 每块: $CHUNK"

# =========================
# 分块循环
# =========================
i=0
for f in "${SHARDS[@]}"; do
  i=$((i+1))
  base=$(basename "$f")
  done_mark="${DONE_DIR}/${base}.done"

  N=$(wc -l < "$f")
  # 已完成则跳过
  if [[ -f "$done_mark" ]]; then
    done_imgs=$(( i*CHUNK < TOTAL ? i*CHUNK : TOTAL ))
    echo "--[SKIP] 已完成：第 $i/$NUM_SHARDS 块（$N 张），累计 ${done_imgs}/${TOTAL}"
    continue
  fi

  # 准备 TODO 目录（软链接本块图片）
  TODO="${TODO_BASE}/${base}"
  rm -rf "$TODO"; mkdir -p "$TODO"
  while IFS= read -r p; do ln -sf "$p" "$TODO/$(basename "$p")"; done < "$f"

  echo "==[RUN]== 第 $i/$NUM_SHARDS 块，当前块 $N 张，累计目标 ${TOTAL} 张"
  # 使用 bash 内置 time（/usr/bin/time 在部分系统可能没有）
  time python compute_dire.py \
    --images_dir "$TODO" \
    --recons_dir "$OUT_R" \
    --recons2_dir "$OUT_R2" \
    --dire_dir   "$OUT_D1" \
    --dire2_dir  "$OUT_D2" \
    --diff_dir   "$OUT_DIFF" \
    --model_path "$MODEL_PATH" \
    --image_size 256 \
    --num_channels 256 \
    --num_res_blocks 2 \
    --channel_mult "1,1,2,2,4,4" \
    --attention_resolutions "32,16,8" \
    --num_heads -1 \
    --num_head_channels 64 \
    --resblock_updown True \
    --use_scale_shift_norm True \
    --learn_sigma True \
    --class_cond False \
    --clip_denoised True \
    --use_ddim "$USE_DDIM" \
    --timestep_respacing "$TIMESTEPS" \
    --batch_size "$BATCH" \
    --num_samples "$N" \
    --save_npy False \
    --abs_d1 True \
    --abs_d2 True \
    --abs_final_diff False \
    --use_fp16 True \
    --use_checkpoint False

  # 标记完成 & 清理
  touch "$done_mark"
  rm -rf "$TODO"

  done_imgs=$(( i*CHUNK < TOTAL ? i*CHUNK : TOTAL ))
  echo "==[DONE]== 第 $i/$NUM_SHARDS 块完成。累计完成 ${done_imgs}/${TOTAL} 张。"
done

echo "==[ALL DONE]== 全部分块处理完成。输出目录："
echo "  recons:   $OUT_R"
echo "  recons2:  $OUT_R2"
echo "  dire(D1): $OUT_D1"
echo "  dire2(D2):$OUT_D2"
echo "  diff:     $OUT_DIFF"