#!/usr/bin/env bash
set -euo pipefail

# =========================
# 配置区（按需修改）
# =========================
# SPLITS=("test")
# DATASETS=("sd-v1")

SPLITS=("test" "val")
DATASETS=("real" "adm")

MODEL_PATH="/root/autodl-tmp/models/256x256_diffusion_uncond.pt"

BATCH=140            # 固定 batch size（不要自适应）
TIMESTEPS="ddim20"  # DDIM 步数
USE_DDIM=True       # True=ddim采样；False=随机p_sample_loop
VIS_MODE="fixed"
HAS_SUBFOLDER="True"     # 如果 IMAGES_DIR 下有类目子文件夹就设 True，否则 False

# 多卡：留空默认启用全部可见GPU；或者显式指定，比如 "0,1,2"
#: "${CUDA_VISIBLE_DEVICES:=0,1,2}"
export CUDA_VISIBLE_DEVICES=0
# PyTorch 显存碎片缓解
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

for SPLIT in "${SPLITS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        IMG_DIR="/root/autodl-tmp/data/images_imagenet/$SPLIT/imagenet/$DATASET"

        OUT_R="/root/autodl-tmp/data/first_imagenet/$VIS_MODE/$SPLIT/img_adm/recons/$DATASET"
        OUT_R2="/root/autodl-tmp/data/first_imagenet/$VIS_MODE/$SPLIT/img_adm/recons2/$DATASET"
        OUT_D1="/root/autodl-tmp/data/first_imagenet/$VIS_MODE/$SPLIT/img_adm/dire/$DATASET"
        OUT_D2="/root/autodl-tmp/data/first_imagenet/$VIS_MODE/$SPLIT/img_adm/dire2/$DATASET"
        OUT_DIFF="/root/autodl-tmp/data/first_imagenet/$VIS_MODE/$SPLIT/img_adm/diffrecon/$DATASET"

        # =========================
        # 准备
        # =========================
        mkdir -p "$OUT_R" "$OUT_R2" "$OUT_D1" "$OUT_D2" "$OUT_DIFF"

        echo "==[INFO]== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        # 为每个 (SPLIT, DATASET) 生成独立的清单文件，避免并发/覆盖
        LIST_FILE="/tmp/recon_${SPLIT}_${DATASET}.txt"

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

        echo "==[INFO]== 总图片数: $TOTAL（不切块，直接处理全部图像）"

        echo "==[RUN]== 开始处理全部图像（一次性）"
        time python recon_metrics.py \
          --images_dir "$IMG_DIR" \
          --recons_dir "$OUT_R" \
          --recons2_dir "$OUT_R2" \
          --dire_dir   "$OUT_D1" \
          --dire2_dir  "$OUT_D2" \
          --diff_dir   "$OUT_DIFF" \
          --model_path "$MODEL_PATH" \
          --has_subfolder "$HAS_SUBFOLDER" \
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
          --num_samples "$TOTAL" \
          --save_npy True \
          --abs_d1 True \
          --abs_d2 True \
          --abs_final_diff False \
          --use_fp16 True \
          --use_checkpoint False \
          --vis_mode "$VIS_MODE"

        echo "==[ALL DONE]== 全部处理完成（未使用切块）。输出目录："
        echo "  recons:   $OUT_R"
        echo "  recons2:  $OUT_R2"
        echo "  dire(D1): $OUT_D1"
        echo "  dire2(D2):$OUT_D2"
        echo "  diff:     $OUT_DIFF"

        # 清理当前任务的临时清单文件
        rm -f "$LIST_FILE"


    done
done





