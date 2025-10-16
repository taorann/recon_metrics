#!/usr/bin/env bash
set -euo pipefail

# ===== 基本配置 =====
SPLIT_in="test/lsun_bedroom/pndm"
SPLIT="test/lsun_adm/pndm"

IMAGES_DIR="/root/autodl-tmp/data/images/${SPLIT_in}"   # 输入图像根目录（可含子文件夹）
MODEL_PATH="/root/autodl-tmp/models/lsun_bedroom.pt"    # ADM 权重
OUT_ROOT="/root/autodl-tmp/data/our_no_no"              # 输出根

DIRE_DIR="${OUT_ROOT}/dire/${SPLIT}"
DIRE2_DIR="${OUT_ROOT}/dire2/${SPLIT}"

# 仅用于保存中间重建（npz），不再写 PNG
RECONS_DIR="${OUT_ROOT}/recons/${SPLIT}"
RECONS2_DIR="${OUT_ROOT}/recons2/${SPLIT}"
FLOATS_DIR="${OUT_ROOT}/floats/${SPLIT}"

BATCH_SIZE=135
TIMESTEP_RESPACING="ddim20"
HAS_SUBFOLDER="False"     # 如果 IMAGES_DIR 下有类目子文件夹就设 True，否则 False
RANK_SUBDIR="False"       # 多卡时是否按 rank 分子目录保存 npz；不需要可 False
SAVE_LATENT="False"       # 需要保存 latent/latent2 再改为 True
USE_FP16="True"
USE_CHECKPOINT="False"

mkdir -p "${RECONS_DIR}" "${RECONS2_DIR}" "${FLOATS_DIR}" "${DIRE_DIR}" "${DIRE2_DIR}"

# 统计总数
N_TOTAL=$(
  find "$IMAGES_DIR" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) \
    | wc -l | tr -d ' '
)
echo "[INFO] 输入总数: $N_TOTAL"
if [[ "$N_TOTAL" -eq 0 ]]; then
  echo "[ERROR] ${IMAGES_DIR} 下没有找到图片（jpg/jpeg/png/bmp/webp）。退出。"
  exit 1
fi

# 选择 time 工具
if command -v /usr/bin/time >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -v)
else
  TIME_CMD=(time)
fi

echo "[RUN] 一次性处理全部 ${N_TOTAL} 张图片"

# 采样/扩散相关
SAMPLE_FLAGS=(
  --batch_size "${BATCH_SIZE}"
  --num_samples "${N_TOTAL}"
  --timestep_respacing "${TIMESTEP_RESPACING}"
  --use_ddim True            # 与 compute_dire 一致，重建确定性
)

# I/O 与保存（recon.py 内部已统一为 float32 保存，不需要你在脚本中设置 dtype）
SAVE_FLAGS=(
  --images_dir "${IMAGES_DIR}"
  --recons_dir "${RECONS_DIR}"
  --recons2_dir "${RECONS2_DIR}"
  --floats_dir "${FLOATS_DIR}"
  --has_subfolder "${HAS_SUBFOLDER}"
  --rank_subdir "${RANK_SUBDIR}"
  --save_npz True
  --save_latent "${SAVE_LATENT}"
  --save_dire_npy True
  --dire_dir "${DIRE_DIR}"
  --dire2_dir "${DIRE2_DIR}"
)

# 模型结构/推理精度
MODEL_FLAGS=(
  --model_path "${MODEL_PATH}"
  --image_size 256
  --num_channels 256
  --num_res_blocks 2
  --dropout 0.1
  --channel_mult "1,1,2,2,4,4"
  --attention_resolutions "32,16,8"
  --num_heads -1
  --num_head_channels 64
  --resblock_updown True
  --use_scale_shift_norm True
  --diffusion_steps 1000
  --noise_schedule linear
  --learn_sigma True
  --class_cond False
  --clip_denoised True
  --use_fp16 "${USE_FP16}"
  --use_checkpoint "${USE_CHECKPOINT}"
)

"${TIME_CMD[@]}" python recon.py \
  "${MODEL_FLAGS[@]}" \
  "${SAVE_FLAGS[@]}" \
  "${SAMPLE_FLAGS[@]}"

# # ====== 所有循环结束后再关机 ======
# echo "[ALL DONE] 所有任务完成，准备关机..."
# /usr/bin/shutdown
