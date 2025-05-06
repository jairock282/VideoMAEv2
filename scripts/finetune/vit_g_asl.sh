#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR="/mnt/HDD/models/outputs/video_mae2_test/"
DATA_PATH="/mnt/HDD/datasets/mae_v2/splits"
MODEL_PATH="/mnt/HDD/models/model_zoo/vit_g_hybrid_pt_1200e_k710_ft.pth"

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
python run_class_finetuning.py \
  --model vit_small_patch16_224 \
  --data_set MSASL121 \
  --nb_classes 121 \
  --data_path ${DATA_PATH} \
  --finetune ${MODEL_PATH} \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 10 \
  --num_sample 2 \
  --input_size 224 \
  --short_side_size 224 \
  --save_ckpt_freq 10 \
  --num_frames 16 \
  --sampling_rate 2 \
  --opt adamw \
  --lr 5e-4 \
  --layer_decay 0.90 \
  --num_workers 10 \
  --opt_betas 0.9 0.999 \
  --weight_decay 0.05 \
  --epochs 100 \
  --drop_path 0.35 \
  --head_drop_rate 0.5 \
  --test_num_segment 5 \
  --test_num_crop 3 \
  --dist_eval --enable_deepspeed \
  ${PY_ARGS}
