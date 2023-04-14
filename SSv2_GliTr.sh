# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


PRETRAINED_DIR="/absolute/path/to/directory/with/pretrained/weights/"
TEACHER_CHECKPOINT="/absolute/path/to/teacher/weights/"
OUTPUT_DIR="/absolute/path/to/output/directory/"
DATA_DIR="/absolute/path/to/data/directory/"
LOG_DIR="/absolute/path/to/log/directory/"
EXP_DIR="SSv2_GliTr"

python3 -m torch.distributed.launch --use_env --nproc_per_node=4 main.py ssv2 \
    --distributed \
    --dist_eval \
    --output_dir=$OUTPUT_DIR \
    --log_dir=$LOG_DIR \
    --pretrained_dir=$PRETRAINED_DIR \
    --teacher_checkpoint=$TEACHER_CHECKPOINT \
    --exp_dir=$EXP_DIR \
    --data_dir=$DATA_DIR \
    --device='cuda' \
    --input_size=224 \
    --batch-size=90 \
    --num_segments=16 \
    --workers=8 \
    --epochs=100 \
    --base_lr_backbone=1e-5 \
    --base_lr_temphead=1e-5 \
    --base_lr_temploc=1e-5 \
    --min_lr_backbone=1e-7 \
    --min_lr_temphead=1e-6 \
    --min_lr_temploc=1e-6 \
    --seed=0 \
    --backbone='deit_small_patch16_224' \
    --weight_decay=0.05 \
    --wandbmode='online' \
    --num_patches_in_glimpse=6 \
    --attntype='student' \
    --mixup=0 \
    --cutmix=0 \
    --reprob=0 \
