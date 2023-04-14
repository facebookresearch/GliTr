# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


PRETRAINED_DIR="/absolute/path/to/directory/with/pretrained/weights/"
OUTPUT_DIR="/absolute/path/to/output/directory/"
DATA_DIR="/absolute/path/to/data/directory/"
LOG_DIR="/absolute/path/to/log/directory/"
EXP_DIR="SSv2_Teacher"

python3 -m torch.distributed.launch --use_env --nproc_per_node=4 main.py ssv2 \
    --distributed \
    --dist_eval \
    --output_dir=$OUTPUT_DIR \
    --log_dir=$LOG_DIR \
    --exp_dir=$EXP_DIR \
    --pretrained_dir=$PRETRAINED_DIR \
    --data_dir=$DATA_DIR \
    --device='cuda' \
    --batch-size=15 \
    --num_segments=16 \
    --workers=8 \
    --epochs=40 \
    --warmup_epochs=15 \
    --base_lr_backbone=1e-5 \
    --base_lr_temphead=1e-4 \
    --base_lr_temploc=1e-4 \
    --min_lr_backbone=1e-7 \
    --min_lr_temphead=1e-6 \
    --min_lr_temploc=1e-6 \
    --seed=0 \
    --backbone='deit_small_patch16_224' \
    --backbone_teacher='vit_base_patch16_224' \
    --weight_decay=0.05 \
    --wandbmode='online' \
    --num_patches_in_glimpse=6 \
    --attntype='teacher' \
    --print-freq=10 \

