# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


PRETRAINED_DIR="/absolute/path/to/directory/with/pretrained/weights/"
JESTER_PRETRAINED="/absolute/path/to/learnt/ssv2/teacher/weights/"
OUTPUT_DIR="/absolute/path/to/output/directory/"
DATA_DIR="/absolute/path/to/data/directory/"
LOG_DIR="/absolute/path/to/log/directory/"
EXP_DIR="Jester_Teacher"

python3 -m torch.distributed.launch --use_env --nproc_per_node=4 main.py jester \
        --distributed \
        --dist_eval \
        --device='cuda' \
	--output_dir=$OUTPUT_DIR \
        --log_dir=$LOG_DIR \
	--pretrained_dir=$PRETRAINED_DIR \
        --exp_dir=$EXP_DIR \
        --data_dir=$DATA_DIR \
	--jester_teacher_pretrained_weights=$JESTER_PRETRAINED \
        --batch-size=25 \
        --num_segments=8 \
        --workers=8 \
        --epochs=50 \
        --warmup_epochs=0 \
        --base_lr_backbone=1e-5 \
        --base_lr_temphead=1e-5 \
        --base_lr_temploc=1e-5 \
        --min_lr_backbone=1e-7 \
        --min_lr_temphead=1e-7 \
        --min_lr_temploc=1e-7 \
        --seed=0 \
        --backbone='deit_small_patch16_224' \
        --weight_decay=0.05 \
        --num_patches_in_glimpse=6 \
        --attntype='teacher' \
        --wandbmode='online' \
        --print-freq=10 \
