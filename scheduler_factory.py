#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
 
Scheduler Factory adapted from: 
https://github.com/rwightman/pytorch-image-models/ 
"""
from timm.scheduler.cosine_lr import CosineLRScheduler


def create_scheduler(optimizer, epochs, sched='cosine', lr_cycle_mul=1.0, min_lr=1e-5, decay_epochs=30, decay_rate=0.1, warmup_lr=1e-6, warmup_epochs=0, lr_cycle_limit=1, lr_noise_pct=0.67, lr_noise_std=1.0, seed=42, cooldown_epochs=10, patience_epochs=10):

    num_epochs = epochs

    noise_range = None

    lr_scheduler = None
    if sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=lr_cycle_mul,
            lr_min=min_lr,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            cycle_limit=lr_cycle_limit,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=lr_noise_pct,
            noise_std=lr_noise_std,
            noise_seed=seed,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs
    return lr_scheduler
