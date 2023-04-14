#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Adapted from: https://github.com/mengyuest/AR-Net/blob/master/ops/dataset_config.py
"""

from os.path import join as ospj
from dataset import TSNDataSet
from sampler import RandomSubsetSampler, DistributedSamplerWrapper
from transforms import *
import utils
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from VideoMAE_transforms import aug

def return_somethingv2(data_dir):
    filename_categories = ospj(data_dir,'category.txt')
    root_data = ospj(data_dir, '20bn-something-something-v2-frames')
    filename_imglist_train = ospj(data_dir, 'train_videofolder.txt')
    filename_imglist_val = ospj(data_dir, 'val_videofolder.txt')
    prefix = '{:06d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_jester(data_dir):
    filename_categories = ospj(data_dir,'category.txt')
    root_data = ospj(data_dir, '20bn-jester-v1')
    filename_imglist_train = ospj(data_dir, 'train_videofolder.txt')
    filename_imglist_val = ospj(data_dir, 'val_videofolder.txt')
    prefix = '{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, data_dir):
    dict_single = {'ssv2': return_somethingv2, 'jester': return_jester}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](data_dir)
    else:
        raise ValueError('Unknown dataset ' + dataset)

    if isinstance(file_categories, str):
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix


def get_dataloaders(args):
    '''
    loads video frames of size: (B, T*C, H, W)
    '''
    args.num_class, args.train_list, args.val_list, args.root_path, prefix = return_dataset(args.dataset, args.data_dir)

    # Data loading code
    normalize = GroupNormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    train_transform=aug(crop_size=args.input_size, aa=args.aa, reprob=args.reprob, remode=args.remode, recount=args.recount)

    val_transform=torchvision.transforms.Compose([
        GroupScale(int(args.input_size * 256 // 224)),
        GroupCenterCrop(args.input_size),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize,
        ])

    train_dataset = TSNDataSet(args.root_path,
                        args.train_list, num_segments=args.num_segments,
                        image_tmpl=prefix,
                        transform=train_transform,
                        dataset=args.dataset,
                        num_class=args.num_class,
                        test_mode=False,
                        )

    val_dataset = TSNDataSet(args.root_path,
                      args.val_list, num_segments=args.num_segments,
                      image_tmpl=prefix,
                      random_shift=False,
                      transform=val_transform,
                      dataset=args.dataset,
                      num_class=args.num_class,
                      test_mode=True,
                      )


    subset_samples = len(train_dataset) 
    train_sampler  = RandomSubsetSampler(train_dataset, generator=torch.Generator(), subset_samples=subset_samples)
    val_sampler    = torch.utils.data.SequentialSampler(val_dataset)

    if args.distributed:
        train_sampler = DistributedSamplerWrapper(train_sampler, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=True)
    if args.distributed and args.dist_eval:
        val_sampler   = DistributedSamplerWrapper(val_sampler,   num_replicas=args.num_tasks, rank=args.global_rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
                       train_dataset,
                       sampler=train_sampler,
                       batch_size=args.batch_size,
                       num_workers=args.workers, pin_memory=True,
                       worker_init_fn=utils.seed_worker,
                       generator=torch.Generator(),
                       drop_last=False,
                       timeout=1000,
                       prefetch_factor=1,
                       )  

    val_loader = torch.utils.data.DataLoader(
                       val_dataset,
                       sampler=val_sampler,
                       batch_size=args.batch_size,
                       num_workers=args.workers, pin_memory=True,
                       worker_init_fn=utils.seed_worker,
                       generator=torch.Generator(),
                       drop_last=False,
                       timeout=1000,
                       prefetch_factor=1,
                       )


    return train_loader, val_loader
