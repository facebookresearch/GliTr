#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
import json
import torch
import wandb
import random
import datetime
import numpy as np
import torch.backends.cudnn as cudnn

import utils

from opts import parser
from model import VideoTransformer
from dataset_config import get_dataloaders
from engine import train_one_epoch, eval_one_epoch

from timm.data.mixup import Mixup
from timm.models import create_model
from scheduler_factory import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def main(args):

    ''' init distributed '''
    if args.distributed:
        gpu = utils.init_distributed_mode()
        args.num_tasks = utils.get_world_size()
        args.global_rank = utils.get_rank()
        torch.cuda.set_device(args.global_rank)
        torch.cuda.empty_cache()
    else:
        args.num_tasks=1
        args.global_rank=0

    print('rank {}, task {} \n'.format(args.global_rank, args.num_tasks))

    ''' init wandb '''
    if utils.is_main_process():
        wandb.init(project="GliTr", resume="allow", dir=args.log_dir, group=args.dataset, save_code=True, mode=args.wandbmode, settings=wandb.Settings(_disable_stats=True))
        wandb.config.update(args, allow_val_change=True)

    ''' log arguments '''
    if utils.is_main_process():
        if not os.path.isdir(os.path.join(args.output_dir, args.exp_dir)):
            os.mkdir(os.path.join(args.output_dir, args.exp_dir))
        with open(os.path.join(args.output_dir, args.exp_dir, 'args.json'),'a+') as f:
            json.dump(vars(args),f)
        if (args.wandbmode != 'disabled'):
            with open(os.path.join(wandb.run._settings._sync_dir,'files', 'args.json'),'a+') as f:
                json.dump(vars(args),f)

    ''' arg setup '''
    args.output_dir = os.path.join(args.output_dir, args.exp_dir)
    args.device = torch.device(args.device)

    ''' log dir '''
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write('wandb run dir '+wandb.run.dir+'\n')
            f.write('\n')
        wandb.save(os.path.join(args.output_dir, "log.txt"), policy="live")


    ''' fix seed for reproducibility '''
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
 
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ''' data loaders '''
    train_loader, val_loader = get_dataloaders(args)

    ''' mixup '''
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_class)

    ''' criterion '''
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    ''' model '''

    if ('student' in args.attntype): # and (args.dataset=='ssv2'): 
        teacher = VideoTransformer(backbone=args.backbone, num_classes=args.num_class, num_frames_per_video=args.num_segments,
                                 drop=args.dropout, drop_path=args.drop_path, num_patches_in_glimpse=args.num_patches_in_glimpse,
                                 criterion=criterion, attntype='teacher', pretrained_dir=args.pretrained_dir).to(args.device)
        teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=args.device)['model'])
        for p in teacher.parameters(): p.requires_grad = False
    elif ('teacher' in args.attntype) and (args.dataset=='ssv2'):
        teacher = create_model(
            args.backbone_teacher,
            pretrained=False,
            pretrained_dir=args.pretrained_dir,
            num_classes=args.num_class,
            all_frames=args.num_segments,
            tubelet_size=args.tubelet_size,
            drop_rate=args.dropout,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=args.init_scale,
            patches_in_glimpse=args.num_patches_in_glimpse,
            ).to(args.device)
        for p in teacher.parameters(): p.requires_grad = False
    else:
        teacher = None

    model = VideoTransformer(backbone=args.backbone, num_classes=args.num_class, num_frames_per_video=args.num_segments,
                             drop=args.dropout, drop_path=args.drop_path, num_patches_in_glimpse=args.num_patches_in_glimpse,
                             criterion=criterion, attntype=args.attntype, pretrained_dir=args.pretrained_dir, teacher=teacher).to(args.device)

    if ('teacher' in args.attntype) and (args.dataset=='jester'):
        checkpoint_model = torch.load(args.jester_teacher_pretrained_weights, map_location=args.device)['model']
        for k in ['temporal_head.head.weight', 'temporal_head.head.bias']:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

        if args.num_segments==8:
            # interpolate position embedding
            checkpoint_model['temporal_head.tmp_embed'] = checkpoint_model['temporal_head.tmp_embed'][:,::2,:]
            checkpoint_model['temploc_head.tmp_embed'] = checkpoint_model['temploc_head.tmp_embed'][:,::2,:]

        model.load_state_dict(checkpoint_model, strict=False)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    ''' optimizer '''
    backbone_lr = args.base_lr_backbone * args.batch_size * (utils.get_world_size())/ 128.0
    temphead_lr = args.base_lr_temphead * args.batch_size * (utils.get_world_size())/ 128.0
    temploc_lr  = args.base_lr_temploc  * args.batch_size * (utils.get_world_size())/ 128.0

    optimizerB = torch.optim.AdamW(model_without_ddp.backbone.parameters()     , weight_decay=args.weight_decay, lr=backbone_lr) 
    optimizerC = torch.optim.AdamW(model_without_ddp.temporal_head.parameters(), weight_decay=args.weight_decay, lr=temphead_lr) 
    optimizerL = torch.optim.AdamW(model_without_ddp.temploc_head.parameters() , weight_decay=args.weight_decay, lr=temploc_lr) 

    ''' scheduler '''
    schedulerB = create_scheduler(optimizerB, args.epochs*len(train_loader), sched='cosine', min_lr=args.min_lr_backbone, warmup_epochs=0)
    schedulerC = create_scheduler(optimizerC, args.epochs*len(train_loader), sched='cosine', min_lr=args.min_lr_temphead, warmup_epochs=0)
    schedulerL = create_scheduler(optimizerL, args.epochs*len(train_loader), sched='cosine', min_lr=args.min_lr_temploc, warmup_epochs=args.warmup_epochs*len(train_loader))

    ''' training '''
    for epoch in range(1, args.epochs+1):
        
        ''' seeding '''
        epoch_seed = args.global_rank * args.epochs + epoch
        torch.manual_seed(epoch_seed)
        torch.cuda.manual_seed(epoch_seed)
        np.random.seed(epoch_seed)
        random.seed(epoch_seed)
        train_loader.generator.manual_seed(epoch_seed)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        else:
            train_loader.sampler.generator.manual_seed(epoch_seed)

        start_time = epoch_start_routine(args.output_dir, epoch, optimizerB, optimizerC, optimizerL, args.num_segments)

        ''' train '''

        train_stats = train_one_epoch(epoch, train_loader, model, [optimizerB, optimizerC, optimizerL], [schedulerB, schedulerC, schedulerL], mixup_fn, args)
        checkpointing(args.output_dir, epoch, model_without_ddp, optimizerB, optimizerC, optimizerL, schedulerB, schedulerC, schedulerL, args.global_rank)
        logging(train_stats, epoch, args.num_segments, args.output_dir, 'train', args.num_patches_in_glimpse)

        ''' validation '''

        val_stats = eval_one_epoch(epoch, val_loader, model, args)
        logging(val_stats, epoch, args.num_segments, args.output_dir, 'eval', args.num_patches_in_glimpse)

        epoch_end_routine(args.output_dir, epoch, start_time)

    wandb.finish()


def epoch_end_routine(output_dir, epoch, start_time):
    with open(os.path.join(output_dir, "log.txt"),"a") as f:
        f.write('epoch ended at '+datetime.datetime.now().isoformat(sep='-')+'\n')
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write('Total time {}'.format(total_time_str))
        f.write('\n\n\n')


def epoch_start_routine(output_dir, epoch, optimizerB, optimizerC, optimizerL, num_segments):
    start_time = time.time()
    with open(os.path.join(output_dir, "log.txt"),"a") as f:
        f.write('epoch started at '+datetime.datetime.now().isoformat(sep='-')+'\n')
        f.write('Learning rate B for this epoch is {} \n'.format(optimizerB.param_groups[0]['lr']))
        f.write('Learning rate C for this epoch is {} \n'.format(optimizerC.param_groups[0]['lr']))
        f.write('Learning rate L for this epoch is {} \n'.format(optimizerL.param_groups[0]['lr']))
    if utils.is_main_process():
        for prefix in ['T','E']:
            wandb.log({'LR/backbone': optimizerB.param_groups[0]['lr']}, step=epoch)
            wandb.log({'LR/temphead': optimizerC.param_groups[0]['lr']}, step=epoch)
            wandb.log({'LR/temploc' : optimizerL.param_groups[0]['lr']}, step=epoch)
            for i in range(num_segments):
                wandb.define_metric('Acc/'+prefix+'_Acc_partial_'+str(i), summary="max")

    return start_time

def logging(stats, epoch, num_segments, output_dir, train_eval_mode, num_patches_in_glimpse):

    n_glimpse = 14 - num_patches_in_glimpse + 1

    prefix = 'T' if train_eval_mode=='train' else 'E'

    with open(os.path.join(output_dir, "log.txt"),"a") as f:
        f.write('Time stamp '+datetime.datetime.now().isoformat(sep='-')+'\n')

    if utils.is_main_process():
        if  prefix == 'T':
            wandb.log({'Loss/'+prefix+'_cls'     : stats['L_cls'    ]}, step=epoch)
            wandb.log({'Loss/'+prefix+'_tch'     : stats['L_tch'    ]}, step=epoch)
            wandb.log({'Loss/'+prefix+'_mse'     : stats['L_mse'    ]}, step=epoch)
            wandb.log({'Loss/'+prefix+'_kld'     : stats['L_kld'    ]}, step=epoch)
        for i in range(num_segments):
            wandb.log({'Acc/'+prefix+'_Acc_partial_'+str(i): stats['acc_partial_'+str(i)]}, step=epoch)
            try:
                wandb.log({'loc/'+prefix+'_mask_'+str(i)       : wandb.Image(stats['mask'][i].reshape(1,56,56))}, step=epoch)
            except:
                pass
            
        acc = (np.array([stats['acc_partial_'+str(i)] for i in range(num_segments)]), np.arange(num_segments+1))
        wandb.log({'Acc/'+prefix+'_hist': wandb.Histogram(np_histogram=acc)}, step=epoch)

def checkpointing(output_dir, epoch, model_without_ddp, optimizerB, optimizerC, optimizerL, schedulerB, schedulerC, schedulerL, rank):
    checkpoint_path = os.path.join(output_dir, 'checkpoint_train_'+str(epoch)+'_'+str(rank)+'.pth')
    teacher = model_without_ddp.teacher
    del model_without_ddp.teacher
    torch.save({
        'model'       : model_without_ddp.state_dict(),
        'optimizerB'   : optimizerB.state_dict(),
        'optimizerC'   : optimizerC.state_dict(),
        'optimizerL'   : optimizerL.state_dict(),
        'schedulerB'   : schedulerB.state_dict(),
        'schedulerC'   : schedulerC.state_dict(),
        'schedulerL'   : schedulerL.state_dict(),
        },
        checkpoint_path)
    model_without_ddp.teacher = teacher

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
