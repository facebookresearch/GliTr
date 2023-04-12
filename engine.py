import os
import torch
import torchvision
from torch.nn import functional as F

import utils

def train_one_epoch(epoch, train_loader, model, optimizers, schedulers, mixup_fn, args):

    model.train()
    metric_logger         = utils.MetricLogger(args.output_dir, delimiter="  ")
    metric_logger_noprint = utils.MetricLogger(args.output_dir, delimiter="  ")
    model_without_ddp     = model.module if args.distributed else model
    header                = 'Train Epoch: [{}]'.format(epoch)

    mask = 0
    counter = 0
    for data, label_1hot in metric_logger.log_every(train_loader, args.print_freq, header):
        counter += 1
    
        data, label_1hot = data.to(args.device), label_1hot.to(args.device)
        data = data.reshape(-1, args.num_segments, 3, args.input_size, args.input_size)
        B = data.size(0)

        if mixup_fn is not None:
            if (B%2)!=0:
                data, label_1hot = data[:-1], label_1hot[:-1]
                B = data.size(0)
            data, label_1hot = mixup_fn(data.flatten(1,2), label_1hot.long().argmax(-1))
            data = data.reshape(B, args.num_segments, 3, args.input_size, args.input_size)

        model_without_ddp.epoch = epoch
        prediction, cls, tch, mse, kld, pmask = model(data, label_1hot)
        for opt in optimizers: opt.zero_grad()
        (cls+tch+mse+kld).backward()
        for opt in optimizers: opt.step()
        if not (args.device==torch.device('cpu')): torch.cuda.synchronize()

        for i in range(args.num_segments):
            acc_partial = (prediction[i].argmax(-1) == label_1hot.float().argmax(-1)).float().mean() * 100
            metric_logger_noprint.meters['acc_partial_'+str(i)].update(acc_partial.item(), n=B)

        metric_logger.meters['L_cls'].update(cls.item()   , n=B)
        metric_logger.meters['L_tch'].update(tch.item()   , n=B)
        metric_logger.meters['L_mse'].update(mse.item()   , n=B) 
        metric_logger.meters['L_kld'].update(kld.item()   , n=B) 
        metric_logger.meters['acc_partial'].update(acc_partial.item() , n=B)
        for sched in schedulers: sched.step((epoch-1)*len(train_loader) + counter)

        del prediction, cls, tch, mse, kld, pmask


    metric_logger.synchronize_between_processes()
    metric_logger_noprint.synchronize_between_processes()

    with open(os.path.join(args.output_dir,"log.txt"), "a") as f:
        f.write("Average stats:")
        f.write(str(metric_logger))
        f.write('\n')
        f.write(str(metric_logger_noprint))
        f.write('\n\n')

    print('Epoch {} complete'.format(epoch))
    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()}, **{k: meter.global_avg for k, meter in metric_logger_noprint.meters.items()}, **{'mask': mask}}


@torch.no_grad()
def eval_one_epoch(epoch, val_loader, model, args):
    model.eval()
    metric_logger         = utils.MetricLogger(args.output_dir, delimiter="  ")
    metric_logger_noprint = utils.MetricLogger(args.output_dir, delimiter="  ")
    header = 'Eval Epoch: [{}]'.format(epoch)
    model_without_ddp = model.module if args.distributed else model

    mask = 0
    for data, label_1hot in metric_logger.log_every(val_loader, args.print_freq, header):

        data, label_1hot = data.to(args.device), label_1hot.to(args.device)
        data = data.reshape(-1, args.num_segments, 3, args.input_size, args.input_size)
        B = data.size(0)

        model_without_ddp.epoch = epoch
        prediction, cls, tch, mse, kld, pmask = model(data, label_1hot)

        if not (args.device==torch.device('cpu')): torch.cuda.synchronize()

        for i in range(args.num_segments):
            acc_partial = (prediction[i].argmax(-1) == label_1hot.float().argmax(-1)).float().mean() * 100
            metric_logger_noprint.meters['acc_partial_'+str(i)].update(acc_partial.item(), n=B)
    
        metric_logger.meters['L_cls'].update(cls.item() , n=B)
        metric_logger.meters['acc_partial'].update(acc_partial.item() , n=B)
        if not (pmask is None): mask = mask + F.interpolate(pmask, scale_factor=0.25).sum(0)

    if not (pmask is None):
        n = min(B,16)
        visualize(data[:n], pmask[:n], args, epoch)

    metric_logger.synchronize_between_processes()
    metric_logger_noprint.synchronize_between_processes()

    with open(os.path.join(args.output_dir,"log.txt"), "a") as f:
        f.write("Average stats:")
        f.write(str(metric_logger))
        f.write('\n')
        f.write(str(metric_logger_noprint))
        f.write('\n\n')

    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()}, **{k: meter.global_avg for k, meter in metric_logger_noprint.meters.items()}, **{'mask':mask}}


def visualize(data, pmask, args, epoch, idx=None):
    B = data.size(0)
    data = data.reshape(B, args.num_segments, 3, args.input_size, args.input_size)
    IMAGENET_DEFAULT_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,1,3,1,1).to(data.device)
    IMAGENET_DEFAULT_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,1,3,1,1).to(data.device)
    data = (data * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
    data = (data*255).type(torch.uint8)
    img = torch.cat([data, data*pmask[:,:,None,:,:].to(data.device)],1).flatten(0,1)
    if idx is None:
        torchvision.utils.save_image(img.float(), os.path.join(args.output_dir,'glimpse_ep'+str(epoch)+'_r'+str(utils.get_rank())+'.png'), nrow=args.num_segments, normalize=True)
    else:
        torchvision.utils.save_image(img.float(), os.path.join(args.output_dir,'glimpse_ep'+str(epoch)+'_'+str(idx)+'_r'+str(utils.get_rank())+'.png'), nrow=args.num_segments, normalize=True)

   
