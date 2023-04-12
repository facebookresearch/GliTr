'''
adapted from: https://github.com/mengyuest/AR-Net/blob/master/opts.py
'''
import argparse

def none_or_str(value):
    if (value == 'None') or (value =='none'):
        return None
    return value

def none_or_int(value):
    if (value == 'None') or (value =='none'):
        return None
    return int(value)

parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str)

# ========================== paths ===================

parser.add_argument('--exp_dir', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--pretrained_dir', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--input_size', default=224, type=int)
parser.add_argument('--num_segments', type=int, default=16)

# ========================= Reproducibility ==========================
parser.add_argument('--seed', type=int, default=0)

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')  # TODO(changed from 120 to 50)
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--min_lr_backbone', default=1e-7, type=float)
parser.add_argument('--min_lr_temphead', default=1e-6, type=float)
parser.add_argument('--min_lr_temploc',  default=1e-6, type=float)
parser.add_argument('--base_lr_backbone', '--learning-rate-backbone', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--base_lr_temphead', '--learning-rate-temporal-head', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--base_lr_temploc', '--learning-rate-temploc-head', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  # TODO(changed from 5e-4 to 1e-4)
parser.add_argument('--warmup_epochs', type=int, default=0)

# ========================= WandB ======================
parser.add_argument('--wandbmode', type=str, default="online", help='wandb mode can be "online", "offline" or "disabled"')

# ========================= Bookkeeping ==========================
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')

# ===================== Distributed =============================
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--dist_eval', default=False, action="store_true")
parser.add_argument('--distributed', default=False, action="store_true")
parser.add_argument('--device', type=str, default='cpu')

# ====================== Transformer ============================
parser.add_argument('--backbone', type=str, default='deit_small_patch16_224')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--drop_path', type=float, default=0.1)
parser.add_argument('--num_patches_in_glimpse', type=none_or_int, default=3)

# ============================ Teacher-Student ==========================
parser.add_argument('--attntype', type=str, choices=["teacher", "student"])
parser.add_argument('--teacher_checkpoint', type=str, default="none")
parser.add_argument('--jester_teacher_pretrained_weights', type=str, default="none")

# ========================= Mixup param =======================
parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)')

# ============================== VideoMAE =============================
parser.add_argument('--tubelet_size', type=int, default=2)
parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='Attention dropout rate (default: 0.)')
parser.add_argument('--init_scale', default=0.1, type=float)
parser.add_argument('--backbone_teacher', type=str, default='deit_small_patch16_224')

