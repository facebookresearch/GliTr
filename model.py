# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, Attention
from timm.models.registry import register_model
from timm.models import create_model
import types
import copy

# teacher student pair
__all__ = [
    'deit_tiny_patch16_224',
    'deit_small_patch16_224',
    'deit_base_patch16_224',
    ]

@register_model
def deit_tiny_patch16_224(pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def deit_small_patch16_224(pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def deit_base_patch16_224(pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError
    return model


def causalforward(self, x):
    #print('in causal attn')
    B, N, C = x.shape

    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.masked_fill(self.mask == 0, float('-inf'))  # Causal mask
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x

def MakeCausalAttention(m):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == Attention:
            print('CausalAttention on ', attr_str)
            target_attr.forward = types.MethodType(causalforward, target_attr)
    for n, ch in m.named_children():
        MakeCausalAttention(ch)

class TemporalPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        '''
        We repurpose some of the arguments.
        Name of these arguments are kept consistent with timm, but there meaning is different.
        Refer to the following:
        img_size = num_frames_per_video
        patch_size = No meaning yet
        in_chans = backbone embed_dim
        embed_dim = temporal head embed_dim
        '''
        self.num_patches = img_size
        if in_chans == embed_dim:
            self.embed_layer = nn.Identity()
        else:
            self.embed_layer = nn.Sequential(
                nn.Linear(in_chans, embed_dim),
                nn.LayerNorm(embed_dim, eps=1e-6),
                )

    def forward(self, x):
        return self.embed_layer(x)


def temporal_forward(self, x):
    B, T = x.size(0), x.size(1)
    x = self.patch_embed(x) + self.tmp_embed[:,:T,:]

    mask = torch.tril(torch.ones(T,T)).reshape(1, 1, T, T).to(x.device)
    for b in self.blocks:
        b.attn.mask = mask

    x = self.blocks(x)
    x = self.norm(x)
    x = self.pre_logits(x)
    out = self.head(x)
    return out

def backbone_forward(self, x, pos_embed):
    B, T = x.size(0), x.size(1)

    # transformer op
    x = self.patch_embed(x.flatten(0,1))
    x = x + pos_embed.flatten(0,1)

    cls_token = self.cls_token.expand(B*T, -1, -1) + self.cls_embed
    x = torch.cat((cls_token, x), dim=1)

    x = self.blocks(x) 
    x = self.norm(x)
    x = self.pre_logits(x)

    return x[:,0].reshape(B, T, -1)

class VideoTransformer(nn.Module):
    def __init__(self, backbone, num_classes, num_frames_per_video, drop, drop_path, num_patches_in_glimpse, criterion, attntype, pretrained_dir, teacher=None):
        super().__init__()

        self.num_patches_in_glimpse = num_patches_in_glimpse
        n_glimpse = 14 - num_patches_in_glimpse + 1
        self.n_glimpse = n_glimpse
        self.num_class = num_classes
        self.num_segments   = num_frames_per_video
        self.attntype = attntype
        self.teacher = teacher

        # Define spatial backbone
    
        self.backbone_name = backbone
        self.backbone = create_model(backbone, pretrained=False, drop_rate=drop, drop_path_rate=drop_path, drop_block_rate=None)
        self.backbone.patch_embed.img_size = [num_patches_in_glimpse*16, num_patches_in_glimpse*16]
        self.backbone.num_patches_in_glimpse = num_patches_in_glimpse
        self.backbone.forward = types.MethodType(backbone_forward, self.backbone)

        del self.backbone.head
        checkpoint = torch.load(os.path.join(pretrained_dir, 'ibot_vits_16_checkpoint_teacher.pth'), map_location=torch.device('cpu'))['state_dict']
        self.backbone.load_state_dict(checkpoint)

        pos_embed = self.backbone.pos_embed.data.clone()
        self.backbone.pos_embed = nn.Parameter(pos_embed[:,1:,:])
        self.backbone.cls_embed = nn.Parameter(pos_embed[:,:1,:])

        # Define temporal head
        in_chans=self.backbone.embed_dim

        self.temporal_head = VisionTransformer(
            img_size=num_frames_per_video, in_chans=in_chans, embed_layer=TemporalPatchEmbed,
            num_classes=num_classes, embed_dim=2*self.backbone.embed_dim, depth=4, num_heads=6,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=drop, drop_path_rate=drop_path,
            distilled=False)
     
        MakeCausalAttention(self.temporal_head)
        self.temporal_head.tmp_embed = nn.Parameter(self.temporal_head.pos_embed.data.clone())
        del self.temporal_head.pos_embed # renaming pos_embed

        self.temporal_head.tmp_embed = nn.Parameter(self.temporal_head.tmp_embed.data[:,1:].clone()) # removing cls embed
        self.temporal_head.forward = types.MethodType(temporal_forward, self.temporal_head)

        del self.temporal_head.cls_token

        self.temploc_head = copy.deepcopy(self.temporal_head)
        self.temploc_head.init_loc = nn.Parameter(torch.Tensor([0, 0]).reshape(1,1,2))
        self.temploc_head.head = nn.Linear(self.temploc_head.embed_dim, 2, bias=False)
        self.temploc_head.head.weight.data *= 0.1

        if 'student' in self.attntype:
            self.backbone.load_state_dict(self.teacher.backbone.state_dict())
            self.temporal_head.load_state_dict(self.teacher.temporal_head.state_dict())
            self.temploc_head.load_state_dict(self.teacher.temploc_head.state_dict())
            self.forward = self.forward_student

        elif 'teacher' in self.attntype:
            self.backbone.patch_embed.img_size = [224,224]
            self.forward = self.forward_teacher

        self.criterion = criterion

    @torch.jit.ignore
    def no_weight_decay(self):
        return ['backbone.cls_token',
                'backbone.pos_embed',
                'backbone.cls_embed',
                'temporal_head.cls_token',
                'temporal_head.tmp_embed',
                'temploc_head.cls_token',
                'temploc_head.tmp_embed',]

    def forward_teacher(self, x, label_1hot):
        B, T = x.size(0), x.size(1)

        # forward with random
        self.backbone.patch_embed.img_size = [224,224]
        feat = self.backbone(x, self.backbone.pos_embed[None,...].repeat(B,T,1,1))
        cls = self.temporal_head(feat)
        loc = self.temploc_head(feat.detach())
        L_cls = self.criterion(cls.reshape(B*T,-1), label_1hot[:,None,:].float().repeat(1, T, 1).reshape(B*T, -1)).mean()
        loc = torch.cat([self.temploc_head.init_loc.repeat(B,1,1), loc[:,:-1]], 1)

        if self.training:
            if self.teacher is not None:
                # distillation
                self.teacher.eval()
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                L_tch = self.kld_loss(cls[:,-1,:], teacher_logits)
            else:
                L_tch = torch.zeros_like(L_cls)

            # SSL for loc
            copy_backbone = copy.deepcopy(self.backbone)
            for p in copy_backbone.parameters(): p.requires_grad = False
            copy_backbone.patch_embed.img_size = [self.num_patches_in_glimpse*16, self.num_patches_in_glimpse*16]
            patches, pos, mask = self.prepare_patch(loc, x, copy_backbone.pos_embed)
            feat_ = copy_backbone(patches, pos)
            L_mse = F.mse_loss(feat_, feat.detach())

            copy_temporal_head = copy.deepcopy(self.temporal_head)
            for p in copy_temporal_head.parameters(): p.requires_grad = False
            cls_ = copy_temporal_head(feat_)
            L_kld = self.kld_loss(cls_, cls.detach())

        else:
            _, _, mask = self.prepare_patch(loc, x, self.backbone.pos_embed)
            L_tch = torch.zeros_like(L_cls)
            L_mse = torch.zeros_like(L_cls)
            L_kld = torch.zeros_like(L_cls)

        return cls.permute(1,0,2), L_cls, L_tch, L_mse, L_kld, mask

    def kld_loss(self, student_logits, teacher_logits):
        loss = (F.softmax(student_logits, dim=-1)*(F.log_softmax(student_logits, dim=-1) - torch.log_softmax(teacher_logits.detach(), dim=-1))).sum(-1).mean()
        return loss

    def forward_student(self, x, label_1hot):
        B, T = x.size(0), x.size(1)

        if (self.training):
            with torch.no_grad():
                self.teacher.eval()
                teacher_feat = self.teacher.backbone(x, self.teacher.backbone.pos_embed[None,...].repeat(B,T,1,1))
                teacher_cls = self.teacher.temporal_head(teacher_feat)

        # label
        label_1hot  = label_1hot.byte().argmax(-1).long()
        label_1hot = label_1hot.reshape(B)

        L_cls, L_tch, L_mse, L_kld = torch.zeros(1).mean().to(x.device), torch.zeros(1).mean().to(x.device), torch.zeros(1).mean().to(x.device), torch.zeros(1).mean().to(x.device) 

        all_masks = []
        all_feat = []
        all_cls = []

        loc = self.temploc_head.init_loc.repeat(B,1,1)
        for t in range(T):
            patches, pos, mask = self.prepare_patch(loc, x[:,t:t+1], self.backbone.pos_embed)
            feat = self.backbone(patches, pos)
            all_feat.append(feat)
            loc = self.temploc_head(torch.cat(all_feat, 1).detach())
            cls = self.temporal_head(torch.cat(all_feat, 1))
            loc = loc[:,-1:,:]
            all_cls.append(cls[:,-1:,:])
            all_masks.append(mask)
            
        all_masks = torch.cat(all_masks,1)
        all_feat = torch.cat(all_feat, 1)
        all_cls = torch.cat(all_cls, 1)

        L_cls = self.criterion(all_cls.reshape(B*T,-1), label_1hot[:,None].repeat(1,T).reshape(B*T)).mean()
        if (self.training):
            L_mse = F.mse_loss(all_feat, teacher_feat.detach())
            L_kld = self.kld_loss(all_cls, teacher_cls.detach())
        else:
            L_mse = torch.zeros_like(L_cls)
            L_kld = torch.zeros_like(L_cls)

       
        return all_cls.permute(1,0,2), L_cls, L_tch, L_mse, L_kld, all_masks

    def prepare_patch(self, loc, x, pos_embed):
        # remaining glimpses are sampled
        B, T, C, H, W = x.size()
        D = pos_embed.size(-1)

        scale = self.num_patches_in_glimpse/14
        loc = loc.reshape(B*T, -1)
        shift = loc

        theta = torch.zeros(B*T, 2, 3).to(loc.device)
        theta[:,0,0] = scale
        theta[:,1,1] = scale
        theta[:,:,2] = shift

        grid_x = F.affine_grid(theta, (B*T, C, self.num_patches_in_glimpse*16, self.num_patches_in_glimpse*16), align_corners=True)
        patch = F.grid_sample(x.flatten(0,1), grid_x, align_corners=True)
        patch = patch.reshape(B, T, C, self.num_patches_in_glimpse*16, self.num_patches_in_glimpse*16)

        grid_p = F.affine_grid(theta, (B*T, D, self.num_patches_in_glimpse, self.num_patches_in_glimpse), align_corners=True)
        pos_embed = pos_embed.permute(0,2,1).reshape(1,D,14,14).repeat(B*T,1,1,1)
        pos_embed = F.grid_sample(pos_embed, grid_p, align_corners=True)
        pos_embed = pos_embed.reshape(B, T, D, self.num_patches_in_glimpse**2).permute(0, 1, 3, 2)

        with torch.no_grad():
            mask = torch.ones(B*T,1,self.num_patches_in_glimpse*16, self.num_patches_in_glimpse*16).to(x.device)
            theta = torch.zeros(B*T, 2, 3).to(loc.device)
            theta[:,0,0] = 1
            theta[:,1,1] = 1
            theta[:,:,2] = -shift
            theta = theta / scale
            theta = theta.detach()
            grid_m = F.affine_grid(theta, (B*T, 1, 224, 224), align_corners=True)
            mask = F.grid_sample(mask, grid_m, align_corners=True)
            mask = mask.reshape(B, T, 224, 224)

        return patch, pos_embed, mask

