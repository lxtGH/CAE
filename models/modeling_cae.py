import math
import time
import torch
import torch.nn as nn
from functools import partial

from models.modeling_finetune import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from models.modeling_cae_helper import *

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__()

        self.encoder = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)

        # alignment constraint
        self.teacher = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)

        self.init_std = init_std
        self.args = args
        self.num_patches = self.encoder.patch_embed.num_patches

        self.pretext_neck = VisionTransformerNeck(patch_size=patch_size, num_classes=args.decoder_num_classes, embed_dim=args.decoder_embed_dim, depth=args.regressor_depth,
            num_heads=args.decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=args.decoder_layer_scale_init_value, num_patches=self.num_patches, init_std=init_std, args=args)

        # encoder to decoder projection, borrowed from mae.
        if args.decoder_embed_dim != embed_dim:
            self.encoder_to_decoder = nn.Linear(embed_dim, args.decoder_embed_dim, bias=True)
            self.encoder_to_decoder_norm = norm_layer(args.decoder_embed_dim)
        else:
            self.encoder_to_decoder = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        ### whether to use 'rescale' to init the weight, borrowed from beit.
        if not args.fix_init_weight:
            self.apply(self._init_weights)
        self._init_teacher()
        
        
    def _init_teacher(self):  
        # init the weights of teacher with those of backbone
        for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0):
        """Momentum update of the teacher network."""
        for param_encoder, param_teacher in zip(self.encoder.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                param_encoder.data * (1. - base_momentum)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    '''
    Input shape:
        x: [bs, 3, 224, 224]
        bool_masked_pos: [bs, num_patch * num_patch]
    '''
    def forward(self, x, bool_masked_pos, return_all_tokens=None):
        batch_size = x.size(0)

        '''
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        '''
        x_unmasked = self.encoder(x, bool_masked_pos=bool_masked_pos)

        # encoder to decoder projection
        if self.encoder_to_decoder is not None:
            x_unmasked = self.encoder_to_decoder(x_unmasked)
            x_unmasked = self.encoder_to_decoder_norm(x_unmasked)

        '''
        Alignment constraint
        '''
        with torch.no_grad():
            latent_target = self.teacher(x, bool_masked_pos=(~bool_masked_pos))
            latent_target = latent_target[:, 1:, :] # remove class token
            if self.encoder_to_decoder is not None:
                latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))

            self.momentum_update(self.args.base_momentum)

        '''
        Latent contextual regressor and decoder
        '''
        b, num_visible_plus1, dim = x_unmasked.shape
        # remove class token
        x_unmasked = x_unmasked[:, 1:, :]

        num_masked_patches = self.num_patches - (num_visible_plus1-1)
        
        # generate position embeddings.
        pos_embed = self.encoder.build_2d_sincos_position_embedding(dim, use_cls_token=True).expand(batch_size, self.num_patches+1, dim).cuda(x_unmasked.device)

        # pos embed for masked patches
        pos_embed_masked = pos_embed[:,1:][bool_masked_pos].reshape(batch_size, -1, dim) 

        # pos embed for unmasked patches
        pos_embed_unmasked = pos_embed[:,1:][~bool_masked_pos].reshape(batch_size, -1, dim) 

        # masked embedding '''
        x_masked = self.mask_token.expand(batch_size, num_masked_patches, -1)

        logits, latent_pred = self.pretext_neck(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos)
        logits = logits.view(-1, logits.shape[2])

        return logits, latent_pred, latent_target


@register_model
def cae_small_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cae_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cae_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
