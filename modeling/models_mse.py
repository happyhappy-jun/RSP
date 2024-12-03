from functools import partial
import torch
import torch.nn as nn
from modeling.models_rsp import RSP

class RSP_MSE(RSP):
    """RSP model variant that uses MSE loss instead of KL divergence"""
    
    def compute_kl_loss(self, post_logits, prior_logits):
        """Override KL loss with MSE loss between posterior and prior logits"""
        mse_loss = nn.MSELoss()(post_logits, prior_logits)
        return mse_loss, mse_loss.detach()  # Return same value for loss and metric

def rsp_mse_vit_small_patch8_dec512d8b(**kwargs):
    model = RSP_MSE(
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def rsp_mse_vit_small_patch16_dec512d8b(**kwargs):
    model = RSP_MSE(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def rsp_mse_vit_base_patch16_dec512d8b(**kwargs):
    model = RSP_MSE(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def rsp_mse_vit_large_patch16_dec512d8b(**kwargs):
    model = RSP_MSE(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

# Aliases
rsp_mse_vit_small_patch8 = rsp_mse_vit_small_patch8_dec512d8b
rsp_mse_vit_small_patch16 = rsp_mse_vit_small_patch16_dec512d8b
rsp_mse_vit_base_patch16 = rsp_mse_vit_base_patch16_dec512d8b
rsp_mse_vit_large_patch16 = rsp_mse_vit_large_patch16_dec512d8b
