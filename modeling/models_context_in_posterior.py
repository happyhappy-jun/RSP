from functools import partial
import torch
import torch.nn as nn

from modeling.layers import RMSNorm
from modeling.models_rsp import RSP
from modeling.models_rsp_caption import RspCaption


class RspContextInPosterior(RspCaption):
    """RSP model variant that uses MSE loss instead of KL divergence"""
    def __init__(self, 
                 enable_rms_norm=False, 
                 embed_scale_factor=1.0, 
                 context_embed_dim=3072,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.to_posterior = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 3),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 3, self.stoch_size),
        )
        self.enable_rms_norm = enable_rms_norm
        if self.enable_rms_norm:
            self.rms_norm = RMSNorm(self.embed_dim, scale_factor=embed_scale_factor, eps=1e-6)
        else:
            self.rms_norm = nn.Identity()
            
        nn.init.normal_(self.context_proj.weight, std=0.02)
        self.to_language_prior = None
        self.language_type_embed = None
        self.image_type_embed = None


    def forward(self, src_imgs, tgt_imgs, embedding, epoch):
        # Extract embeddings
        src_imgs = src_imgs.reshape(-1, *src_imgs.shape[2:])
        tgt_imgs = tgt_imgs.reshape(-1, *tgt_imgs.shape[2:])
        embedding = embedding.reshape(-1, embedding.size(-1))

        src_h, _, _ = self.forward_encoder(src_imgs, mask_ratio=0)
        tgt_h, _, _ = self.forward_encoder(self.perturb(tgt_imgs), mask_ratio=0)

        # Posterior distribution from both images
        # h_context = self.resize_embed(embedding, self.embed_dim)
        h_context = self.resize_embed(embedding, self.embed_dim)
        h_context = self.rms_norm(h_context)
        
        post_h = torch.cat([src_h[:, 0], tgt_h[:, 0], h_context], -1)
        post_logits = self.to_posterior(post_h)
        post_dist = self.make_dist(post_logits)
        post_z = post_dist.rsample()

        # Prior distribution only from current images
        prior_h = src_h[:, 0]
        prior_logits = self.to_prior(prior_h)
        prior_dist = self.make_dist(prior_logits)
        prior_z = prior_dist.rsample()

        tgt_pred = self.forward_decoder_fut(src_h, post_z)
        loss_post = self.forward_loss(tgt_imgs, tgt_pred)
        kl_loss, kl_value = self.compute_kl_loss(post_logits, prior_logits)

        # MAE
        img_h, mask, ids_restore = self.forward_encoder(tgt_imgs, mask_ratio=self.mask_ratio)
        pred_masked = self.forward_decoder_mae(img_h, ids_restore)
        mae_loss = self.forward_loss(tgt_imgs, pred_masked, mask)

        with torch.no_grad():
            tgt_pred_prior = self.forward_decoder_fut(src_h, prior_z)
            loss_prior = self.forward_loss(tgt_imgs, tgt_pred_prior)

        loss = loss_post + self.kl_scale * kl_loss + mae_loss

        detailed_loss = {
            "loss_post": loss_post,
            "loss_prior": loss_prior,
            "kl_loss": kl_loss,
            "kl": kl_value,
            "loss_mae": mae_loss,
        }

        return loss, tgt_pred, detailed_loss

def rsp_context_in_post_small_patch8_dec512d8b(**kwargs):
    model = RspContextInPosterior(
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

def rsp_context_in_post_small_patch16_dec512d8b(**kwargs):
    model = RspContextInPosterior(
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

def rsp_context_in_post_base_patch16_dec512d8b(**kwargs):
    model = RspContextInPosterior(
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

def rsp_context_in_post_large_patch16_dec512d8b(**kwargs):
    model = RspContextInPosterior(
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
rsp_context_in_posterior_vit_small_patch8 = rsp_context_in_post_small_patch8_dec512d8b
rsp_context_in_posterior_vit_small_patch16 = rsp_context_in_post_small_patch16_dec512d8b
rsp_context_in_posterior_vit_base_patch16 = rsp_context_in_post_base_patch16_dec512d8b
rsp_context_in_posterior_vit_large_patch16 = rsp_context_in_post_large_patch16_dec512d8b