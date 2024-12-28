from functools import partial
import torch
import torch.nn as nn

from modeling.layers import RMSNorm
from modeling.models_rsp import RSP
from modeling.models_rsp_caption import RspCaption


class RspCaptionMseReg(RspCaption):
    """
    RSP model variant that uses MSE loss instead of KL divergence
    Also includes a register token for storing extra information

    """

    def __init__(
            self,
            *args,
            mse_scale=1.0,
            enable_rms_norm=False,
            embed_scale_factor=1.0,
            num_register_tokens=4,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mse_scale = mse_scale
        self.image_type_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim),
        )
        nn.init.normal_(self.image_type_embed, std=0.02)
        self.language_type_embed = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embed_dim),
        )
        nn.init.normal_(self.language_type_embed, std=0.02)
        self.rms_norm = RMSNorm(self.decoder_embed_dim, scale_factor=embed_scale_factor, eps=1e-6)
        self.enable_rms_norm = enable_rms_norm

        # register token for store extra information
        self.register_token = nn.Parameter(torch.randn(num_register_tokens, self.embed_dim))

    def forward_encoder(self, imgs, mask_ratio=0.0):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio != 0.0:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def get_feat(self, h, h_context, z):
        # Process deterministic path
        h = self.decoder_embed_deter(h)  # [B, L, decoder_embed_dim]
        h = h + self.decoder_pos_embed  # Add positional embedding
        h = h + self.image_type_embed
        h_context = h_context + self.language_type_embed

        # Concatenate along sequence dimension
        h_concat = torch.cat([h, h_context], dim=1)  # [B, L+1, decoder_embed_dim]

        # Process stochastic path
        if self.discrete != 0:
            z = z.reshape(*z.shape[:-2], 1, self.stoch * self.discrete)
        z = self.decoder_embed_stoch(z)

        # Final concatenation
        feat = torch.cat([z, h_concat], dim=1)
        return feat

    def forward_decoder_fut(self, h, h_context, z):
        kvx_h = self.get_feat(h, h_context, z)

        mask_tokens = self.mask_token.repeat(h.shape[0], h.shape[1], 1)
        x = mask_tokens + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, kvx=kvx_h)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, src_imgs, tgt_imgs, embedding, epoch):
        # Extract embeddings
        src_imgs = src_imgs.reshape(-1, *src_imgs.shape[2:])
        tgt_imgs = tgt_imgs.reshape(-1, *tgt_imgs.shape[2:])
        embedding = embedding.reshape(-1, embedding.size(-1))

        src_h, _, _ = self.forward_encoder(src_imgs, mask_ratio=0)
        tgt_h, _, _ = self.forward_encoder(self.perturb(tgt_imgs), mask_ratio=0)

        # Posterior distribution from both images
        post_h = torch.cat([src_h[:, 0], tgt_h[:, 0]], -1)
        post_logits = self.to_posterior(post_h)
        post_dist = self.make_dist(post_logits)
        post_z = post_dist.rsample()

        # Prior distribution only from current images
        prior_h = src_h[:, 0]
        prior_logits = self.to_prior(prior_h)
        prior_dist = self.make_dist(prior_logits)
        prior_z = prior_dist.rsample()

        embedding = embedding.view(-1, 1, embedding.size(-1))
        h_context = self.resize_embed(embedding, self.decoder_embed_dim)
        if self.enable_rms_norm:
            h_context = self.rms_norm(h_context)

        # Project context to prior space
        h_context_prime = self.to_language_prior(src_h[:, 0])

        tgt_pred = self.forward_decoder_fut(src_h, h_context, post_z)
        loss_post = self.forward_loss(tgt_imgs, tgt_pred)
        kl_loss, kl_value = self.compute_kl_loss(post_logits, prior_logits)
        h_context_prime = h_context_prime.view(h_context.shape)  # Ensure same shape as h_context
        context_loss = torch.nn.functional.mse_loss(h_context, h_context_prime)

        # MAE
        img_h, mask, ids_restore = self.forward_encoder(tgt_imgs, mask_ratio=self.mask_ratio)
        pred_masked = self.forward_decoder_mae(img_h, ids_restore)
        mae_loss = self.forward_loss(tgt_imgs, pred_masked, mask)

        with torch.no_grad():
            tgt_pred_prior = self.forward_decoder_fut(src_h, h_context, prior_z)
            loss_prior = self.forward_loss(tgt_imgs, tgt_pred_prior)

        loss = loss_post + self.kl_scale * kl_loss + self.mse_scale * context_loss + mae_loss

        detailed_loss = {
            "loss_post": loss_post,
            "loss_prior": loss_prior,
            "loss_kl": kl_loss,
            "kl": kl_value,
            "context_loss": context_loss,
            "loss_mae": mae_loss,
        }

        return loss, tgt_pred, detailed_loss


def rsp_mse_vit_small_patch8_dec512d8b(**kwargs):
    model = RspCaptionMse(
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
    model = RspCaptionMse(
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
    model = RspCaptionMse(
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
    model = RspCaptionMse(
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
