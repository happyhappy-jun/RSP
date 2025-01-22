from functools import partial
import torch
import torch.nn as nn

from modeling.layers import RMSNorm, CrossAttention, CrossAttentionBlock
from modeling.models_rsp_caption import RspCaption
from util.pos_embed import get_2d_sincos_pos_embed


class RspCaptionJointM3AE(RspCaption):
    """RSP model variant that uses MSE loss instead of KL divergence"""

    def __init__(self,
                 *args,
                 context_emb_dim=3072,
                 cos_scale=1.0,
                 embed_decoder_num_heads=8,
                 embed_decoder_depth=4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cos_scale = cos_scale
        self.image_type_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim),
        )
        nn.init.normal_(self.image_type_embed, std=0.02)
        self.language_type_embed = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embed_dim),
        )
        nn.init.normal_(self.language_type_embed, std=0.02)

        self.src_tgt_proj = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.decoder_embed_deter = nn.Linear(self.embed_dim, self.decoder_embed_dim)

        # self.image_cls_proj = nn.Linear(self.embed_dim * 2, self.decoder_embed_dim)
        self.joint_emb_decoder = nn.ModuleList([
            CrossAttentionBlock(self.decoder_embed_dim, self.decoder_embed_dim, embed_decoder_num_heads)
        for _ in range(embed_decoder_depth)
        ])
        self.joint_emb_norm = nn.LayerNorm(self.decoder_embed_dim)

        self.to_posterior = nn.Sequential(
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_embed_dim, self.stoch_size),
        )
        
        # MAE decoder
        self.language_patch_num = self.context_emb_dim // self.decoder_embed_dim
        self.mae_decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.language_patch_num, self.decoder_embed_dim),
            requires_grad=False
        )
        mae_decoder_pos_embed = get_2d_sincos_pos_embed(
            self.mae_decoder_pos_embed.shape[-1],
            int((self.patch_embed.num_patches+self.language_patch_num)**0.5),
            cls_token=True,
        )
                
        self.mae_decoder_pos_embed.data.copy_(
            torch.from_numpy(mae_decoder_pos_embed).float().unsqueeze(0)
        )


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

    def forward_decoder_fut(self, h, context_patch, h_context, z):
        h = torch.cat([h, context_patch], dim=1)
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

    def forward_joint_emb(self, src_cls, tgt_cls, embedding):
        # Add sequence dimension if not present
        if len(src_cls.shape) == 2:
            src_cls = src_cls.unsqueeze(1)  # [B, 1, C]
        if len(tgt_cls.shape) == 2:
            tgt_cls = tgt_cls.unsqueeze(1)  # [B, 1, C]

        # Concatenate features and project
        # h = self.image_cls_proj(torch.cat([src_cls, tgt_cls], dim=-1))  # [B, 1, decoder_embed_dim]
        kv = self.src_tgt_proj(torch.cat([src_cls, tgt_cls], dim=1))

        # Add sequence dimension to embedding if needed
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(1)  # [B, 1, C]

        # Apply cross attention blocks
        for blk in self.joint_emb_decoder:
            embedding = blk(embedding, kv)

        embedding = self.joint_emb_norm(embedding)
        return embedding
    
    def reshape_context(self, embedding):
        num_patches = embedding.shape[-1] // self.decoder_embed_dim
        truncated_embedding = self.resize_embed(embedding, num_patches * self.decoder_embed_dim)
        patchfied_embedding = truncated_embedding.view(-1, num_patches, self.decoder_embed_dim)
        return patchfied_embedding

    def forward_decoder_mae(self, h, future_embedding, ids_restore):
        h = self.decoder_embed_mae(h)
        mask_tokens = self.mask_token.repeat(h.shape[0], ids_restore.shape[1] + 1 - h.shape[1], 1)
        h_ = torch.cat([h[:, 1:, :], mask_tokens], dim=1)  # no cls token
        h_ = torch.gather(h_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, h.shape[2]))  # unshuffle
        h = torch.cat([h[:, :1, :], h_], dim=1)  # append cls token
        
        future_patch = self.reshape_context(future_embedding)
        _m_future_patch, _, _m_ids_restore = self.random_masking(future_patch, self.mask_ratio)
        _future_mask_tokens = self.mask_token.repeat(_m_future_patch.shape[0], _m_ids_restore.shape[1] + 1 - _m_future_patch.shape[1], 1)
        _future_mask_tokens = torch.gather(_future_mask_tokens, dim=1, index=_m_ids_restore.unsqueeze(-1).repeat(1, 1, _m_future_patch.shape[2]))
        m_future_patch = torch.cat([_m_future_patch, _future_mask_tokens], dim=1)
        
        h = torch.cat([h, m_future_patch], dim=1)
        
        kvx_h = h + self.mae_decoder_pos_embed

        # embed tokens
        mask_tokens = self.mask_token.repeat(h.shape[0], h.shape[1], 1)
        x = mask_tokens + self.mae_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, kvx=kvx_h)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, src_imgs, tgt_imgs, embedding, future_embedding, epoch):
        # Extract embeddings
        src_imgs = src_imgs.reshape(-1, *src_imgs.shape[2:])
        tgt_imgs = tgt_imgs.reshape(-1, *tgt_imgs.shape[2:])
        embedding = embedding.reshape(-1, embedding.size(-1))
        embedding = embedding.unsqueeze(1)
        h_context = self.resize_embed(embedding, self.decoder_embed_dim)
        src_h, _, _ = self.forward_encoder(src_imgs, mask_ratio=0)
        tgt_h, _, _ = self.forward_encoder(self.perturb(tgt_imgs), mask_ratio=0)

        # Posterior distribution from both images
        # post_h = torch.cat([src_h[:, 0], tgt_h[:, 0]], -1)
        post_h = self.forward_joint_emb(src_h[:, 0], tgt_h[:, 0], h_context)
        post_logits = self.to_posterior(post_h)
        post_dist = self.make_dist(post_logits)
        post_z = post_dist.rsample()

        # Prior distribution only from current images
        prior_h = src_h[:, 0]
        prior_logits = self.to_prior(prior_h)
        prior_dist = self.make_dist(prior_logits)
        prior_z = prior_dist.rsample()

        # Project context to prior space
        h_context_prime = self.to_language_prior(src_h[:, 0])

        tgt_pred = self.forward_decoder_fut(src_h, h_context, post_z)
        loss_post = self.forward_loss(tgt_imgs, tgt_pred)
        kl_loss, kl_value = self.compute_kl_loss(post_logits, prior_logits)
        h_context_prime = h_context_prime.view(h_context.shape)  # Ensure same shape as h_context
        context_loss = 1 - torch.nn.functional.cosine_similarity(h_context.squeeze(1), h_context_prime.squeeze(1), dim=1)
        context_loss = context_loss.mean()

        # MAE
        img_h, mask, ids_restore = self.forward_encoder(tgt_imgs, mask_ratio=self.mask_ratio)
        pred_masked = self.forward_decoder_mae(img_h, future_embedding, ids_restore)
        mae_loss = self.forward_loss(tgt_imgs, pred_masked, mask)

        with torch.no_grad():
            tgt_pred_prior = self.forward_decoder_fut(src_h, h_context, prior_z)
            loss_prior = self.forward_loss(tgt_imgs, tgt_pred_prior)

        loss = loss_post + self.kl_scale * kl_loss + self.cos_scale * context_loss + mae_loss

        detailed_loss = {
            "loss_post": loss_post,
            "loss_prior": loss_prior,
            "loss_kl": kl_loss,
            "kl": kl_value,
            "context_loss": context_loss,
            "loss_mae": mae_loss,
        }

        return loss, tgt_pred, detailed_loss


def rsp_joint_m3ae_vit_small_patch8_dec512d8b(**kwargs):
    model = RspCaptionJointM3AE(
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


def rsp_joint_m3ae_vit_small_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointM3AE(
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


def rsp_joint_m3ae_vit_base_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointM3AE(
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


def rsp_joint_m3ae_vit_large_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointM3AE(
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
rsp_joint_m3ae_vit_small_patch8 = rsp_joint_m3ae_vit_small_patch8_dec512d8b
rsp_joint_m3ae_vit_small_patch16 = rsp_joint_m3ae_vit_small_patch16_dec512d8b
rsp_joint_m3ae_vit_base_patch16 = rsp_joint_m3ae_vit_base_patch16_dec512d8b
rsp_joint_m3ae_vit_large_patch16 = rsp_joint_m3ae_vit_large_patch16_dec512d8b
