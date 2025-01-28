from functools import partial
import numpy as np
import torch
import torch.nn as nn

from modeling.layers import RMSNorm, CrossAttention, CrossAttentionBlock
from modeling.models_rsp_caption import RspCaption
from util.pos_embed import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed


class RspCaptionJointM3AE(RspCaption):
    """
    RSP model variant that uses MSE loss instead of KL divergence with joint M3AE architecture.

    This model combines RSP caption generation with M3AE pretraining for improved
    multimodal understanding.
    """

    def __init__(
        self,
        *args,
        context_emb_dim=3072,
        cos_scale=1.0,
        embed_decoder_num_heads=8,
        embed_decoder_depth=4,
        image_loss_weight=1.0,
        text_loss_weight=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Model hyperparameters
        self.cos_scale = cos_scale
        self.image_loss_weight = image_loss_weight
        self.text_loss_weight = text_loss_weight
        self.language_patch_num = context_emb_dim // self.embed_dim
        del self.language_type_embed
        del self.image_type_embed

        # Type embeddings
        self.encoder_image_type_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )
        self.rsp_decoder_language_type_embed = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embed_dim)
        )
        self.shard_decoder_image_type_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim)
        )
        self.m3ae_enc_language_type_embed = nn.Parameter(
            torch.zeros(1, self.language_patch_num, self.embed_dim)
        )
        self.m3ae_decoder_language_type_embed = nn.Parameter(
            torch.zeros(1, self.language_patch_num, self.decoder_embed_dim)
        )

        # Initialize type embeddings
        self._init_type_embeddings()

        # Linear projections
        self.decoder_embed_deter = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.m3ae_decoder_lang_proj = nn.Linear(self.decoder_embed_dim, self.embed_dim)

        # Joint embedding decoder components
        self.joint_emb_decoder = nn.ModuleList(
            [
                CrossAttentionBlock(
                    self.embed_dim, self.embed_dim, embed_decoder_num_heads
                )
                for _ in range(embed_decoder_depth)
            ]
        )
        self.joint_emb_norm = nn.LayerNorm(self.embed_dim * 2)
        self.to_language_prior = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.decoder_embed_dim),
        )

        # MAE decoder components
        total_patches = 1 + self.num_patches + self.language_patch_num
        self.mae_decoder_pos_embed = nn.Parameter(
            torch.zeros(1, total_patches, self.decoder_embed_dim), requires_grad=False
        )
        self._init_mae_pos_embeddings()

    def _init_type_embeddings(self):
        """Initialize all type embeddings with normal distribution."""
        for param in [
            self.encoder_image_type_embed,
            self.rsp_decoder_language_type_embed,
            self.shard_decoder_image_type_embed,
            self.m3ae_enc_language_type_embed,
            self.m3ae_decoder_language_type_embed,
        ]:
            nn.init.normal_(param, std=0.02)

    def _init_mae_pos_embeddings(self):
        """Initialize MAE positional embeddings."""
        mae_img_pos_embed = get_2d_sincos_pos_embed(
            self.mae_decoder_pos_embed.shape[-1],
            int(np.sqrt(self.patch_embed.num_patches)),
            cls_token=True,
        )
        mae_lang_pos_embed = get_1d_sincos_pos_embed(
            self.mae_decoder_pos_embed.shape[-1], self.language_patch_num
        )
        mae_decoder_pos_embed = np.concatenate(
            [mae_img_pos_embed, mae_lang_pos_embed], axis=0
        )
        self.mae_decoder_pos_embed.data.copy_(
            torch.from_numpy(mae_decoder_pos_embed).float().unsqueeze(0)
        )

    def get_feat(self, h, h_context, z):
        # Process deterministic path
        h = self.decoder_embed_deter(h)  # [B, L, decoder_embed_dim]
        h = h + self.decoder_pos_embed  # Add positional embedding
        h = h + self.shard_decoder_image_type_embed
        h_context = h_context + self.rsp_decoder_language_type_embed

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

    def forward_joint_emb(self, src_cls, tgt_cls, embedding):
        # Add sequence dimension if not present
        if len(src_cls.shape) == 2:
            src_cls = src_cls.unsqueeze(1)  # [B, 1, C]
        if len(tgt_cls.shape) == 2:
            tgt_cls = tgt_cls.unsqueeze(1)  # [B, 1, C]

        # Concatenate features and project
        # h = self.image_cls_proj(torch.cat([src_cls, tgt_cls], dim=-1))  # [B, 1, decoder_embed_dim]
        q = torch.cat([src_cls, tgt_cls], dim=1)

        # Add sequence dimension to embedding if needed
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(1)  # [B, 1, C]

        for blk in self.joint_emb_decoder:
            q = blk(q, embedding)

        q = q.reshape(q.size(0), 1, -1)

        q = self.joint_emb_norm(q)
        return q

    def reshape_context(self, embedding):
        truncated_embedding = self.resize_embed(
            embedding, self.language_patch_num * self.embed_dim
        )
        patchfied_embedding = truncated_embedding.view(
            -1, self.language_patch_num, self.embed_dim
        )
        return patchfied_embedding

    def forward_decoder_mae(self, h, ids_restore):
        ids_restore_img, ids_restore_emb = ids_restore

        img_len = 1 + int(self.num_patches * (1 - self.mask_ratio))
        h = self.decoder_embed_mae(h)
        img_h = h[:, :img_len, :]
        emb_h = h[:, img_len:, :]

        # Handle image part
        mask_tokens = self.mask_token.repeat(
            img_h.shape[0], ids_restore_img.shape[1] + 1 - img_h.shape[1], 1
        )
        img_h_ = torch.cat([img_h[:, 1:, :], mask_tokens], dim=1)  # no cls token
        img_h_ = torch.gather(
            img_h_,
            dim=1,
            index=ids_restore_img.unsqueeze(-1).repeat(1, 1, img_h.shape[2]),
        )  # unshuffle
        img_h = torch.cat([img_h[:, :1, :], img_h_], dim=1)  # append cls token

        emb_mask_tokens = self.mask_token.repeat(
            emb_h.shape[0], ids_restore_emb.shape[1] + 1 - emb_h.shape[1], 1
        )
        emb_h_ = torch.cat([emb_h, emb_mask_tokens], dim=1)  # no cls token
        emb_h = torch.gather(
            emb_h_,
            dim=1,
            index=ids_restore_emb.unsqueeze(-1).repeat(1, 1, emb_h.shape[2]),
        )  # unshuffle

        # Add positional embeddings
        img_h = img_h + self.shard_decoder_image_type_embed
        emb_h = emb_h + self.m3ae_decoder_language_type_embed

        # Concatenate all parts
        h = torch.cat([img_h, emb_h], dim=1)
        kvx_h = h + self.mae_decoder_pos_embed

        # Create input tokens
        mask_tokens = self.mask_token.repeat(h.shape[0], h.shape[1], 1)
        x = mask_tokens + self.mae_decoder_pos_embed

        # Apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, kvx=kvx_h)
        x = self.decoder_norm(x)

        # Predictor projection
        x_img = self.decoder_pred(x[:, : 1 + self.num_patches, :])
        x_emb = self.m3ae_decoder_lang_proj(x[:, 1 + self.num_patches :, :])

        # Remove cls token
        x_img = x_img[:, 1:, :]

        return x_img, x_emb

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
        x = x + self.encoder_image_type_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_m3ae_encoder(self, imgs, future_embedding, mask_ratio=0.0):
        x_img = self.patch_embed(imgs)
        x_img = x_img + self.pos_embed[:, 1:, :]
        x_img = x_img + self.encoder_image_type_embed[:, 1:, :]

        x_emb = self.reshape_context(future_embedding)
        x_emb = x_emb + self.m3ae_enc_language_type_embed

        if mask_ratio != 0.0:
            x_img, mask_img, ids_restore_img = self.random_masking(x_img, mask_ratio)
            x_emb, mask_emb, ids_restore_emb = self.random_masking(x_emb, mask_ratio)
        else:
            mask_img, ids_restore_img = None, None
            mask_emb, ids_restore_emb = None, None

        cls_token = (
            self.cls_token
            + self.pos_embed[:, :1, :]
            + self.encoder_image_type_embed[:, :1, :]
        )
        cls_tokens = cls_token.expand(x_img.shape[0], -1, -1)
        x_img = torch.cat((cls_tokens, x_img), dim=1)

        x = torch.cat([x_img, x_emb], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, (mask_img, mask_emb), (ids_restore_img, ids_restore_emb)

    def forward_m3ae_loss(self, imgs, future_embed, pred_img, pred_emb, mask=None):
        """
        Calculate M3AE reconstruction loss for both images and embeddings.
        Args:
            imgs: Input images
            future_embed: Future embedding target
            pred_img: Predicted image reconstructions
            pred_emb: Predicted embedding reconstructions
            mask: Optional mask for masked modeling
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        image_loss = self._compute_image_reconstruction_loss(
            imgs, pred_img, mask[0] if mask else None
        )
        embed_loss = self._compute_embedding_reconstruction_loss(
            future_embed, pred_emb, mask[1] if mask else None
        )

        # Combine losses with weights
        total_loss = (
            self.image_loss_weight * image_loss + self.text_loss_weight * embed_loss
        )

        return total_loss, {
            "loss_img_recon": image_loss,
            "loss_emb_recon": embed_loss,
        }

    def _compute_image_reconstruction_loss(self, imgs, pred_img, mask=None):
        """Compute reconstruction loss for images."""
        target = self.patchify(imgs)

        # Normalize pixels if required
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-4) ** 0.5

        # Calculate squared error
        loss = (pred_img - target) ** 2

        # Apply mask if provided
        if mask is not None:
            loss = loss.mean(dim=-1)  # Mean over feature dimension
            loss = (loss * mask).sum() / mask.sum()  # Mean over masked tokens
        else:
            loss = loss.mean()  # Mean over all dimensions

        return loss

    def _compute_embedding_reconstruction_loss(self, future_embed, pred_emb, mask=None):
        """Compute reconstruction loss for embeddings."""
        emb_target = self.reshape_context(future_embed)
        loss = (pred_emb - emb_target) ** 2

        # Apply mask if provided
        if mask is not None:
            loss = loss.mean(dim=-1)  # Mean over feature dimension
            loss = (loss * mask).sum() / mask.sum()  # Mean over masked tokens
        else:
            loss = loss.mean()  # Mean over all dimensions

        return loss

    def forward(self, src_imgs, tgt_imgs, embedding, future_embedding, epoch):
        # Extract embeddings
        src_imgs = src_imgs.reshape(-1, *src_imgs.shape[2:])
        tgt_imgs = tgt_imgs.reshape(-1, *tgt_imgs.shape[2:])
        embedding = embedding.reshape(-1, embedding.size(-1))
        embedding = embedding.unsqueeze(1)
        h_context_embed_dim = self.resize_embed(embedding, self.embed_dim)
        src_h, _, _ = self.forward_encoder(src_imgs, mask_ratio=0)
        tgt_h, _, _ = self.forward_encoder(self.perturb(tgt_imgs), mask_ratio=0)

        # Posterior distribution from both images
        # post_h = torch.cat([src_h[:, 0], tgt_h[:, 0]], -1)
        post_h = self.forward_joint_emb(src_h[:, 0], tgt_h[:, 0], h_context_embed_dim)
        post_logits = self.to_posterior(post_h)
        post_dist = self.make_dist(post_logits)
        post_z = post_dist.rsample()

        # Prior distribution only from current images
        prior_h = src_h[:, 0]
        prior_logits = self.to_prior(prior_h)
        prior_dist = self.make_dist(prior_logits)
        prior_z = prior_dist.rsample()

        # Project context to prior space
        h_context_prime_decoder_embed_dim = self.to_language_prior(src_h[:, 0])

        h_context_decoder = self.resize_embed(embedding, self.decoder_embed_dim)
        tgt_pred = self.forward_decoder_fut(src_h, h_context_decoder, post_z)
        loss_post = self.forward_loss(tgt_imgs, tgt_pred)
        kl_loss, kl_value = self.compute_kl_loss(post_logits, prior_logits)
        h_context_prime_decoder_embed_dim = h_context_prime_decoder_embed_dim.view(
            h_context_decoder.shape
        )
        context_loss = 1 - torch.nn.functional.cosine_similarity(
            h_context_prime_decoder_embed_dim.squeeze(1),
            h_context_decoder.squeeze(1),
            dim=1,
        )
        context_loss = context_loss.mean()

        # MAE
        x, mask, ids_restore = self.forward_m3ae_encoder(
            tgt_imgs, future_embedding, mask_ratio=self.mask_ratio
        )
        img_pred_masked, emb_pred_masked = self.forward_decoder_mae(x, ids_restore)
        m3ae_loss, m3ae_losses = self.forward_m3ae_loss(
            tgt_imgs, future_embedding, img_pred_masked, emb_pred_masked, mask
        )

        with torch.no_grad():
            tgt_pred_prior = self.forward_decoder_fut(src_h, h_context_decoder, prior_z)
            loss_prior = self.forward_loss(tgt_imgs, tgt_pred_prior)

        loss = (
            loss_post
            + self.kl_scale * kl_loss
            + self.cos_scale * context_loss
            + m3ae_loss
        )

        detailed_loss = {
            "loss_post": loss_post,
            "loss_prior": loss_prior,
            "loss_kl": kl_loss,
            "kl": kl_value,
            "context_loss": context_loss,
            "loss_mae": m3ae_loss,
            "loss_img_recon": m3ae_losses["loss_img_recon"],
            "loss_emb_recon": m3ae_losses["loss_emb_recon"],
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
