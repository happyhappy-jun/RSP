from functools import partial
import torch
import torch.nn as nn
from modeling.layers import CrossAttentionBlock
from modeling.models_rsp_caption import RspCaption
import torch.nn.functional as F

class RspCaptionJointImplicitScale(RspCaption):
    """RSP model variant that uses MSE loss instead of KL divergence, with scaling"""

    def __init__(self,
                 *args,
                 context_emb_dim=3072,
                 cos_scale=1.0,
                 embed_decoder_num_heads=8,
                 embed_decoder_depth=4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cos_scale = cos_scale
        self.decoder_embed_deter = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.joint_emb_decoder = nn.ModuleList([
            CrossAttentionBlock(self.embed_dim, self.embed_dim, embed_decoder_num_heads)
            for _ in range(embed_decoder_depth)
        ])
        self.joint_emb_norm = nn.LayerNorm(self.embed_dim * 2)
        self.to_language_prior = None
        self.language_type_embed = None
        self.image_type_embed = None
        from transformers import AutoModel
        self.text_model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
        self.text_model.eval()

    def get_feat(self, h, z):
        # Process deterministic path
        h = self.decoder_embed_deter(h)  # [B, L, decoder_embed_dim]
        h = h + self.decoder_pos_embed  # Add positional embedding
        # Process stochastic path
        if self.discrete != 0:
            z = z.reshape(*z.shape[:-2], 1, self.stoch * self.discrete)
        z = self.decoder_embed_stoch(z)
        # Final concatenation
        feat = torch.cat([z, h], dim=1)
        return feat

    def forward_decoder_fut(self, h, z):
        kvx_h = self.get_feat(h, z)
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
        q = torch.cat([src_cls, tgt_cls], dim=1)  # [B, 2, decoder_embed_dim]
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(1)  # [B, 1, C]
        for blk in self.joint_emb_decoder:
            q = blk(q, embedding)
        q = q.reshape(q.size(0), 1, -1)
        q = self.joint_emb_norm(q)
        return q

    def patchify_embedding(self, embedding):
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(1)
        embedding = F.normalize(embedding, p=2, dim=-1)
        embedding_patch = embedding.reshape(embedding.shape[0], -1, self.embed_dim)
        return embedding_patch

    def forward(self, src_imgs, tgt_imgs, captions, epoch):
        src_imgs = src_imgs.reshape(-1, *src_imgs.shape[2:])
        tgt_imgs = tgt_imgs.reshape(-1, *tgt_imgs.shape[2:])
        with torch.no_grad():
            outputs = self.text_model(**captions)
            text_embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        text_embeddings = text_embeddings.unsqueeze(1)
        h_context_embed_dim = self.patchify_embedding(text_embeddings)
        src_h, _, _ = self.forward_encoder(src_imgs, mask_ratio=0)
        tgt_h, _, _ = self.forward_encoder(self.perturb(tgt_imgs), mask_ratio=0)
        post_h = self.forward_joint_emb(src_h[:, 0], tgt_h[:, 0], h_context_embed_dim)
        post_logits = self.to_posterior(post_h)
        post_dist = self.make_dist(post_logits)
        post_z = post_dist.rsample()
        prior_h = src_h[:, 0]
        prior_logits = self.to_prior(prior_h)
        prior_dist = self.make_dist(prior_logits)
        prior_z = prior_dist.rsample()
        tgt_pred = self.forward_decoder_fut(src_h, post_z)
        loss_post = self.forward_loss(tgt_imgs, tgt_pred)
        kl_loss, kl_value = self.compute_kl_loss(post_logits, prior_logits)
        img_h, mask, ids_restore = self.forward_encoder(tgt_imgs, mask_ratio=self.mask_ratio)
        pred_masked = self.forward_decoder_mae(img_h, ids_restore)
        mae_loss = self.forward_loss(tgt_imgs, pred_masked, mask)
        with torch.no_grad():
            tgt_pred_prior = self.forward_decoder_fut(src_h, prior_z)
            loss_prior = self.forward_loss(tgt_imgs, tgt_pred_prior)
        loss = loss_post + self.kl_scale * kl_loss + mae_loss
        loss = loss * self.cos_scale
        detailed_loss = {
            "loss_post": loss_post,
            "loss_prior": loss_prior,
            "loss_kl": kl_loss,
            "kl": kl_value,
            "loss_mae": mae_loss,
        }
        return loss, tgt_pred, detailed_loss

def rsp_joint_implicit_scale_vit_small_patch8_dec512d8b(**kwargs):
    model = RspCaptionJointImplicitScale(
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

def rsp_joint_implicit_scale_vit_small_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointImplicitScale(
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

def rsp_joint_implicit_scale_vit_base_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointImplicitScale(
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

def rsp_joint_implicit_scale_vit_large_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointImplicitScale(
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
rsp_joint_implicit_scale_vit_small_patch8 = rsp_joint_implicit_scale_vit_small_patch8_dec512d8b
rsp_joint_implicit_scale_vit_small_patch16 = rsp_joint_implicit_scale_vit_small_patch16_dec512d8b
rsp_joint_implicit_scale_vit_base_patch16 = rsp_joint_implicit_scale_vit_base_patch16_dec512d8b
rsp_joint_implicit_scale_vit_large_patch16 = rsp_joint_implicit_scale_vit_large_patch16_dec512d8b
