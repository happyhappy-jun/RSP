from functools import partial
import torch
import torch.nn as nn

from modeling.layers import RMSNorm, CrossAttention, CrossAttentionBlock
from modeling.models_rsp_caption import RspCaption

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    device = pos.device if hasattr(pos, "device") else torch.device("cpu")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos = torch.arange(length, dtype=torch.float32, device=device)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    emb = emb.unsqueeze(0)
    return emb


class RspCaptionJointSelf(RspCaption):
    """RSP model variant that uses MSE loss instead of KL divergence"""

    def __init__(self,
                 *args,
                 cos_scale=1.0,
                 embed_decoder_num_heads=8,
                 embed_decoder_depth=4,
                 vocab_size=30522,
                 text_embed_dim=None,
                 num_text_layers=2,
                 nhead_text=8,
                 max_text_length=33,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cos_scale = cos_scale
        self.image_type_embed = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embed_dim),
        )
        nn.init.normal_(self.image_type_embed, std=0.02)
        self.language_type_embed = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embed_dim),
        )
        nn.init.normal_(self.language_type_embed, std=0.02)

        self.decoder_embed_deter = nn.Linear(self.embed_dim, self.decoder_embed_dim)

        self.joint_emb_decoder = nn.ModuleList([
            CrossAttentionBlock(self.embed_dim, self.embed_dim, embed_decoder_num_heads)
        for _ in range(embed_decoder_depth)
        ])
        self.joint_emb_norm = nn.LayerNorm(self.embed_dim *2)
        self.to_language_prior = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.decoder_embed_dim),
        )
        
        if text_embed_dim is None:
            text_embed_dim = self.embed_dim
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, text_embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=text_embed_dim, nhead=nhead_text),
            num_layers=num_text_layers
        )
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_text_length, text_embed_dim))
        nn.init.trunc_normal_(self.text_pos_embed, std=0.02)


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

    def forward_joint_emb(self, src_cls, tgt_cls, embedding):
        # Add sequence dimension if not present
        if len(src_cls.shape) == 2:
            src_cls = src_cls.unsqueeze(1)  # [B, 1, C]
        if len(tgt_cls.shape) == 2:
            tgt_cls = tgt_cls.unsqueeze(1)  # [B, 1, C]

        # Concatenate features and project
        # h = self.image_cls_proj(torch.cat([src_cls, tgt_cls], dim=-1))  # [B, 1, decoder_embed_dim]
        q = torch.cat([src_cls, tgt_cls], dim=1) # [B, 2, decoder_embed_dim]

        # Add sequence dimension to embedding if needed
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(1)  # [B, 1, C]

        # Apply cross attention blocks
        for blk in self.joint_emb_decoder:
            q = blk(q, embedding)
            
        q = q.reshape(q.size(0), 1, -1)

        q = self.joint_emb_norm(q)
        return q


    def forward(self, batch, epoch):
        src_imgs = batch["src_images"]
        tgt_imgs = batch["tgt_images"]
        input_ids = batch["input_ids"]
        attention_map = batch["attention_map"]

        # Process text: Convert input_ids to embeddings and add sincos positional encoding
        text_x = (
            self.text_embedding(input_ids)
            + get_1d_sincos_pos_embed(self.embed_dim, input_ids.shape[1]).to(input_ids.device)
            + self.get_type_embedding('encoder_text_type_embedding')
        )
        # Transformer encoder expects [seq_len, batch, d]
        text_encoded = self.text_encoder(text_x.transpose(0, 1),
                                         src_key_padding_mask=(attention_map==0))
        text_encoded = text_encoded.transpose(0, 1)  # [B, L, d]
        # Use [CLS] token representation (assuming first token is [CLS])
        caption_embedding = text_encoded[:, 0, :].unsqueeze(1)  # [B, 1, d]

        # Image encoding
        src_imgs = src_imgs.reshape(-1, *src_imgs.shape[2:])
        tgt_imgs = tgt_imgs.reshape(-1, *tgt_imgs.shape[2:])
        src_h, _, _ = self.forward_encoder(src_imgs, mask_ratio=0)
        tgt_h, _, _ = self.forward_encoder(self.perturb(tgt_imgs), mask_ratio=0)

        h_context_embed_dim = self.resize_embed(caption_embedding, self.embed_dim)
        post_h = self.forward_joint_emb(src_h[:, 0], tgt_h[:, 0], h_context_embed_dim)
        post_logits = self.to_posterior(post_h)
        post_dist = self.make_dist(post_logits)
        post_z = post_dist.rsample()

        prior_h = src_h[:, 0]
        prior_logits = self.to_prior(prior_h)
        prior_dist = self.make_dist(prior_logits)
        prior_z = prior_dist.rsample()

        h_context_prime = self.to_language_prior(src_h[:, 0])
        h_context_decoder = self.resize_embed(caption_embedding, self.decoder_embed_dim)
        tgt_pred = self.forward_decoder_fut(src_h, h_context_decoder, post_z)
        loss_post = self.forward_loss(tgt_imgs, tgt_pred)
        kl_loss, kl_value = self.compute_kl_loss(post_logits, prior_logits)
        h_context_prime = h_context_prime.view(h_context_decoder.shape)
        context_loss = 1 - torch.nn.functional.cosine_similarity(h_context_prime.squeeze(1), h_context_decoder.squeeze(1), dim=1)
        context_loss = context_loss.mean()

        img_h, mask, ids_restore = self.forward_encoder(tgt_imgs, mask_ratio=self.mask_ratio)
        pred_masked = self.forward_decoder_mae(img_h, ids_restore)
        mae_loss = self.forward_loss(tgt_imgs, pred_masked, mask)

        with torch.no_grad():
            tgt_pred_prior = self.forward_decoder_fut(src_h, h_context_decoder, prior_z)
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


def rsp_cos_joint_self_vit_small_patch8_dec512d8b(**kwargs):
    model = RspCaptionJointSelf(
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


def rsp_cos_joint_self_vit_small_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointSelf(
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


def rsp_cos_joint_self_vit_base_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointSelf(
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


def rsp_cos_joint_self_vit_large_patch16_dec512d8b(**kwargs):
    model = RspCaptionJointSelf(
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
rsp_cos_joint_self_vit_small_patch8 = rsp_cos_joint_self_vit_small_patch8_dec512d8b
rsp_cos_joint_self_vit_small_patch16 = rsp_cos_joint_self_vit_small_patch16_dec512d8b
rsp_cos_joint_self_vit_base_patch16 = rsp_cos_joint_self_vit_base_patch16_dec512d8b
rsp_cos_joint_self_vit_large_patch16 = rsp_cos_joint_self_vit_large_patch16_dec512d8b


