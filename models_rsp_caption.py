from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from timm.models.vision_transformer import PatchEmbed, Block, CrossAttention

from util.pos_embed import get_2d_sincos_pos_embed

class RSPCaption(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        context_emb_dim=1536,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        stoch=32,
        discrete=32,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=True,
        kl_scale=0.01,
        kl_balance=0.2,
        kl_freebit=0.1,
        mask_ratio=0.75,
        noise_scale=0.5,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Determine the size of stochastic variable
        stoch_size = stoch * discrete if discrete != 0 else stoch * 2

        self.to_posterior = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, stoch_size),
        )

        self.to_prior = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, stoch_size),
        )
        
        self.decoder_embed_dim = decoder_embed_dim
        self.to_language_prior = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, self.decoder_embed_dim),
        )
        
        self.decoder_embed_mae = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_deter = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_stoch = nn.Linear(stoch_size, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.context_proj = nn.Linear(context_emb_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        self.image_type_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
        )
        nn.init.normal_(self.image_type_embed, std=0.02)
        self.language_type_embed = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim), requires_grad=True
        )
        nn.init.normal_(self.language_type_embed, std=0.02)
        
        self.decoder_blocks = nn.ModuleList(
            [
                CrossAttention(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        self.norm_pix_loss = norm_pix_loss
        self.kl_scale = kl_scale
        self.kl_balance = kl_balance
        self.kl_freebit = kl_freebit

        self.stoch = stoch
        self.discrete = discrete

        self.noise_scale = noise_scale
        self.mask_ratio = mask_ratio

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
                
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def perturb(self, x):
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise

    def forward_encoder(self, imgs, mask_ratio=0.0):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio != 0.0:
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

    def forward_embedding(self, h_context):
        batch_size = h_context.size(0)
        h_context = h_context.view(batch_size, -1, h_context.size(-1))
        h_context = self.context_proj(h_context)
        h_context = h_context + self.language_type_embed
        return h_context

    def forward_decoder_fut(self, h, h_context, z):
        h = self.decoder_embed_deter(h)
        h = h + self.decoder_pos_embed
        h = h + self.image_type_embed

        h_concat = torch.cat([h, h_context], dim=1)
    
        z = self.decoder_embed_stoch(z)
        feat = torch.cat([z, h_concat], dim=1)

        mask_tokens = self.mask_token.repeat(h.shape[0], h.shape[1], 1)
        x = mask_tokens + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, kvx=feat)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_mae(self, h, ids_restore):
        h = self.decoder_embed_mae(h)
        mask_tokens = self.mask_token.repeat(h.shape[0], ids_restore.shape[1] + 1 - h.shape[1], 1)
        h_ = torch.cat([h[:, 1:, :], mask_tokens], dim=1)  # no cls token
        h_ = torch.gather(h_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, h.shape[2]))  # unshuffle
        h = torch.cat([h[:, :1, :], h_], dim=1)  # append cls token
        h = h + self.decoder_pos_embed

        # embed tokens
        mask_tokens = self.mask_token.repeat(h.shape[0], h.shape[1], 1)
        x = mask_tokens + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, kvx=h)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask=None):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        recon_loss = (pred - target) ** 2
        if mask is not None:
            recon_loss = recon_loss.mean(dim=-1)  # [N, L], mean loss per patch
            recon_loss = (recon_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            recon_loss = recon_loss.mean()

        return recon_loss

    def make_dist(self, logits):
        if self.discrete != 0:
            logits = logits.reshape([-1, self.stoch, self.discrete])
            dist = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)
        else:
            mean, std = torch.split(logits, 2, -1)
            dist = td.Normal(mean, std)
        return dist

    def compute_kl_loss(self, post_logits, prior_logits):
        balance = self.kl_balance
        freebit = self.kl_freebit
        post_to_prior_kl = td.kl_divergence(
            self.make_dist(post_logits), self.make_dist(prior_logits.detach())
        )
        prior_to_post_kl = td.kl_divergence(
            self.make_dist(post_logits.detach()), self.make_dist(prior_logits)
        )
        kl_value = (
            post_to_prior_kl * balance + prior_to_post_kl * (1.0 - balance)
        ).mean()
        kl_loss = torch.maximum(kl_value, torch.ones_like(kl_value) * freebit)
        return kl_loss, kl_value
    
    def context_kl_loss(self, h_context, h_context_prime):
        log_prob_prime = F.log_softmax(h_context_prime, dim=-1)
        prob_context = F.softmax(h_context.squeeze(1), dim=-1)
        
        return F.kl_div(
            log_prob_prime,
            prob_context,
            reduction='batchmean',
            log_target=False
        )

    def forward(self, src_imgs, tgt_imgs, embedding, epoch):
        # Extract embeddings
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

        h_context = self.forward_embedding(embedding)
        h_context_prime = self.to_language_prior(src_h[:, 0])
        
        tgt_pred = self.forward_decoder_fut(src_h, h_context, post_z)
        loss_post = self.forward_loss(tgt_imgs, tgt_pred)
        kl_loss, kl_value = self.compute_kl_loss(post_logits, prior_logits)
        context_kl = self.context_kl_loss(h_context, h_context_prime)

        # MAE
        img_h, mask, ids_restore = self.forward_encoder(tgt_imgs, mask_ratio=self.mask_ratio)
        pred_masked = self.forward_decoder_mae(img_h, ids_restore)
        mae_loss = self.forward_loss(tgt_imgs, pred_masked, mask)

        with torch.no_grad():
            tgt_pred_prior = self.forward_decoder_fut(src_h, h_context, prior_z)
            loss_prior = self.forward_loss(tgt_imgs, tgt_pred_prior)

        loss = loss_post + self.kl_scale * kl_loss + self.kl_scale * context_kl + mae_loss

        return loss, tgt_pred, (loss_post, loss_prior, kl_loss, kl_value, mae_loss, context_kl)

def rsp_vit_small_patch8_caption(**kwargs):
    model = RSPCaption(
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

def rsp_vit_small_patch16_caption(**kwargs):
    model = RSPCaption(
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

def rsp_vit_base_patch16_caption(**kwargs):
    model = RSPCaption(
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

def rsp_vit_large_patch16_caption(**kwargs):
    model = RSPCaption(
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
