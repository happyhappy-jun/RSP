# rsp_vit_small_patch8 = rsp_vit_small_patch8_dec512d8b  # decoder: 512 dim, 8 blocks
# rsp_vit_small_patch16= rsp_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# rsp_vit_base_patch16 = rsp_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# rsp_vit_large_patch16 = rsp_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks


from modeling import models_rsp, models_rsp_caption, models_mse

# Original RSP models
rsp_vit_small_patch8 = models_rsp.rsp_vit_small_patch8_dec512d8b
rsp_vit_small_patch16 = models_rsp.rsp_vit_small_patch16_dec512d8b
rsp_vit_base_patch16 = models_rsp.rsp_vit_base_patch16_dec512d8b
rsp_vit_large_patch16 = models_rsp.rsp_vit_large_patch16_dec512d8b

# Caption models
rsp_vit_small_patch8_caption = models_rsp_caption.rsp_vit_small_patch8_dec512d8b
rsp_vit_small_patch16_caption = models_rsp_caption.rsp_vit_small_patch16_dec512d8b
rsp_vit_base_patch16_caption = models_rsp_caption.rsp_vit_base_patch16_dec512d8b
rsp_vit_large_patch16_caption = models_rsp_caption.rsp_vit_large_patch16_dec512d8b

# MSE variants
rsp_mse_vit_small_patch8 = models_mse.rsp_mse_vit_small_patch8_dec512d8b
rsp_mse_vit_small_patch16 = models_mse.rsp_mse_vit_small_patch16_dec512d8b
rsp_mse_vit_base_patch16 = models_mse.rsp_mse_vit_base_patch16_dec512d8b
rsp_mse_vit_large_patch16 = models_mse.rsp_mse_vit_large_patch16_dec512d8b
