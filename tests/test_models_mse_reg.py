import torch
import pytest
from modeling.models_mse_reg import RspCaptionMseReg

@pytest.fixture
def model_config():
    return {
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'mlp_ratio': 4,
        'mse_scale': 1.0,
        'enable_rms_norm': True,
        'embed_scale_factor': 1.0,
        'num_register_tokens': 4,
        'stoch_size': 128  # Add this if needed in your base class
    }

@pytest.fixture
def sample_inputs():
    batch_size = 2
    img_size = 224
    embedding_dim = 512
    
    src_imgs = torch.randn(batch_size, 3, img_size, img_size)
    tgt_imgs = torch.randn(batch_size, 3, img_size, img_size)
    embedding = torch.randn(batch_size, embedding_dim)
    
    return src_imgs, tgt_imgs, embedding

def test_model_initialization(model_config):
    model = RspCaptionMseReg(**model_config)
    assert model is not None
    assert hasattr(model, 'register_token')
    assert model.register_token.shape == (model_config['num_register_tokens'], model_config['embed_dim'])

def test_forward_pass(model_config, sample_inputs):
    model = RspCaptionMseReg(**model_config)
    src_imgs, tgt_imgs, embedding = sample_inputs
    
    # Test forward pass
    loss, tgt_pred, detailed_loss = model(src_imgs, tgt_imgs, embedding, epoch=0)
    
    # Basic assertions
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert isinstance(detailed_loss, dict)
    assert all(k in detailed_loss for k in [
        'loss_post', 'loss_prior', 'loss_kl', 'kl', 'context_loss', 'loss_mae'
    ])

def test_encoder(model_config, sample_inputs):
    model = RspCaptionMseReg(**model_config)
    src_imgs, _, _ = sample_inputs
    
    # Test encoder
    encoded, mask, ids_restore = model.forward_encoder(src_imgs)
    
    # Check shapes
    expected_seq_length = (model_config['num_register_tokens'] + 1 + 
                          (224 // model_config['patch_size']) ** 2)
    assert encoded.shape == (src_imgs.shape[0], expected_seq_length, model_config['embed_dim'])

def test_decoder(model_config, sample_inputs):
    model = RspCaptionMseReg(**model_config)
    src_imgs, _, embedding = sample_inputs
    
    # Example of indexing specific patches
    def test_patch_indexing():
        # Create a sample tensor [Batch, patches, embed_dim]
        batch_size = 2
        num_patches = 10
        embed_dim = 384
        x = torch.randn(batch_size, num_patches, embed_dim)
        
        # Create indices [0, 3, 4, 5, ..., num_patches-1]
        indices = torch.cat([torch.tensor([0]), torch.arange(3, num_patches)])
        
        # Method 1: Using fancy indexing
        selected_patches = x[:, indices, :]
        assert selected_patches.shape == (batch_size, len(indices), embed_dim)
        
        # Method 2: Using index_select
        selected_patches_alt = x.index_select(1, indices)
        assert torch.equal(selected_patches, selected_patches_alt)
        
        # Verify first patch is 0 and subsequent patches start from 3
        assert torch.equal(selected_patches[:, 0, :], x[:, 0, :])  # First patch is 0
        assert torch.equal(selected_patches[:, 1:, :], x[:, 3:, :])  # Rest starts from 3
        
        return selected_patches
    
    # Run the indexing test
    patches = test_patch_indexing()
    
    # Get encoded features
    h, _, _ = model.forward_encoder(src_imgs)
    
    # Create dummy inputs for decoder
    batch_size = src_imgs.shape[0]
    h_context = torch.randn(batch_size, 1, model_config['decoder_embed_dim'])
    z = torch.randn(batch_size, model_config['stoch_size'])
    
    # Test decoder
    decoded = model.forward_decoder_fut(h, h_context, z)
    
    # Check output shape
    expected_patch_size = (224 // model_config['patch_size']) ** 2
    assert decoded.shape == (batch_size, expected_patch_size, 3 * model_config['patch_size']**2)

if __name__ == '__main__':
    pytest.main([__file__])