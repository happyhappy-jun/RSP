import torch
import torch.nn as nn

class LinearProbing(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get feature dimension from backbone
        self.feature_dim = backbone.embed_dim
        
        # Add batch norm and linear classification head
        self.bn = nn.BatchNorm1d(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()
        self.head = nn.Sequential(self.bn, self.classifier)
        for _, p in self.bn.named_parameters():
            p.requires_grad = True
        
    def forward(self, x):
        # Get features from backbone
        with torch.no_grad():
            if len(x.shape) == 4:  # Single frame case (B, C, H, W)
                x = x.unsqueeze(1)  # Add time dimension (B, 1, C, H, W)

            B, T, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)  # Reshape to (B*T, C, H, W)
            features = self.backbone.forward_encoder(x, mask_ratio=0)[0]
            cls_token = features[:, 0]
            features = cls_token.reshape(B, T, -1).mean(dim=1)  # Average over frames
        
        # Pass through batch norm and classifier
        logits = self.head(features)
        return logits
