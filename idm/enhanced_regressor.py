import torch
import torch.nn as nn
from transformers import Dinov2WithRegistersModel
import torch.nn.functional as F


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=6, dilation=6),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=3, dilation=3),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        )
        self.coord_conv = CoordConv(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [B, 37, 37, 768]
        x = x.permute(0,3,1,2)  # [B,768,H,W]
        x = self.conv_block(x)  # [B,256,H,W]
        x = self.coord_conv(x)
        return x  # [B,256,37,37]


class CoordConv(nn.Module):
    """coordinate feature enhancement"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels+2, out_channels, 3, padding=1)
        
    def forward(self, x):
        batch, _, h, w = x.shape
        x_coord = torch.linspace(-1, 1, w).repeat(h,1)
        y_coord = torch.linspace(-1, 1, h).repeat(w,1).t()
        grid = torch.stack([x_coord, y_coord], dim=0).unsqueeze(0).repeat(batch,1,1,1)
        grid = grid.to(x.device)
        x = torch.cat([x, grid], dim=1)
        return self.conv(x)
    

class OrientationAwareBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.directional_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1),  # out: [B, 64, 37, 37]
            nn.Conv2d(in_channels, in_channels//4, 3, padding=2, dilation=2),  # out: [B, 64, 37, 37]
            nn.Conv2d(in_channels, in_channels//4, (1,3), padding=(0,1)),  # out: [B, 64, 37, 37]
            nn.Conv2d(in_channels, in_channels//4, (3,1), padding=(1,0))  # out: [B, 64, 37, 37]
        ])
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spatial_feat):
        # spatial_feat: [B, 256, 37, 37]
        features = [conv(spatial_feat) for conv in self.directional_convs]  # [B, 64, 37, 37] for each feature
        combined = torch.cat(features, dim=1)  # [B, 256, 37, 37]
        attn = self.attention(spatial_feat)  # [B, 1, 37, 37]
        return combined * attn  # [B, 256, 37, 37]


class EnhancedRegressor(nn.Module):
    def __init__(self, 
                 use_depth = False,
                 dinov2_name: str = "facebook/dinov2-base",
                 freeze_dinov2 = False,
                 output_dim: int = 14):
        super().__init__()
        
        self.dino_model = Dinov2WithRegistersModel.from_pretrained(dinov2_name)
        self.output_dim = output_dim
        self.use_depth = use_depth
        
        hidden_size = self.dino_model.config.hidden_size  # 768
        patch_size = self.dino_model.config.patch_size    # 14
        
        self.dino_wh = 518
        self.token_size = self.dino_wh // patch_size  # 37 when dino_wh=518
        
        if freeze_dinov2:
            for param in self.dino_model.parameters():
                param.requires_grad_(False)

        self.spatial_extractor = SpatialFeatureExtractor(
            in_dim=hidden_size, 
            hidden_dim=256
        )

        self.orientation_block = OrientationAwareBlock(256)

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256*3, 512, 1),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        
        self.joint_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [B, 768, 1, 1]
                nn.Flatten(),  # [B, 768]
                nn.Linear(768, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(output_dim)
        ])

    def forward(self, images):
        # images shape is [B, 3, 518, 518]

        outputs = self.dino_model(images) 
        num_register_tokens = self.dino_model.config.num_register_tokens  # 4
        patch_embeddings = outputs.last_hidden_state[:, num_register_tokens+1:, :]
        patch_embeddings = patch_embeddings.reshape(-1, self.token_size, self.token_size, 768)  # [B, 37, 37, 768]
        
        spatial_feat = self.spatial_extractor(patch_embeddings)  # [B, 256, 37, 37]  (token_size=37)
        orientation_feat = self.orientation_block(spatial_feat)  # [B, 256, 37, 37]
        
        enhanced_feat = spatial_feat + orientation_feat  # [B, 256, 37, 37]
        
        down_feat = F.avg_pool2d(enhanced_feat, 3, stride=2, padding=1)  # [B, 256, 18, 18]
        up_feat = F.interpolate(enhanced_feat, scale_factor=2)  # [B, 256, 74, 74]
        fused_feat = torch.cat([
            enhanced_feat,
            F.interpolate(down_feat, size=(self.token_size, self.token_size)),
            F.adaptive_avg_pool2d(up_feat, (self.token_size, self.token_size))
        ], dim=1)  # [B, 768, 37, 37]
        
        predictions = [branch(fused_feat) for branch in self.joint_branches]  # [B, 1, 37, 37] for each branch
        return torch.cat(predictions, dim=1)  # [B, 14]
