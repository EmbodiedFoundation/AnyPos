import torch
import torch.nn as nn
import torch.nn.functional as F

# you can use: 
from torchvision.ops import DeformConv2d  # But you sholud give offset as arg of forward()


def rotate_tensor(x, angle_degrees):
    """Rotate a 4D tensor (B,C,H,W) by the given angle in degrees
    
    Args:
        x: Input tensor of shape [B,C,H,W]
        angle_degrees: Rotation angle in degrees
        
    Returns:
        Rotated tensor of same shape as input
    """
    B, C, H, W = x.shape  # [8, 256, 37, 37]
    device = x.device
    
    # Convert angle to radians
    angle_rad = torch.deg2rad(torch.tensor(angle_degrees))
    
    # Create rotation matrix components
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    
    # Create affine transformation matrix for rotation around center
    # We need a 2x3 matrix for 2D transformations in the form:
    # [a b c]
    # [d e f]
    # Where [a b; d e] is the rotation part and [c; f] is translation
    
    # For rotation around center, we need a translation component
    # First create the rotation matrix
    rotation_matrix = torch.zeros(2, 3, device=device)
    rotation_matrix[0, 0] = cos_theta
    rotation_matrix[0, 1] = -sin_theta
    rotation_matrix[1, 0] = sin_theta
    rotation_matrix[1, 1] = cos_theta
    
    # The grid coordinates in affine_grid are in [-1, 1] range
    # No translation needed for rotation around center in normalized coordinates
    
    # Create batch of transformation matrices
    theta = rotation_matrix.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, 3]
    
    # Apply rotation around center using grid sampling
    grid = F.affine_grid(theta, x.size(), align_corners=True)  # [B, H, W, 2]
    
    # Sample from the input using the grid
    rotated = F.grid_sample(x, grid, align_corners=True, mode='bilinear')  # [B, C, H, W]
    
    return rotated


class DirectionAwareDecoder(nn.Module):
    def __init__(self, in_channels, angle_num=4):
        super().__init__()
        # Direction convolution group
        # directions = [1, 2, 3, 5, 7, 11, 13, 17]
        directions = [1, 2, 3, 6]
        conv_channels = 256
        self.direction_convs = nn.ModuleList([
            nn.Conv2d(in_channels, conv_channels // len(directions), 3, padding=d, dilation=d) 
            for d in directions
        ])
        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        # o = [i + 2*p - 3 - 2*(d-1)]/1 + 1 = i + 2*p - 3 - 2*(d-1) + 1 =(p==d) i
        
        # Deformable convolution
        self.deform_conv = DeformConv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.offset_conv = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=2 * 3 * 3,  # 2*K*K (K=3)
            kernel_size=3,
            padding=1
        )
        self.modulation_conv = nn.Sequential(
            nn.Conv2d(conv_channels, 3*3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Angle-sensitive pooling
        self.angle_pool = nn.ModuleDict({
            str(a): nn.AdaptiveAvgPool2d(1) 
            for a in range(0, 180, 180//angle_num)  # angles
        })
        
    def forward(self, x):
        # Multi-direction feature extraction
        directional_feats = [conv(x) for conv in self.direction_convs]  # 4 x [B, 64, H, W]
        x = torch.cat(directional_feats, dim=1)  # [B, 256, H, W]
        
        # get offset and modulation
        offsets = self.offset_conv(x)  # [B, 2*3*3, H, W]
        modulation = self.modulation_conv(x)  # [B, 9, H, W]
        
        # apply modulation
        B, C, H, W = x.shape
        x_modulated = x * modulation.view(B, 1, 9, H, W).mean(dim=2)
        
        # apply deformable convolution
        x_deformed = self.deform_conv(
            x_modulated, 
            offsets
        )  # [B, 256, H, W]
        
        # Direction-sensitive pooling
        angle_features = []
        for angle, pool in self.angle_pool.items():
            rotated = rotate_tensor(x_deformed, float(angle))  # [B, 256, H, W]
            angle_features.append(pool(rotated))  # [B, 256, 1, 1]
        
        # Concatenate features from different angles
        pooled = torch.cat(angle_features, dim=3)  # [B, 256, 1, 4] - 4 angles(0,45,90,135)
        return pooled.flatten(1)  # [B, 1024]

def generate_heatmap(positions, img_size=(224, 224), sigma=4.0):
    """Generate heatmaps from joint positions
    
    Args:
        positions: Tensor of joint positions with shape [B, num_joints, 2]
                  where each position is (x, y) in pixel coordinates
        img_size: Output heatmap size (H, W)
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Tensor of heatmaps with shape [B, num_joints, H, W]
    """
    B, num_joints, _ = positions.shape  # [B, J, 2]
    H, W = img_size
    device = positions.device
    
    # Create coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),  # [H]
        torch.arange(W, device=device)   # [W]
    )  # y_grid: [H, W], x_grid: [H, W]
    
    # Stack and reshape for broadcasting
    grid = torch.stack([x_grid, y_grid], dim=2)  # [H, W, 2]
    grid = grid.view(1, 1, H, W, 2).expand(B, num_joints, -1, -1, -1)  # [B, J, H, W, 2]
    
    # Reshape positions for broadcasting
    positions = positions.view(B, num_joints, 1, 1, 2)  # [B, J, 1, 1, 2]
    
    # Calculate squared distances
    diff = grid - positions  # [B, J, H, W, 2]
    dist_sq = torch.sum(diff * diff, dim=-1)  # [B, J, H, W]
    
    # Create Gaussian heatmap
    heatmaps = torch.exp(-dist_sq / (2 * sigma * sigma))  # [B, J, H, W]
    
    return heatmaps


class OrientationAwareHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Direction field prediction branch
        self.orientation_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 28, 1)  # 14 joints × [cosθ, sinθ]
        )
        
        # Heatmap integration branch
        self.heatmap_head = nn.Conv2d(feature_dim, 14, 3, padding=1)
        
        # Physical constraint fusion layer
        # self.fusion = nn.Linear(14*3, 14)  # (heatmap + orientation)
        self.attention_fusion = AttentionFusion(14)

    def forward(self, patch_emb, patch_size, hw_size):
        # Spatial feature reconstruction
        B, L, C = patch_emb.shape  # [B, L, C]
        H = W = hw_size  # For example, 37 (518/14)
        feat_map = patch_emb.view(B, H, W, C).permute(0,3,1,2)  # [B, C, H, W]
        
        # Direction field prediction
        orientation = self.orientation_head(feat_map)  # [B, 28, H, W]
        cos_sin = orientation.view(B, 14, 2, H, W)  # [B, 14, 2, H, W]
        
        # Heatmap integration
        heatmaps = torch.softmax(self.heatmap_head(feat_map).view(B, 14, -1), -1)  # [B, 14, H*W]
        angle_weights = (heatmaps.unsqueeze(2) * cos_sin.view(B, 14, 2, -1)).sum(-1)  # [B, 14, 2]
        
        # Angle calculation
        angles = torch.atan2(angle_weights[:,:,1], angle_weights[:,:,0])  # [B, 14]
        
        # Physical constraint fusion
        # joint_features = torch.cat([angles, heatmaps.mean(-1), heatmaps.max(-1)[0]], dim=-1)  # [B, 14*3]
        # return self.fusion(joint_features) + angles  # [B, 14]
        return self.attention_fusion(angles, heatmaps.mean(-1), heatmaps.max(-1)[0])  # [B, 14]


class KinematicsLayer(nn.Module):
    def __init__(self, link_lengths=None, num_joints=14):
        super().__init__()
        # Learnable link length parameters
        self.links = nn.Parameter(torch.randn(num_joints) if link_lengths is None 
                                else torch.tensor(link_lengths))
        
    def forward(self, angles):
        # Implement differentiable forward kinematics (2D simplified version)
        B = angles.size(0)  # [B, num_joints]
        positions = []
        current_pos = torch.zeros(B, 2, device=angles.device)  # [B, 2]
        current_angle = torch.zeros_like(angles[:,0])  # [B]
        
        for i in range(angles.size(1)):  # num_joints
            current_angle = current_angle + angles[:,i]
            delta_x = self.links[i] * torch.cos(current_angle)  # [B]
            delta_y = self.links[i] * torch.sin(current_angle)  # [B]
            current_pos = current_pos + torch.stack([delta_x, delta_y], -1)
            positions.append(current_pos.clone())
            
        positions = torch.stack(positions, 1)  # [B, num_joints, 2]
        return positions.view(B, -1)  # [B, num_joints*2]


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim*3, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1))
        
    def forward(self, a, b, c):
        concat = torch.cat([a, b, c], dim=-1)
        weights = self.attn(concat)
        return weights[:,0:1]*a + weights[:,1:2]*b + weights[:,2:3]*c
