import torch.nn as nn
import torch
from transformers import ResNetModel, ResNetConfig
from torchvision.transforms import Resize, ToTensor

class ResnetWithSplitLines(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.region_models = nn.ModuleList([
            self._build_region_model(output_dim=6),
            self._build_region_model(output_dim=1),
            self._build_region_model(output_dim=6),
            self._build_region_model(output_dim=1)
        ])
    
    def _build_region_model(self, output_dim):
        return ResNet50Regressor(output_dim)

    def forward(self, region_images):
        # Input: region_images: [4, B, 3, H, W]
        
        outputs = []
        for i in range(4):
            out = self.region_models[i](region_images[i])
            outputs.append(out)
        
        final_output = torch.cat([
            outputs[0],  # 0-6
            outputs[1],  # 6-7
            outputs[2],  # 7-13
            outputs[3]   # 13-14
        ], dim=1)
        
        return final_output  # [B, 14]
    
class ResNet50Regressor(nn.Module):
    def __init__(self, output_dim=14, *args, **kwargs):
        super(ResNet50Regressor, self).__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.regressor = nn.Linear(2048, output_dim)
        self.transform =  nn.Sequential(
            Resize((224, 224)),
        )

    def forward(self, pixel_values):
        pixel_values = self.transform(pixel_values)
        outputs = self.resnet(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        flattened_output = pooled_output.view(pooled_output.size(0), -1) 
        regression_output = self.regressor(flattened_output)
        return regression_output