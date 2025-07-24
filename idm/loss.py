import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLoss(nn.Module):
    def __init__(self, target_precision, mean, std):
        super().__init__()
        self.register_buffer('precision', target_precision.clone().detach())
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('ema', target_precision.clone().detach())
        self.register_buffer('_current_mae', None)
        
    def forward(self, y_pred_norm, y_true_norm):
        # denormalize
        mean = self.mean.to(y_pred_norm.device)
        std = self.std.to(y_pred_norm.device)
        precision = self.precision.to(y_pred_norm.device)
        
        y_pred = y_pred_norm * std + mean
        y_true = y_true_norm * std + mean
        
        abs_error = torch.abs(y_pred - y_true)  # [batch_size, 14]
        current_mae = torch.mean(abs_error, dim=0)  # [14]
        
        # store current MAE(no grad)
        self._current_mae = current_mae.detach().clone().cpu()
        
        # Dynamic Weight Calculation (Core Logic)
        # unmet_mask = (current_mae > precision).float()  # [14]
        unmet_mask = (self.ema > self.precision).float().to(y_pred_norm.device)
        # weights = (1.0 / (precision ** 2 + 1e-8)) * (1.0 + 2.0 * unmet_mask)  # [14]
        weights = (1.0 / (precision + 1e-8)) * unmet_mask  # [14]
        
        # high precision: [0:6], [7:13]
        # low precision: [6:7], [13:14]
        high_precision_indices = torch.cat([torch.arange(0, 6), torch.arange(7, 13)]).to(y_pred.device)
        low_precision_indices = torch.tensor([6, 13]).to(y_pred.device)
        
        # MSE loss for high precision joints
        mse_part = (y_pred[:, high_precision_indices] - y_true[:, high_precision_indices]) ** 2
        
        # Smooth L1 loss for low precision joints
        l1_part = F.smooth_l1_loss(
            y_pred[:, low_precision_indices],
            y_true[:, low_precision_indices],
            reduction='none'
        )
        
        loss = 0.7 * torch.mean(weights[high_precision_indices] * mse_part) + \
               0.3 * torch.mean(weights[low_precision_indices] * l1_part)
        
        return loss

    def update_ema(self):
        """ update EMA for each batch"""
        if self._current_mae is None:
            return
        
        # Dynamic Adjustment of EMA Coefficient:
        # For dimensions that do not meet the standard, use fast updating (alpha=0.2); for dimensions that meet the standard, use slow updating (alpha=0.05).
        alpha = torch.where(
            self._current_mae > self.precision,
            torch.tensor(0.2, device=self.precision.device),
            torch.tensor(0.05, device=self.precision.device)
        )
        
        # EMA update
        self.ema = (1 - alpha) * self.ema + alpha * self._current_mae

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, learning_dim=None):
        super().__init__()
        weights = torch.ones(14)
        wrist_indices = [4, 11]  # weighted joints
        weights[wrist_indices] = 2.0 # wrist: 2
        self.register_buffer('joint_weights', weights)
        self.beta = beta
        
        # Create learning dimension mask
        self.dim_mask = torch.zeros(14, dtype=torch.bool)
        if learning_dim is not None:
            for dim in learning_dim:
                if 0 <= dim < 14:
                    self.dim_mask[dim] = True
        else:
            self.dim_mask.fill_(True)  # Default: learn all dimensions
        self.register_buffer('learning_mask', self.dim_mask)

    def forward(self, pred, target):
        # cal SmoothL1Loss
        diff = torch.abs(pred - target)
        smooth_l1_loss = torch.where(diff < self.beta,
                                   0.5 * diff * diff / self.beta,
                                   diff - 0.5 * self.beta)
        
        learning_mask = self.learning_mask.to(pred.device)
        weights = self.joint_weights.view(1, -1).to(pred.device)
        masked_weights = weights * learning_mask.float()
        weighted_loss = smooth_l1_loss * masked_weights.view(1, -1)
        
        # Only average over dimensions we're learning
        active_dims = learning_mask.sum()
        if active_dims > 0:
            return weighted_loss.sum() / (pred.size(0) * active_dims)
        else:
            return torch.tensor(0.0, device=pred.device)



class WeightedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.ones(14)
        wrist_indices = [4, 11]  # weighted joints
        weights[wrist_indices] = 2.0 # wrist: 2
        self.register_buffer('joint_weights', weights)

    def forward(self, pred, target):
        # Calculate L2 loss (squared error)
        squared_diff = (pred - target) ** 2
        
        # Apply joint weights
        weighted_loss = squared_diff * self.joint_weights.view(1, -1).to(pred.device)
        
        return weighted_loss.mean()
    

class DynamicWeight(nn.Module):
    """ Dynamic Weight"""
    def __init__(self, num_tasks=3):
        self.weights = nn.Parameter(torch.ones(num_tasks))
        
    def forward(self, losses):
        return torch.sum(F.softmax(self.weights,0) * torch.stack(losses))
