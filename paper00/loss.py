import torch.nn as nn
import torch
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth 
        
    def forward(self, inputs, targets):
        # Handle shape mismatch - squeeze the channel dimension if it's 1
        if inputs.dim() == 3 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # Remove channel dimension
        
        # Ensure targets are float and same shape as inputs
        targets = targets.float()
        if targets.shape != inputs.shape:
            targets = targets.view_as(inputs)
        
        # Use BCEWithLogitsLoss which applies sigmoid internally
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # Apply sigmoid for Dice calculation
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # Combine losses
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE