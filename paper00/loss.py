import torch.nn as nn
import torch
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth 
        
    def forward(self, inputs, targets, weight=None):
        # Handle shape mismatch - squeeze the channel dimension if it's 1
        if inputs.dim() == 3 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # Remove channel dimension
        
        # Ensure targets are float and same shape as inputs
        targets = targets.float()
        if targets.shape != inputs.shape:
            targets = targets.view_as(inputs)
        
        # Use BCEWithLogitsLoss which applies sigmoid internally
        if weight is None:
            BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        else:
            BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean', weight=weight)

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
    
class ContinuityLoss(nn.Module):
    def __init__(self, smooth=1, lambda_smooth=0.1):
        super(ContinuityLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        self.DiceLoss = DiceBCELoss(smooth=smooth)

    def continuity_loss(self, preds):
        # Handle shape mismatch - squeeze the channel dimension if it's 1
        if preds.dim() == 3 and preds.size(1) == 1:
            preds = preds.squeeze(1)  # Remove channel dimension

        return torch.mean(torch.abs(preds[:, 1:] - preds[:, :-1]))
    
    def forward(self, inputs, targets, weight=None):
        dice_loss = self.DiceLoss(inputs, targets, weight=weight)
        continuity_loss = self.continuity_loss(inputs)

        return dice_loss + self.lambda_smooth * continuity_loss
    
class WeightedCenterLoss(nn.Module):
    def __init__(self, smooth=1, lambda_smooth=0.1):
        super(WeightedCenterLoss, self).__init__()
        self.continuity_loss = ContinuityLoss(smooth=smooth, lambda_smooth=lambda_smooth)
        self.device = None

    def compute_weight(self, labels, center_weight_factor=1.6):
        """
        Create a weight mask that emphasizes the center 2/3 of gestures when fully contained.
        Handles multiple gestures per window - only weights center for fully contained gestures.
        
        Args:
            predictions: Model predictions (batch_size, window_size, num_classes)
            labels: Ground truth labels (batch_size, window_size)
            window_size: Length of the time window
            center_weight_factor: How much more to weight center (1.6 = 60% more)
        
        Returns:
            weight_mask: Tensor of weights for each position
        """
        weight_mask = torch.ones_like(labels, dtype=torch.float32)
        if labels.dim() == 3:
            window_size = labels.size(2)
        else:
            window_size = labels.size(1)

        self.device = labels.get_device()

        
        for batch_idx in range(labels.shape[0]):
            batch_labels = labels[batch_idx]
            
            # Find all gesture segments by detecting class changes
            # Pad with background class (0) to detect boundary gestures
            padded_labels = torch.cat([(torch.tensor([0])).to(self.device), batch_labels, (torch.tensor([0])).to(self.device)])
            
            # Find where class changes occur
            class_changes = torch.where(padded_labels[1:] != padded_labels[:-1])[0]
            
            # Group changes into start/end pairs for each gesture
            for i in range(0, len(class_changes), 2):
                if i + 1 >= len(class_changes):
                    break
                    
                gesture_start = class_changes[i].item()
                gesture_end = class_changes[i + 1].item() - 1
                
                # Skip if this is background (class 0)
                if batch_labels[gesture_start] == 0:
                    continue
                
                # Check if gesture is fully contained (doesn't touch window boundaries)
                is_fully_contained = gesture_start > 0 and gesture_end < window_size - 1
                
                if is_fully_contained:
                    # Calculate center 2/3 region
                    gesture_length = gesture_end - gesture_start + 1
                    center_start = gesture_start + gesture_length // 6  # Skip first 1/6
                    center_end = gesture_end - gesture_length // 6      # Skip last 1/6
                    
                    # Apply higher weight to center region
                    weight_mask[batch_idx, center_start:center_end+1] = center_weight_factor
        
        return weight_mask


    def forward(self, inputs, target):
        weights = self.compute_weight(target)
        res =  self.continuity_loss(inputs, target, weights)
        return res

