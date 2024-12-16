import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DBloss(nn.Module):
    def __init__(self, class_freq, train_num):
        super(DBloss, self).__init__()
        
        self.class_freq = torch.from_numpy(np.asarray(class_freq)).float().cuda()
        self.train_num = train_num
        self.num_classes = self.class_freq.shape[0]
        
        # Rebalance parameters
        self.map_alpha = 0.1
        self.map_beta = 10.0
        self.map_gamma = 0.9
        
        # Focal loss parameters
        self.focal_alpha = 0.5
        self.focal_gamma = 2
        
        # Logit regularization parameters
        self.neg_scale = 2.0
        self.init_bias = 0.05
        
        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq
        
        self.init_bias = -torch.log(self.train_num / self.class_freq - 1) * self.init_bias

    def forward(self, logits, labels):
        # Apply logit regularization
        logits = self.logit_reg_functions(labels, logits)
        
        # Calculate focal loss
        probs = torch.sigmoid(logits)
        focal_weight = (1 - probs) ** self.focal_gamma
        focal_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        focal_loss = self.focal_alpha * focal_weight * focal_loss
        
        # Calculate rebalance weight
        rebalance_weight = self.rebalance_weight(labels)
        
        # Combine losses
        loss = rebalance_weight * focal_loss
        
        return loss.mean()

    def logit_reg_functions(self, labels, logits):
        logits += self.init_bias
        logits = logits * (1 - labels) * self.neg_scale + logits * labels
        return logits

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight