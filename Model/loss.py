import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def cross_entropy_loss(output, target):
    return nn.CrossEntropyloss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def mae_loss(output, target):
    return F.l1_loss(output, target)

def smooth_l1_loss(output, target):
    return F.smooth_l1_loss(output, target)

