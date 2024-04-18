import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

#交叉熵损失函数
def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

################################
def mse_loss(output, target):
    return F.mse_loss(output, target)

def mae_loss(output, target):
    return F.l1_loss(output, target)

def smooth_l1_loss(output, target):
    return F.smooth_l1_loss(output, target)

# def mae_loss(output, target):
#     return mean_absolute_error(output, target)

# def mse_loss(output, target):
#     return mean_squared_error(output, target)

# def rmse_loss(output, target):
#     return np.sqrt(mean_squared_error(output, target))

# def huber_loss(output, target):
#     return F.smooth_l1_loss(output, target)