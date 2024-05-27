import torch
import torch.nn as nn
import torch.nn.functional as F

class HuberLoss(nn.Module):
    def __init__(self, delta=1.5):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        i=0
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadratic = torch.where(abs_error < self.delta, 0.5 * abs_error ** 2, self.delta * (abs_error - 0.5 * self.delta))
        #quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        i += 1
        #print('loss',i,loss)
        return torch.mean(loss)



#
