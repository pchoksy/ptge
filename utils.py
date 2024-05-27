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



# class HuberLoss(nn.Module):
#     def __init__(self, delta=1.5):
#         super(HuberLoss, self).__init__()
#         self.delta = delta

#     def forward(self, y_pred, y_true):
#         error = y_true - y_pred
#         abs_error = torch.abs(error)
        
#         # Compute quadratic loss for x and y components separately
#         quadratic_x = torch.where(abs_error[:, 0] < self.delta, 0.5 * abs_error[:, 0] ** 2, self.delta * (abs_error[:, 0] - 0.5 * self.delta))
#         quadratic_y = torch.where(abs_error[:, 1] < self.delta, 0.5 * abs_error[:, 1] ** 2, self.delta * (abs_error[:, 1] - 0.5 * self.delta))

#         # Compute linear loss for x and y components separately
#         linear_x = abs_error[:, 0] - quadratic_x
#         linear_y = abs_error[:, 1] - quadratic_y

#         # Combine quadratic and linear losses for both components
#         loss_x = 0.5 * quadratic_x ** 2 + self.delta * linear_x
#         loss_y = 0.5 * quadratic_y ** 2 + self.delta * linear_y

#         # Compute total loss by summing losses for both components
#         total_loss = torch.mean(loss_x + loss_y)
#         print(total_loss)
#         return total_loss

