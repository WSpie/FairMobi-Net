import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic(nn.Module):
    def __init__(self, device='cpu'):
        super(Basic, self).__init__()
        self.device = torch.device(device)