import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# 5 * 3 * 8
# 1 * 3 * 8

class dumbmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.LazyLinear(1).cuda()            
    
    def forward(self, input_):
        output = self.fc(input_)

        return output