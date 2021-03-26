import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor([[1,2, 12,34, 56,78, 90,80]])

model1 = nn.Embedding(100, 7, padding_idx=0)
model2 = nn.LSTM(input_size=7, hidden_size=1, num_layers=1, batch_first=True)

out1 = model1(x)
# out2 = model2(out1)
out, (ht, ct) = model2(out1)
print(out)