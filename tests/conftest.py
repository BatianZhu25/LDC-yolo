import torch

# Avoid excessive CPU threading for small model verification inputs.
torch.set_num_threads(2)
