import torch
import constants

x = torch.tensor([-0.0418, 0.0123, 0.0645, 0.3298, 0.0349])
_, act_max_idx = torch.max(x, dim=0)
_, act_min_idx = torch.min(x, dim=0)
act_max = constants.action_to_name(act_max_idx.item())
act_min = constants.action_to_name(act_min_idx.item())

print(act_max)
print(act_min)