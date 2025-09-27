import torch
import constants
import torch.nn.functional as F

x = torch.tensor([1.8239, 6.7343, -0.1782, -5.0324, -2.6164])
sm = torch.softmax(x, dim = 0)
_, act_max_idx = torch.max(x, dim=0)
_, act_min_idx = torch.min(x, dim=0)
act_max = constants.action_to_name(act_max_idx.item())
act_min = constants.action_to_name(act_min_idx.item())

print(act_max)
print(act_min)
print(sm)

inv = 1.0 / (sm + 1e-9)
print(inv)
actions = F.softmax(inv, dim=0)
print(actions)