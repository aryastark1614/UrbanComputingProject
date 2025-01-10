import torch
import numpy as np
from tqdm import tqdm

from data import load
from regular import MultiDataset

in_days = 4
out_days = 14
batch_size = 8

train_df = load(years=['2017', '2018', '2019', '2020'], indicators=['NOx'])
train_dataset = MultiDataset(train_df, in_days=in_days, out_days=out_days)

torch.manual_seed(3141)
val_df = load(years=['2021', '2022'], indicators=['NOx'])
val_dataset = MultiDataset(val_df, in_days=in_days, out_days=out_days, m=train_dataset.m, s=train_dataset.s)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

losses = []
for batch_idx, (data, data_time, data_mask, static_real_features, target, target_time) in enumerate(val_loader) :
    if (len(data.shape) < 3) : 
        data = data.view(*data.shape, 1)
        target = target.view(*target.shape, 1)

    data = (data * train_dataset.s + train_dataset.m)
    target = (target * train_dataset.s + train_dataset.m)
    
    losses.append((target - data.mean(dim=1).view(8,1,39)).square().mean().sqrt().item())
    if batch_idx >= (10_000-1) : break

print(np.mean(losses))