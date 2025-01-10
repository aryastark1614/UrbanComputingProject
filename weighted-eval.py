import numpy as np

from transformers import AutoformerForPrediction, AutoformerConfig
import torch
from weighted import WeightedDataset
from data import load

from tqdm import tqdm

from argparse import ArgumentParser

config = AutoformerConfig.from_json_file('models/weighted_config.json')
model = AutoformerForPrediction(config)
model.load_state_dict(torch.load('models/weighted_model.pth', weights_only=True))
model.eval();

torch.manual_seed(3141)
val_df = load(years=['2021', '2022'], indicators=['NOx', 'NO', 'NO2'])
val_dataset = WeightedDataset(val_df, in_days=4, out_days=14)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

losses = []
pbar = tqdm(total=40, leave=False)
for batch_idx, (data, data_time, data_mask, static_real, target, target_time) in enumerate(val_loader) :
    outputs = model.generate(
        past_values=data,
        past_time_features=data_time,
        past_observed_mask=data_mask,
        static_real_features=static_real,
        future_time_features=target_time
    )

    pred = outputs.sequences.mean(dim=1)
    
    if (len(pred.shape) < 3) : 
        data = data.view(*data.shape, 1)
        target = target.view(*target.shape, 1)
        pred = pred.view(*pred.shape, 1)
    
    data = data[:,:,0]
    target = target[:,:,0]
    pred = pred[:,:,0]

    losses.append((target - pred).square().mean().sqrt().item())
    # print(losses[-1])
    pbar.update()
    if batch_idx >= (40-1) : break

print(f'weighted, {np.mean(losses)}, {np.std(losses)}\n')
with open('results.txt', 'a') as handle :
    handle.write(f'weighted, {np.mean(losses)}, {np.std(losses)}\n')