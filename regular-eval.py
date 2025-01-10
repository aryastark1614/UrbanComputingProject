import numpy as np

from transformers import AutoformerForPrediction, AutoformerConfig
import torch
from regular import MultiDataset
from data import load

from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--station', type=str, default=None)
args = parser.parse_args()

config = AutoformerConfig.from_json_file('models/regular_config.json')
model = AutoformerForPrediction(config)
model.load_state_dict(torch.load('models/regular_model.pth', weights_only=True))
model.eval();

train_df = load(years=['2017', '2018', '2019', '2020'], indicators=['NOx', 'NO', 'NO2'])
train_dataset = MultiDataset(train_df, stations=[args.station] if args.station is not None else None, in_days=4, out_days=14)

torch.manual_seed(3141)
val_df = load(years=['2021', '2022'], indicators=['NOx', 'NO', 'NO2'])
val_dataset = MultiDataset(val_df, stations=[args.station] if args.station is not None else None, in_days=4, out_days=14, m=train_dataset.m, s=train_dataset.s)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

losses = []
pbar = tqdm(total=10, leave=False)
for batch_idx, (data, data_time, data_mask, static_real_features, target, target_time) in enumerate(val_loader) :
    outputs = model.generate(
        past_values=data,
        past_time_features=data_time,
        past_observed_mask=data_mask,
        future_time_features=target_time
    )

    pred = outputs.sequences.mean(dim=1)
    
    if (len(pred.shape) < 3) : 
        data = data.view(*data.shape, 1)
        target = target.view(*target.shape, 1)
        pred = pred.view(*pred.shape, 1)
    
    data = (data * train_dataset.s + train_dataset.m)
    target = (target * train_dataset.s + train_dataset.m)
    pred = (pred * train_dataset.s + train_dataset.m)

    # only evaluate the forecast on NOx
    if (pred.shape[-1] > 39) : 
        data = data[:,:,:39]
        target = target[:,:,:39]
        pred = pred[:,:,:39]

    losses.append((target - pred).square().mean().sqrt().item())

    pbar.update()
    if batch_idx >= (10-1) : break

print(f'{args.station}, {np.mean(losses)}, {np.std(losses)}\n')
with open('results.txt', 'a') as handle :
    handle.write(f'{args.station}, {np.mean(losses)}, {np.std(losses)}\n')