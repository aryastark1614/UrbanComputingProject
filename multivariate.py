import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from transformers import AutoformerConfig, AutoformerForPrediction

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import argparse
import sys
import platform
from glob import glob
from tqdm import tqdm
from typing import Tuple

import data

class UniDataset(Dataset):
    def __init__(self, df: pd.DataFrame, columns: str=None, in_days=4, out_days=14, device='cpu') -> None : 
        if (columns is None) : columns = [col for col in df.columns if col.startswith('NL')][:25]
            
        self.variates = len(columns)

        self.x = torch.tensor(df[columns].values, dtype=torch.float32, device=device)
        # TODO : add more time features not only hour
        self.h = torch.tensor(df['Hour'].values, dtype=torch.float32, device=device)/24
        self.doy = torch.tensor(df['DayOfYear'].values, dtype=torch.float32, device=device)/365
        
        (self.in_samples, self.out_samples) = (24*in_days, 24*out_days)
        self.n = len(self.x) - 2*(self.in_samples + self.out_samples)

    def __len__(self) -> int : return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor] :
        data = self.x[i:i+self.in_samples]
        data_time = torch.column_stack((
            self.h[i:i+self.in_samples],
            self.doy[i:i+self.in_samples],
        ))
        data_mask = ~data.isnan()
        target = self.x[i+self.in_samples:i+self.in_samples+self.out_samples]
        target_time = torch.column_stack((
            self.h[i+self.in_samples:i+self.in_samples+self.out_samples],
            self.doy[i+self.in_samples:i+self.in_samples+self.out_samples],
        ))
        
        return (
            data, data_time, data_mask, target, target_time            
        )
    

def train(args, model, device, train_loader, optimizer, epoch) :
    model.train()

    loss_avg = 0
    for batch_idx, (data, data_time, data_mask, target, target_time) in tqdm(enumerate(train_loader), leave=False, total=args.batches_per_epoch) :
        optimizer.zero_grad()

        outputs = model(
            past_values=data,
            past_time_features=data_time,
            past_observed_mask=data_mask,
            future_values=target,
            future_time_features=target_time
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loss_avg += (loss.item()/args.batches_per_epoch)
        if (batch_idx >= args.batches_per_epoch) : break
    
    return loss_avg

def eval(args, model, val_loader, epoch):
    model.eval()
    with torch.no_grad():
        rmse = 0
        for batch_idx, (data, data_time, data_mask, target, target_time) in tqdm(enumerate(val_loader), leave=False, total=args.batches_per_val) :
            outputs = model.generate(
                past_values=data,
                past_time_features=data_time,
                past_observed_mask=data_mask,
                future_time_features=target_time
            )

            pred = outputs.sequences.mean(dim=1)
            rmse += (target - pred).square().mean().sqrt() / args.batches_per_val
            if (batch_idx >= args.batches_per_val) : break

    return rmse.item()


def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser(description='Univariate Autoformer')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batches-per-epoch', type=int, default=100)
    parser.add_argument('--batches-per-val', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--in-days', type=int, default=4)
    parser.add_argument('--out-days', type=int, default=14)
    
    parser.add_argument('--no-cuda', action='store_true', default=False)

    return parser.parse_args()

def main(args) :
    device = torch.device('cuda') if (not(args.no_cuda) and torch.cuda.is_available()) else torch.device('cpu')
    
    print(sys.version)
    print(platform.node(), platform.platform())
    print(device)

    train_df = data.load(years=['2017', '2018', '2019', '2020', '2021'])
    train_dataset = UniDataset(train_df, in_days=args.in_days, out_days=args.out_days, device=device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_df = data.load(years=['2022'])
    val_dataset = UniDataset(val_df, in_days=args.in_days, out_days=args.out_days, device=device)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    print(f'number of variates: {train_dataset.variates}')
    # TODO : setup the config
    config = AutoformerConfig(
        input_size=train_dataset.variates, 
        prediction_length=train_dataset.out_samples,
        context_length=89, # idk how to set this value, this is the only way it works
        num_time_features=2,
        d_model=16,
    )
    model = AutoformerForPrediction(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        loss_avg = train(args, model, device, train_loader, optimizer, epoch)
        print(f'[{epoch+1:02d}/{args.epochs:02d}] {loss_avg:.3f}')
    
    torch.save(model.state_dict(), 'multivariate_model.pth')
    model.config.to_json_file('multivariate_config.json')

    val_rmse = eval(args, model, val_loader, epoch)
    print(f'validation rmse: {val_rmse:.3f}')




if (__name__ == '__main__') : main(parse_args())