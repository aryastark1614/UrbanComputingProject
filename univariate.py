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
from glob import glob
from tqdm import tqdm
from typing import Tuple

import data

class UniDataset(Dataset):
    def __init__(self, df: pd.DataFrame, column: str='NL01485', in_days=5, out_days=2) -> None : 
        self.x = torch.tensor(df[column].values, dtype=torch.float32)
        # self.t = torch.tensor(df['Time.Continuous'].values, dtype=torch.float32)
        # TODO : add more time features not only hour
        self.t = torch.tensor(df['Hour'].values, dtype=torch.float32)/24
        
        (self.in_samples, self.out_samples) = (24*in_days, 24*out_days)
        self.n = len(self.x) - 2*(self.in_samples + self.out_samples)

    def __len__(self) -> int : return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor] :
        data = self.x[i:i+self.in_samples]
        data_time = self.t[i:i+self.in_samples].view(-1,1)
        data_mask = ~data.isnan()
        target = self.x[i+self.in_samples:i+self.in_samples+self.out_samples]
        target_time = self.t[i+self.in_samples:i+self.in_samples+self.out_samples].view(-1,1)
        return (
            data, data_time, data_mask, target, target_time            
        )
    

def train(args, model, device, train_loader, optimizer, epoch) :
    model.train()

    for batch_idx, (data, data_time, data_mask, target, target_time) in enumerate(train_loader) :
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

        if (batch_idx >= args.batches_per_epoch) : break

def eval(args, model, val_loader, epoch):
    model.eval()
    with torch.no_grad():
        rmse = 0
        for batch_idx, (data, data_time, data_mask, target, target_time) in enumerate(val_loader) :
            outputs = model.generate(
                past_values=data,
                past_time_features=data_time,
                past_observed_mask=data_mask,
                future_time_features=target_time
            )

            pred = outputs.sequences.mean(dim=1)
            rmse += (target - pred).square().mean().sqrt() / args.batches_per_val
            if (batch_idx >= args.batches_per_val) : break

        print(f'[{epoch+1:02d}/{args.epochs:02d}] {rmse.item()}')


def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser(description='Univariate Autoformer')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--batches-per-epoch', type=int, default=100)
    parser.add_argument('--batches-per-val', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--in-days', type=int, default=4)
    parser.add_argument('--out-days', type=int, default=1)

    return parser.parse_args()

def main(args) :
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_df = data.load(years=['2017', '2018', '2019', '2020', '2021'])
    train_dataset = UniDataset(train_df, in_days=args.in_days, out_days=args.out_days)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_df = data.load(years=['2022'])
    val_dataset = UniDataset(val_df, in_days=args.in_days, out_days=args.out_days)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # TODO : setup the config
    config = AutoformerConfig(
        prediction_length=train_dataset.out_samples, 
        context_length=89, # idk how to set this value, this is the only way it works
        num_time_features=1,
        d_model=16,
    )
    model = AutoformerForPrediction(config)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        eval(args, model, val_loader, epoch)


    torch.save(model.state_dict(), 'univariate_model.pth')
    model.config.to_json_file('univariate_config.json')


if (__name__ == '__main__') : main(parse_args())