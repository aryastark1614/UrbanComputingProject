import pandas as pd

from transformers import AutoformerConfig, AutoformerForPrediction

import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import argparse
import os
import platform
import sys
from tqdm import tqdm
from typing import Tuple, List

from data import load

class MultiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stations: List[str]=None, in_days=4, out_days=14, m=None, s=None, device='cpu') -> None : 
        # selecting the columns
        if (stations is None) : 
            columns = [col for col in df.columns if col.startswith('NL')]
        else: 
            columns = [col for col in df.columns if col.split('-')[0] in stations]
        self.variates = len(columns)
        
        self.x = torch.tensor(df[columns].values, dtype=torch.float32, device=device)
        if (self.variates == 1) : self.x = self.x.squeeze()
        
        # standardization
        (self.m, self.s) = (self.x.mean(axis=0), self.x.std(axis=0)) if (m is None or s is None) else (m, s)
        self.x = (self.x - self.m) / self.s

        self.h = torch.tensor(df['Hour'].values, dtype=torch.float32, device=device)/24
        self.doy = torch.tensor(df['DayOfYear'].values, dtype=torch.float32, device=device)/365
        self.year = (torch.tensor(df['Year'].values, dtype=torch.float32, device=device)-2017)/7
        
        (self.in_samples, self.out_samples) = (24*in_days, 24*out_days)
        self.n = len(self.x) - 2*(self.in_samples + self.out_samples)

    def __len__(self) -> int : return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor] :
        data = self.x[i:i+self.in_samples]
        data_time = torch.column_stack((
            self.h[i:i+self.in_samples],
            self.doy[i:i+self.in_samples],
            # self.year[i:i+self.in_samples],
        ))
        data_mask = ~data.isnan()
        target = self.x[i+self.in_samples:i+self.in_samples+self.out_samples]
        target_time = torch.column_stack((
            self.h[i+self.in_samples:i+self.in_samples+self.out_samples],
            self.doy[i+self.in_samples:i+self.in_samples+self.out_samples],
            # self.year[i+self.in_samples:i+self.in_samples+self.out_samples],
        ))

        static_real = self.year[i]
        
        return (
            data, data_time, data_mask, static_real, target, target_time            
        )
    

def train(args, model, device, train_loader, optimizer, epoch) :
    model.train()

    loss_avg = 0
    for batch_idx, (data, data_time, data_mask, static_real, target, target_time) in tqdm(enumerate(train_loader), leave=False, total=args.batches_per_epoch) :
        optimizer.zero_grad()

        outputs = model(
            past_values=data,
            past_time_features=data_time,
            past_observed_mask=data_mask,
            # static_real_features=static_real.view(-1,1),
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
        loss = 0
        for batch_idx, (data, data_time, data_mask, static_real, target, target_time) in tqdm(enumerate(val_loader), leave=False, total=args.batches_per_val) :
            outputs = model(
                past_values=data,
                past_time_features=data_time,
                past_observed_mask=data_mask,
                # static_real_features=static_real.view(-1,1),
                future_values=target,
                future_time_features=target_time
            )
            loss += outputs.loss.item()/args.batches_per_val
            if (batch_idx >= args.batches_per_val) : break

    return loss


def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser(description='Univariate Autoformer')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batches-per-epoch', type=int, default=100)
    parser.add_argument('--batches-per-val', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--in-days', type=int, default=4)
    parser.add_argument('--out-days', type=int, default=14)
    parser.add_argument('--station', type=str, default=None)
    
    parser.add_argument('--no-cuda', action='store_true', default=False)

    return parser.parse_args()

def main(args) :
    device = torch.device('cuda') if (not(args.no_cuda) and torch.cuda.is_available()) else torch.device('cpu')
    
    print(sys.version)
    print(platform.node(), platform.platform())
    print(device)

    train_df = load(years=['2017', '2018', '2019', '2020'], indicators=['NOx', 'NO', 'NO2'])
    train_dataset = MultiDataset(train_df, stations=[args.station] if args.station is not None else None, in_days=args.in_days, out_days=args.out_days, device=device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_df = load(years=['2021', '2022'], indicators=['NOx', 'NO', 'NO2'])
    val_dataset = MultiDataset(val_df, stations=[args.station] if args.station is not None else None, in_days=args.in_days, out_days=args.out_days, m=train_dataset.m, s=train_dataset.s, device=device)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    print(f'number of variates: {train_dataset.variates}')
    config = AutoformerConfig(
        input_size=train_dataset.variates, 
        prediction_length=train_dataset.out_samples,
        context_length=89, 
        num_time_features=2,
        # num_static_real_features=1,
        d_model=args.dim,
    )
    model = AutoformerForPrediction(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss_avg = train(args, model, device, train_loader, optimizer, epoch)
        loss_val = eval(args, model, val_loader, epoch)
        print(f'[{epoch+1:02d}/{args.epochs:02d}] {loss_avg:.3f} {loss_val:.3f}')
    
    if not(os.path.exists('./models')) : os.mkdir('./models')
    torch.save(model.state_dict(), './models/regular_model.pth')
    model.config.to_json_file('./models/regular_config.json')


if (__name__ == '__main__') : 
    main(args := parse_args())