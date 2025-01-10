import numpy as np
# import matplotlib.pyplot as plt
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
import json
from glob import glob
from tqdm import tqdm
from itertools import permutations
from typing import Tuple, List

from scipy.spatial import Delaunay
from pyproj import Transformer

from data import load

class WeightedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stations: List[str]=None, in_days=4, out_days=14, device='cpu') -> None : 
        if (stations is None) : 
            columns = [col for col in df.columns if col.startswith('NL')]
        else: 
            columns = [col for col in df.columns if col.split('-')[0] in stations]
        self.variates = len(columns)//3
        
        # self.x1 = torch.tensor(df[columns].values, dtype=torch.float32, device=device)
        self.x1 = torch.tensor(df[[c for c in columns if c.split('-')[1] == 'NOx']].values, dtype=torch.float32, device=device)
        self.x2 = torch.tensor(df[[c for c in columns if c.split('-')[1] == 'NO']].values, dtype=torch.float32, device=device)
        self.x3 = torch.tensor(df[[c for c in columns if c.split('-')[1] == 'NO2']].values, dtype=torch.float32, device=device)
        if (self.variates == 1) : self.x = self.x.squeeze()
        
        # spatially weighting the columns
        with open('./data/coordinates.json', 'r') as handle : coords = json.load(handle)

        # these are the same as found in a loaded df, same order since sorted by alphabet
        code2idx = {code: j for (j, code) in enumerate(sorted(coords.keys()))}
        idx2code = {j: code for (code, j) in code2idx.items()}

        C = np.zeros((len(coords), 2), dtype=np.float32)
        for (code, (x, y)) in coords.items() : C[code2idx[code]] = (x, y)

        self.coords = torch.tensor(C, dtype=torch.float32, device=device)

        mesh = Delaunay(C)
        # converting from coordinates to meters
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992") 
        X = np.c_[transformer.transform(C[:,0], C[:,1])]

        # iterating over the edges and filling in the weights matrix
        W = np.zeros((len(X), len(X)), dtype=np.float32)
        for ijk in mesh.simplices : 
            for (i, j) in permutations(ijk, 2) :
                W[i,j] = 1/np.linalg.norm(X[i] - X[j])
        W = W / (W[W > 0].max() - W[W > 0].min())

        # filling in the weighted data
        self.x1_weighted = torch.zeros_like(self.x1, device=device)
        self.x2_weighted = torch.zeros_like(self.x2, device=device)
        self.x3_weighted = torch.zeros_like(self.x3, device=device)
        for i in range(self.x1.shape[1]):
            for j in np.argwhere(W[i]).flatten() : 
                self.x1_weighted[:,i] += W[i,j]*self.x1[:,j]
                self.x2_weighted[:,i] += W[i,j]*self.x2[:,j]
                self.x3_weighted[:,i] += W[i,j]*self.x3[:,j]
            self.x1_weighted[:,i] /= W[i].sum()
            self.x2_weighted[:,i] /= W[i].sum()
            self.x3_weighted[:,i] /= W[i].sum()

        self.h = torch.tensor(df['Hour'].values, dtype=torch.float32, device=device)/24
        self.doy = torch.tensor(df['DayOfYear'].values, dtype=torch.float32, device=device)/365
        self.year = (torch.tensor(df['Year'].values, dtype=torch.float32, device=device)-2017)/7
        
        (self.in_samples, self.out_samples) = (24*in_days, 24*out_days)
        self.n = len(self.x1) - 2*(self.in_samples + self.out_samples)

    def __len__(self) -> int : return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor] :
        data = torch.stack((
            self.x1[i:i+self.in_samples, i%self.variates],
            self.x2[i:i+self.in_samples, i%self.variates],
            self.x3[i:i+self.in_samples, i%self.variates],
            self.x1_weighted[i:i+self.in_samples, i%self.variates],
            self.x2_weighted[i:i+self.in_samples, i%self.variates],
            self.x3_weighted[i:i+self.in_samples, i%self.variates],
        )).T
        data_time = torch.column_stack((
            self.h[i:i+self.in_samples],
            self.doy[i:i+self.in_samples],
            # self.year[i:i+self.in_samples],
        ))
        data_mask = ~data.isnan()
        target = torch.stack((
            self.x1[i+self.in_samples:i+self.in_samples+self.out_samples, i%self.variates],
            self.x2[i+self.in_samples:i+self.in_samples+self.out_samples, i%self.variates],
            self.x3[i+self.in_samples:i+self.in_samples+self.out_samples, i%self.variates],
            self.x1_weighted[i+self.in_samples:i+self.in_samples+self.out_samples, i%self.variates],
            self.x2_weighted[i+self.in_samples:i+self.in_samples+self.out_samples, i%self.variates],
            self.x3_weighted[i+self.in_samples:i+self.in_samples+self.out_samples, i%self.variates],
        )).T
        target_time = torch.column_stack((
            self.h[i+self.in_samples:i+self.in_samples+self.out_samples],
            self.doy[i+self.in_samples:i+self.in_samples+self.out_samples],
            # self.year[i+self.in_samples:i+self.in_samples+self.out_samples],
        ))

        static_real = self.coords[i%self.variates]
        
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
            static_real_features=static_real,
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
                static_real_features=static_real,
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
    train_dataset = WeightedDataset(train_df, stations=[args.station] if args.station is not None else None, in_days=args.in_days, out_days=args.out_days, device=device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_df = load(years=['2021', '2022'], indicators=['NOx', 'NO', 'NO2'])
    val_dataset = WeightedDataset(val_df, stations=[args.station] if args.station is not None else None, in_days=args.in_days, out_days=args.out_days, device=device)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    print(f'number of variates: {train_dataset.variates}')
    config = AutoformerConfig(
        input_size=2*3, 
        prediction_length=train_dataset.out_samples,
        context_length=89,
        num_time_features=2,
        num_static_real_features=2,
        d_model=args.dim,
    )
    model = AutoformerForPrediction(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss_avg = train(args, model, device, train_loader, optimizer, epoch)
        loss_val = eval(args, model, val_loader, epoch)
        print(f'[{epoch+1:02d}/{args.epochs:02d}] {loss_avg:.3f} {loss_val:.3f}')
    
    torch.save(model.state_dict(), 'models/weighted_model.pth')
    model.config.to_json_file('models/weighted_config.json')


if (__name__ == '__main__') : 
    main(args := parse_args())