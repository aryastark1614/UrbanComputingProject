import pandas as pd

import json
import requests
import os
from itertools import product
from tqdm import tqdm

from typing import List, Set


def download_file(url: str, path: str) -> None :
    # downloading the file for the given url and saving it on out

    response = requests.get(url)
    if (response.status_code == 200):
        with open(path, 'wb') as handle : 
            handle.write(response.content)
    else:
        raise Exception(f'failed to download {url} [{response.status_code}]')
    

def _load_indicator(year: str='2017', indicator: str='NOx', stations: Set[str]=None) -> pd.DataFrame :
    if (indicator == 'NO2') : 
        df = _load_NO2(year)    
    else:
        df = pd.read_csv(f'./data/{indicator}/{year}.csv', sep=';', skiprows=5, encoding='latin-1')

        df.rename(columns={'begindatumtijd': 'time', 'meetlocatie_id': 'id', 'waarde': 'value'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.drop(['meetreeks_id', 'bron_id', 'accreditatienummer', 'component', 'matrix', 'meetopstelling_id', 'meetduur', 'eenheid', 'einddatumtijd', 'opm_code'], axis=1, inplace=True)

        df = df.pivot(index='time', columns='id', values='value')

    df.drop([col for col in df.columns if col not in stations], axis=1, inplace=True)
    df.rename(lambda col : f'{col.strip()}-{indicator}', axis=1, inplace=True)

    return df

def _load_NO2(year: str='2017') -> pd.DataFrame :
    df = pd.read_csv(f'./data/NO2/{year}.csv', sep=';', skiprows=9, encoding='latin-1')

    # removing spacing errors in the column names
    df.columns = pd.Index([col.strip() for col in df.columns])

    df.rename(columns={'Begindatumtijd': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)

    df.drop([col for col in df.columns if not(col.startswith('NL'))], axis=1, inplace=True)

    return df

def prep(
    years: List[str]=['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
    indicators: List[str]=['NO2']
) -> pd.DataFrame:
    # prepping the downloaded csv's
    with open('./data/stations.json', 'r') as handle : stations = json.load(handle)

    df = pd.concat([
        pd.concat([_load_indicator(year, indicator, stations) for year in tqdm(years, desc=indicator)])
        for indicator in indicators
    ], axis=1)

    df.fillna(0, inplace=True)
   
    df['Hour'] = df.index.hour
    df['DayOfYear'] = df.index.dayofyear
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.month
    df['Time.Continuous'] = df.index.astype('int64') / 1e10 / 60

    return df
    
def load(
    years: List[str]=['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
    indicators: List[str]=['NO2']
) -> pd.DataFrame:

    if not(os.path.exists('./data')) : os.mkdir('./data')

    url = lambda indicator, year : f'https://data.rivm.nl/data/luchtmeetnet/Vastgesteld-jaar/{year}/{year}_{indicator}.csv'

    for indicator in indicators :
        if not(os.path.exists(f'./data/{indicator}')) : os.mkdir(f'./data/{indicator}')

    print('Downloading data...')
    for (indicator, year) in tqdm(product(indicators, years), total=len(indicators)*len(years)) :
        if not(os.path.exists(f'./data/{indicator}/{year}.csv')) : 
            download_file(url(indicator, year), f'./data/{indicator}/{year}.csv')

    return prep(years, indicators)
    

if (__name__ == '__main__') : load()