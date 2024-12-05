import pandas as pd

import requests
import os
from itertools import product
from tqdm import tqdm

from typing import List


def download_file(url: str, path: str) -> None :
    '''downloading the file for the given url and saving it on out'''

    response = requests.get(url)
    if (response.status_code == 200):
        with open(path, 'wb') as handle : 
            handle.write(response.content)
    else:
        raise Exception(f'failed to download {url} [{response.status_code}]')
    
def prep(
    years: List[str]=['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
    indicators: List[str]=['NO2']
) -> pd.DataFrame:
    '''prepping the downloaded csv's'''

    df = pd.concat([
        pd.read_csv(f'./data/{indicator}/{year}.csv', sep=';', skiprows=9, encoding='latin-1') 
        for (indicator, year) in product(indicators, years)
    ])
    # removing spacing errors in the column names
    df.columns = pd.Index([col.strip() for col in df.columns])

    df.rename(columns={'Begindatumtijd': 'Time'}, inplace=True)
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

    df.drop([col for col in df.columns if not(col.startswith('NL'))], axis=1, inplace=True)
    
    df['Hour'] = df.index.hour
    df['DayOfYear'] = df.index.dayofyear
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.month
    df['Time.Continuous'] = df.index.astype('int64') / 1e10 / 60

    # TODO : how are we going to handle the NaN's?
    #        maybe it's better to just removing any instances containing them while batching the data?
    df.ffill(inplace=True)
    df.bfill(inplace=True)

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