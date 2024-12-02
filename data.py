import requests
import os
from itertools import product
from tqdm import tqdm


def download(url: str, out: str) -> None :
    response = requests.get(url)
    if (response.status_code == 200):
        with open(out, 'wb') as handle : 
            handle.write(response.content)
    else:
        raise Exception(f'failed to download {url} [{response.status_code}]')
    
def main():
    if not(os.path.exists('./data')) : os.mkdir('./data')

    indictors = ['NO2']
    years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023']

    url = lambda indicator, year : f'https://data.rivm.nl/data/luchtmeetnet/Vastgesteld-jaar/{year}/{year}_{indicator}.csv'

    for indicator in indictors :
        if not(os.path.exists(f'./data/{indicator}')) : os.mkdir(f'./data/{indicator}')

    print('Downloading data...')
    for (indicator, year) in tqdm(product(indictors, years), total=len(indictors)*len(years)) :
        if not(os.path.exists(f'./data/{indicator}/{year}.csv')) : 
            download(url(indicator, year), f'./data/{indicator}/{year}.csv')

if (__name__ == '__main__') : main()