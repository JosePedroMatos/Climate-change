'''
Created on 02/03/2025

@author: Jose Pedro Matos

Extracts historical precipitation and temperature data from hydrological model files
'''

import pandas as pd
from pathlib import Path

model_path = Path(r'../data/hydrological models')

# MODEL - pluie - débit Q2 KHANOUSSA ROUMANI
models = {f.name.replace('MODEL - pluie - débit ', '').split(' ')[0]:f for f in model_path.glob('*xlsx') if f.name.startswith('MODEL - pluie - débit Q')}

data_ = []
for k0, f0 in models.items():
    data = pd.read_excel(f0, sheet_name='MODEL - pluie - débit', skiprows=5, index_col=0, header=None,
                         usecols=[0, 5, 14, 15, 16]).dropna()
    data.columns = pd.MultiIndex.from_product([[k0], ['P [mm]', 'Hsim [mm]', 'Hobs [mm]', 'T [C]']], names=['Zone', 'Variables'])
    data.index.name = None
    
    data_.append(data)
data = pd.concat(data_, axis=1)
    
data.to_excel(Path(r'../data/historical_data.xlsx'))
data.to_csv(Path(r'../data/historical_data.csv'))
    
print('Done!')