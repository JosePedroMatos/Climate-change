'''
Created on Mar 2, 2025

@author: Jose Pedro Matos

Extracts data per basin
'''

import pandas as pd
from pathlib import Path
from meteoraster import MeteoRaster


kml = Path(r'../data/catchments.kml')

data_path = Path(r'E:\CORDEX_MOROCCO_MON')
files = [f for f in data_path.rglob('*.mr')]

data = {}
for i0, f0 in enumerate(files):
    data_ = MeteoRaster.load(f0)
    values, centroids = data_.getValuesFromKML(kml, nameField='Indice', getCoverageInfo=False)
    data[f0.name.replace('.mr', '')] = values
    print(i0)

joint = pd.concat(data, axis=1)
columns = joint.columns.to_frame(index=False)
projections = columns[0].str.split('_', expand=True)

columns = columns.loc[:, ['Zone']]
columns.loc[:, 'Variable'] = projections[0]
columns.loc[:, 'Experiment'] = projections[3]
columns.loc[:, 'Code'] = projections[2] + '_' + projections[4] + '_' + projections[5] + '_' + projections[6]
joint.columns = pd.MultiIndex.from_frame(columns.loc[:, ['Variable', 'Experiment', 'Zone', 'Code']])

joint.to_excel(Path(r'../data/projections.xlsx'))
joint.to_csv(Path(r'../data/projections.csv'))

print('Done!')