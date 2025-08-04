'''
Created on 31/07/2025

@author: zepedro
'''

import pickle
import numpy as np
from modules.multi_window_mapper import Multi_Window_Mapper
from modules.quantile_mapping import QuantileMapper, QuantileMapping, QuantileDeltaMapping
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
plt.ion()
import warnings
warnings.filterwarnings('ignore')


#===============================================================================
# historical_data_file = Path(r'data/historical_data.csv')
# projections_file = Path(r'data/projections.csv')
# 
# historical = pd.read_csv(historical_data_file, index_col=0, header = [0, 1])
# projections = pd.read_csv(projections_file, index_col=0, header = [0, 1, 2, 3], skiprows=[4])
# tas_historical = historical.loc[:, (slice(None), 'T [C]')]
# 
# # Pour convertir la lettre à la fin du code en minuscule
# tas_historical.columns = [c[0][0] + c[0][1:].lower() for c in tas_historical.columns]
# 
# cas = 'rcp85'
# modele = 'CCCma-CanESM2_r1i1p1_SMHI-RCA4_v1'
# bv = 'Q11b'
# 
# tas_historical_bv = tas_historical.loc[:, bv]
# tas_projection_historical_bv = projections.loc[:, ('tas', 'historical', bv, modele)]
# tas_projection_bv = projections.loc[:, ('tas', cas, bv, modele)]
# 
# data = pd.concat({'Historical': tas_historical_bv, 'Projections hist.': tas_projection_historical_bv, 'Projections': tas_projection_bv}, axis=1).sort_index()
#===============================================================================

#===============================================================================
# experiment = 'dummy_nnoise_ntrend_namplification_nseasonality_nshift'
# data = pd.DataFrame(np.NaN, index=pd.date_range('1950-01-01', '2099-12-01', freq='1MS'), columns=['Historical', 'Projections hist.', 'Projections'])
# data.loc[:'2019-12-01', 'Historical'] = 10 + 5 * np.cos(np.arange(data.loc[:'2019-12-01', :].shape[0]) * 2 * np.pi / 12 + np.pi)
# data.loc[:, 'Projections hist.'] = 15 + 7 * np.cos(np.arange(data.shape[0]) * 2 * np.pi / 12 + np.pi)
# data.loc[:, 'Projections'] = data.loc[:, 'Projections hist.']
#===============================================================================

experiment = 'dummy_noise_ntrend_namplification_nseasonality_nshift'
data = pd.DataFrame(np.NaN, index=pd.date_range('1950-01-01', '2099-12-01', freq='1MS'), columns=['Historical', 'Projections hist.', 'Projections'])
data.loc[:'2019-12-01', 'Historical'] = 10 + 5 * np.cos(np.arange(data.loc[:'2019-12-01', :].shape[0]) * 2 * np.pi / 12 + np.pi)
data.loc[:, 'Projections hist.'] = 15 + 7 * np.cos(np.arange(data.shape[0]) * 2 * np.pi / 12 + np.pi)
data += np.random.randn(*data.shape)*2
data.loc[:, 'Projections'] = data.loc[:, 'Projections hist.']

fig, ax = plt.subplots(figsize=(8, 5))
data.iloc[:, 1:].plot(ax=ax)
data.iloc[:, [0]].plot(ax=ax, linewidth=2, color='k')
ax.legend(loc='upper left', frameon=False)
plt.show(block=False)
plt.tight_layout()



data_reference = data.loc[:, 'Historical']
data_projections_historical = data.loc[:, 'Projections hist.']
data_projections = data.loc[:, 'Projections']

diagnostics_path = Path(experiment)
kw_kernel = {'model': QuantileDeltaMapping, # Le type de Quantile Mapping à utiliser
             'kw_model': {'trend_window': 15*12, # la fenêtre pour le calcul du "delta"
                          'transformation': 'additive',
                          'modified': False,
                         },
            'windows': [[(np.arange(-1, 2) + i) % 12 + 1 for i in range(0, 12, 1)]],
            'weight_function': lambda x: x**2,
            } 

qm = QuantileMapper(projection_historical=data_projections_historical, reference=data_reference,
                    kernel=Multi_Window_Mapper, kw_kernel=kw_kernel,
                    trend_window=5, # Le nombre d'années condidérées pour la moyenne glissante
                    hydrological_year_month_start=9,
                    diagnostics_path=diagnostics_path)
qm.map()



corrected = qm.apply(data_projections_historical).dropna()
corrected = corrected.to_frame()
corrected.columns = ['Corrected']
corrected = pd.concat([data, corrected]).loc[:, ['Projections hist.', 'Projections', 'Corrected', 'Historical']]

corrected.to_excel(diagnostics_path / 'corrected.xlsx')
corrected.to_csv(diagnostics_path / 'corrected.csv')

fig, ax = plt.subplots(figsize=(11, 5))
plt.tight_layout()
corrected.plot(ax=ax)
plt.savefig(diagnostics_path / 'result.png', dpi=300)
with open(diagnostics_path / 'result.p', 'wb') as f:
    pickle.dump(plt.gcf(), f)
    
print('Done!')