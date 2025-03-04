'''
Created on 03/03/2025

@author: Jose Pedro Matos

Tests quantile mapping routines
'''

import numpy as np
import matplotlib
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from modules.multi_window_mapper import Multi_Window_Mapper
from modules.quantile_mapping import QuantileMapper, QuantileMapping, QuantileDeltaMapping

matplotlib.use('QtAgg')

historical_data_file = Path(r'../data/historical_data.csv')
projections_file = Path(r'../data/projections.csv')

historical = pd.read_csv(historical_data_file, index_col=0, header = [0, 1])
projections = pd.read_csv(projections_file, index_col=0, header = [0, 1, 2, 3], skiprows=[4])

historical_ = historical.loc[:, (slice(None), 'T [C]')]
historical_.columns = [c[0][0] + c[0][1:].lower() for c in historical_.columns]

projections_ = projections.loc[:, ('tas', 'historical')]

for zone in historical_.columns[:1]:
    try:
        data = pd.concat({'Historical': historical_.loc[:, zone],
                          'Projections': projections_.loc[:, zone]}, axis=1).dropna()
        
        #=======================================================================
        # fig, ax = plt.subplots(figsize=(11, 5))
        # data.iloc[:, 1:].plot(ax=ax)
        # data.iloc[:, [0]].plot(ax=ax, linewidth=2, color='k')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        # ax.set_title(zone)
        # plt.tight_layout()
        # plt.show(block=False)
        #=======================================================================
    except Exception as ex:
        raise(ex)

projection_historical = data.iloc[:, 1]
projection = projections.loc[:, ('tas', 'rcp85')].loc[:, zone].iloc[:, 1]
reference = data.iloc[:, 0]

#===============================================================================
# kw_kernel = {
#     'kw_model': {},
#     'weight_function':lambda x: x**2,
#     'windows': [[(np.arange(-1, 2) + i) % 12 + 1 for i in range(0, 12, 1)],],
#     'model': QuantileMapping,
#     }
# 
# group_mappers = [lambda x: x.month]
# 
# qm = QuantileMapper(projection_historical=projection_historical, reference=reference, kernel=Multi_Window_Mapper, kw_kernel=kw_kernel, group_mappers=group_mappers, diagnostics_path=Path(r'../temp/diagnostics'))
# qm.map()
# corrected = qm.apply(projection)
#===============================================================================

#===============================================================================
# kw_kernel = {
#     'kw_model': {'trend_window': 15,
#                  'transformation': 'additive',
#                  'modified': False,
#                  },
#     'weight_function':lambda x: x**2,
#     'windows': [[(np.arange(-1, 2) + i) % 12 + 1 for i in range(0, 12, 1)],],
#     'model': QuantileDeltaMapping,
#     }
# group_mappers = [lambda x: x.month]
# 
# qm = QuantileMapper(projection_historical=projection_historical, reference=reference, hydrological_year_month_start=9,
#                     kernel=Multi_Window_Mapper, kw_kernel=kw_kernel, group_mappers=group_mappers)#,
#                     #diagnostics_path=Path(r'../temp/diagnostics'))
# qm.map()
# corrected = qm.apply(projection)
#===============================================================================



kw_kernel = {'model': QuantileMapping, 
             'kw_model': {'trend_window': 30}
            } 
#===============================================================================
# group_mappers = [lambda x: 1]
#===============================================================================
 
qm = QuantileMapper(projection_historical=projection_historical, reference=reference, hydrological_year_month_start=9,
                    kernel=Multi_Window_Mapper, kw_kernel=kw_kernel,
                    diagnostics_path=Path(r'../temp/diagnostics_lin'))
qm.map()
corrected = qm.apply(projection_historical)

qm = QuantileMapper(projection_historical=corrected, reference=reference, hydrological_year_month_start=9,
                    kernel=Multi_Window_Mapper, kw_kernel=kw_kernel,
                    diagnostics_path=Path(r'../temp/diagnostics_lin'))
qm.map()
corrected = qm.apply(projection)

print('Done!')
