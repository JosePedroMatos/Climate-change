'''
Created on Mar 1, 2025

@author: jose pedro


Converts daily CORDEX files to monthly equivalents
'''

import seaborn as sns
from matplotlib import pyplot as plt
from meteoraster import MeteoRaster
from pathlib import Path

sns.set_context('paper')

mon_path = Path(r'E:\CORDEX_AFR_MON')
available_files = [f for f in mon_path.rglob('*.mr')]

morocco_path = Path(r'E:\CORDEX_MOROCCO_MON')
crop = dict(from_lat=20, to_lat=37, from_lon=-20, to_lon=0)

for f0 in available_files:
    relative_path = f0.relative_to(mon_path)
    cropped_file_path = morocco_path / relative_path.parent / relative_path.name.replace('_AFR-44_', '_MOROCCO_')
    
    if cropped_file_path.exists():
        continue #jump this file if it already exists
    
    africa = MeteoRaster.load(f0)
    morocco = africa.getCropped(**crop)
     
    # save file
    cropped_file_path.parent.mkdir(exist_ok=True, parents=True)
    morocco.save(cropped_file_path)
        
    # prepare regional plot
    plt_folder = cropped_file_path.parent / 'plots'
    plt_folder.mkdir(exist_ok=True, parents=True)
    try:
        units = morocco.units
        variable = cropped_file_path.parent.name
        if variable in ['precipitation']:
            cmap = 'viridis'
        elif variable in ['temperature']:
            cmap = 'magma'
        else:
            raise Exception('Please define a color map for this type of data')
        
        ax = morocco.create_plot(central_longitude=-8)
        morocco.plot_mean(ax=ax, coastline=True, borders=True, colorbar=True, 
                       colorbar_label=f'{variable} [{units}]', cmap=cmap, central_longitude=-8, central_latitude=32)
        plt.tight_layout()
        plt.gcf().savefig(plt_folder / cropped_file_path.name.replace('.mr', '.png'), dpi=300)
    except Exception as ex:
        print(str(ex))
    finally:
        plt.close('all')
    
    
valid_files = [f.name for f in morocco_path.rglob('pr*_rcp45_*.mr')]

valid_queries = []
for experiment in ['historical', 'rcp45', 'rcp85']:
    for f0 in valid_files:
        valid_queries.append('_'.join(f0.replace('_rcp45_', f'_{experiment}_').split('_')[1:-2]))

for f0 in morocco_path.rglob('*_*.*'):
    query = '_'.join(f0.name.split('_')[1:-2])
    if query in valid_queries:
        continue
    else:
        f0.unlink()

valid_files
    
print('Done!')
