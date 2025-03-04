'''
Created on 28/03/2023

@author: zepedro
'''

from meteoraster import MeteoRaster
from .multi_window_mapper import Multi_Window_Mapper
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import warnings 

from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from pathlib import Path
from scipy.stats import norm
import matplotlib.lines as mlines
import seaborn as sns

class QuantileMapping(object):
    def __init__(self, *args, **kwargs):
        '''
        Dummy init that discards arguments
        '''
        pass
    
    def set(self, target, reference, *args, **kwargs):
        df = pd.DataFrame({'target': target, 'reference': reference}).dropna()
        
        t_x = df['target'].sort_values().values.ravel()
        t_y = np.linspace(0, 1, t_x.size)
        self._FModel = interp1d(t_x, t_y, fill_value=(t_y[0], t_y[-1]), kind='linear', bounds_error=False)
        
        t_y = df['reference'].sort_values().values.ravel()
        t_x = np.linspace(0, 1, t_y.size)
        self._FInvReference = interp1d(t_x, t_y, kind='linear', bounds_error=True)
        
        return self
    
    def run(self, data, *args, **kwargs):
        return self._FInvReference(self._FModel(data))
         
    def map(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    def fit(self, *args, **kwargs):
        return self.set(*args, **kwargs)

    def plot(self, n=1000, ax=None, *args, **kwargs):
        
        p_reference = np.linspace(0, 1, n)
        y_reference = self._FInvReference(p_reference)
        y_model = np.linspace(y_reference.min()*0.8, y_reference.max()/0.8, n)
        p_model = self._FModel(y_model)
        first_1_position = np.where(p_model==1)[0]
        if len(first_1_position)==0:
            first_1_position = len(p_model)-1
        else:
            first_1_position = first_1_position[0]
        y_model = y_model[0:first_1_position+1]
        p_model = p_model[0:first_1_position+1]
        
        if isinstance(ax, type(None)):
            plt.figure()
            ax = plt.gca()
            
        ax.plot(y_reference, p_reference, label='Reference', *args, **kwargs)
        ax.plot(y_model, p_model, label='Model', *args, **kwargs)
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.grid()
        plt.legend()
        
        return ax
   

class Linear(QuantileMapping):
    
    
    def set(self, target, reference, *args, **kwargs):
        
        self.b = np.std(reference)/np.std(target)
        self.a =  np.mean(reference) - np.mean(target)*self.b
        
        super().set(target, reference, *args, **kwargs)
        
        df = pd.DataFrame({'target': target, 'reference': reference}).dropna()
        t_y = df['target'].sort_values().values.ravel()
        t_x = np.linspace(0, 1, t_y.size)
        self._FInvModel = interp1d(t_x, t_y, kind='linear', bounds_error=True)
        
        t_x = self.run(df['target'].sort_values().values.ravel())
        t_y = np.linspace(0, 1, t_x.size)
        self._FModel_c = interp1d(t_x, t_y, fill_value=(t_y[0], t_y[-1]), kind='linear', bounds_error=False)
        
        return self
    
    def run(self, data, *args, **kwargs):
        return data * self.b + self.a

    def plot(self, mapped, n=1000, ax=None, *args, **kwargs):
        
        t_x = mapped.dropna().sort_values().values.ravel()
        t_y = np.linspace(0, 1, t_x.size)
        self._FMapped = interp1d(t_x, t_y, fill_value=(t_y[0], t_y[-1]), kind='linear', bounds_error=False)
        
        p_reference = np.linspace(0, 1, n)
        y_reference = self._FInvReference(p_reference)
        
        p_model = np.linspace(0, 1, n)
        y_model = self._FInvModel(p_model)
        
        y_model_c = np.linspace(y_reference.min()*0.8, y_reference.max()/0.8, n)
        p_model_c = self._FModel_c(y_model_c)
        
        y_mapped = np.linspace(mapped.min()*0.8, mapped.max()/0.8, n)
        p_mapped = self._FMapped(y_mapped)
        
        first_1_position = np.where(p_model==1)[0]
        if len(first_1_position)==0:
            first_1_position = len(p_model)-1
        else:
            first_1_position = first_1_position[0]
        y_model = y_model[0:first_1_position+1]
        p_model = p_model[0:first_1_position+1]
        p_model_c = p_model_c[0:first_1_position+1]
        y_model_c = y_model_c[0:first_1_position+1]
        y_mapped = y_mapped[0:first_1_position+1]
        p_mapped = p_mapped[0:first_1_position+1]

        
        if isinstance(ax, type(None)):
            plt.figure()
            ax = plt.gca()
            
        ax.plot(y_reference, p_reference, label='Reference', *args, **kwargs)
        ax.plot(y_model, p_model, label='Model', *args, **kwargs)
        ax.plot(y_model_c, p_model_c, label='Model corrected', *args, **kwargs)
        ax.plot(y_mapped, p_mapped, ':', label='Mapped', *args, **kwargs)
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.grid()
        plt.legend()
        
        return ax


 
class QuantileDeltaMapping(QuantileMapping):
    def __init__(self, trend_window=365*9, quantiles=np.r_[0, norm.cdf(np.linspace(-3,3,19)), 1], transformation='additive', multiplicative_threshold=0.01, modified=False):
        self.quantiles = quantiles
        self.trend_window = trend_window
        self.transformation = transformation
        self.multiplicative_threshold = multiplicative_threshold
        self.modified = modified
    
    def set(self, to_map, reference, *args, **kwargs):
        
        df = pd.DataFrame({'target': to_map, 'reference': reference})
        
        t_y = df['target'].sort_values().values.ravel()
        t_x = np.linspace(0, 1, t_y.size)
        self._FInvModel = interp1d(t_x, t_y, fill_value=(t_y[0], t_y[-1]), kind='linear', bounds_error=False)
        
        t_y = df['reference'].sort_values().values.ravel()
        t_x = np.linspace(0, 1, t_y.size)
        self._FInvReference = interp1d(t_x, t_y, kind='linear', bounds_error=True)
        
        tmp = df.mean()
        self.reference_mean = tmp['reference']
        self.target_mean = tmp['target']
        
        tmp = df.min()
        self.reference_min = tmp['reference']
        self.target_min = tmp['target']
        
        tmp = df.std()
        self.reference_std = tmp['reference']
        self.target_std = tmp['target']
        
        return self
    
    def run(self, data, *args, **kwargs):
        
        quantiles = self._quantiles(data)

        to_interpolate = data.copy()
        if isinstance(to_interpolate, pd.Series):
            to_interpolate = to_interpolate.to_frame()
        to_interpolate.columns = ['value']
        to_interpolate.loc[:, 'quantile'] = self.FTau(to_interpolate, quantiles)
        
        if self.transformation=='additive':
            delta = to_interpolate['value'] - self._FInvModel(to_interpolate['quantile'])
            if self.modified:
                delta_corrected = delta / self.target_std * self.reference_std
            else:
                delta_corrected = delta
            corrected = (self._FInvReference(to_interpolate['quantile']) + delta_corrected).values.ravel()
        elif self.transformation=='multiplicative':
            delta = to_interpolate['value'] / self._FInvModel(to_interpolate['quantile'])
            if self.modified:
                delta[to_interpolate['value'] / self.target_std<self.multiplicative_threshold] = 1
                corrected = ((self._FInvReference(to_interpolate['quantile'])- self.reference_min)  * delta + self.reference_min).values.ravel()
            else:
                corrected = (self._FInvReference(to_interpolate['quantile']) * delta).values.ravel()
        else:
            raise Exception('Transformation "%s" not implemented.' % self.transformation)
        
        return corrected
    
    def plot(self, n=1000, ax=None, *args, **kwargs):
        
        p_reference = np.linspace(0, 1, n)
        y_reference = self._FInvReference(p_reference)
        y_model = self._FInvModel(p_reference)
        
        if isinstance(ax, type(None)):
            plt.figure()
            ax = plt.gca()
            
        ax.plot(y_reference, p_reference, label='Reference', *args, **kwargs)
        ax.plot(y_model, p_reference, label='Model', *args, **kwargs)
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.grid()
        plt.legend()
        
        return ax

    def plotTrend(self, data, ax=None, *args, **kwargs):
        '''
        Cannot be used in window mode. Only full models.
        '''
        
        if isinstance(ax, type(None)):
            plt.figure()
            ax = plt.gca()
        
        quantiles = self._quantiles(data)
        quantiles.plot(color='k', ax=ax, *args, **kwargs)
        for v0 in self._FInvModel(self.quantiles):
            ax.axhline(y=v0, color='r', linestyle=':', *args, **kwargs)
        ax.get_legend().remove()
        black = mlines.Line2D([], [], color='k', label='Projection quantiles')
        red = mlines.Line2D([], [], color='r', linestyle=':', label='Historical quantiles')
        ax.legend(handles=[black, red])
        
        return ax
        
    def plotMatching(self, data, ndates=5, n=1000, *args, **kwargs):
        '''
        Cannot be used in window mode. Only full models.
        '''
        
        quantiles = self._quantiles(data)
        date_sample = pd.DatetimeIndex(pd.date_range(data.index[0], data.index[-1], ndates+2)[1:-1].date)

        tau = np.linspace(0, 1, n)
            
        reference = self._FInvReference(tau)
        model = self._FInvModel(tau)
        corrected = []
        projection = []
        for d1 in date_sample:
            
            closest_index = np.abs(quantiles.index - d1).argmin()
            _quantiles = quantiles.iloc[closest_index,:]

            t_y = _quantiles.values.ravel()
            t_x = _quantiles.index.values
            FInvTmp = interp1d(t_x, t_y, kind='linear', bounds_error=False)
            values = FInvTmp(tau)
            if self.transformation=='additive':
                delta = values - model
                if self.modified:
                    delta_corrected = delta / self.target_std * self.reference_std
                else:
                    delta_corrected = delta
                projected_ = reference + delta_corrected 
            elif self.transformation=='multiplicative':
                delta = values / model
                if self.modified:
                    delta[values/self.target_std<self.multiplicative_threshold] = 1
                    projected_ = (reference - self.reference_min) * delta + self.reference_min
                else:
                    projected_original = reference * delta
            else:
                raise Exception('Transformation "%s" not implemented.' % self.transformation)
            corrected.append(projected_)
            projection.append(values)
        
        corrected = pd.DataFrame(corrected, index=date_sample.strftime('%Y-%m-%d'), columns=tau).transpose()
        corrected.columns.name = 'Date'
        corrected.index.name = 'F(x)'
        
        projection = pd.DataFrame(projection, index=date_sample.strftime('%Y-%m-%d'), columns=tau).transpose()
        projection.columns.name = 'Date'
        projection.index.name = 'F(x)'
        
        data_ = pd.concat((corrected, projection),axis=1, keys=['Corrected', 'Projection'], names=['Experiment', 'Date'])
        
        stacked = data_.stack(['Experiment', 'Date']).to_frame(name='x').reset_index().dropna()
        
        g = sns.FacetGrid(stacked, col='Date', hue='Experiment', row=None)
        axs = g.map(plt.plot, 'x', 'F(x)')
        for ax0 in axs.axes[0]:
            ax0.plot(reference, tau, 'r:', label='Reference')
            ax0.plot(model, tau, 'k--', label='Historical model')
        plt.legend()
        
        return g
    
    def _quantiles(self, data):
        
        freq = pd.infer_freq(data.index[:3])
        if not freq:
            freq = 'MS'
            warnings.warn('Frequency of the series could not be inferred. Monthly assumed.')
            #===================================================================
            # raise Exception('Frequency could not be inferred.')
            #===================================================================
        index_full = pd.date_range(data.index[0], data.index[-1], freq=freq)
        data_ = data.reindex(index_full)
        valid_percentage = np.isfinite(data_).sum() / data_.shape[0]
        
        quantiles_ = []
        for q0 in self.quantiles:
            quantiles_.append(data_.rolling(window=int(self.trend_window), center=True, min_periods=int(self.trend_window * valid_percentage/1.1)).quantile(q0))
        quantiles = pd.concat(quantiles_, axis=1)
        quantiles = quantiles.interpolate(method='linear', limit_area='inside')
        quantiles = quantiles.bfill(axis=0)
        quantiles = quantiles.ffill(axis=0)
        quantiles.columns = self.quantiles
        
        #=======================================================================
        # quantiles = quantiles.loc[data.index, :]
        #=======================================================================
        
        return quantiles
    
    @staticmethod
    def _toDateTimeInterp(datetimes):
        '''
        a unit is equivalent to 100 years
        '''
        return datetimes.map(lambda x: x.value)/1000000000/86400/365.25/100
    
    @staticmethod
    def FTau(values, quantiles):
        y_cols = quantiles.columns
        values_ = values.values.ravel()
        quantiles_ = quantiles.loc[values.index, :].values
        quantiles_estimated = np.empty(values_.shape[0])
        for r0 in range(values_.shape[0]):
            quantiles_estimated[r0] = np.interp(values_[r0], quantiles_[r0, :], y_cols)
    
        tmp = values.copy()
        tmp.loc[:] = np.expand_dims(quantiles_estimated, axis=1)
        
        return tmp
    
class QuantileMapper(object):
    '''
    classdocs
    '''

    def __init__(self, projection_historical, reference, kernel=QuantileMapping, kw_kernel={}, group_mappers=[lambda x: 1],
                 normalize_dates=True, diagnostics_path=None,
                 trend_window=5, hydrological_year_month_start=1, yearly_window=1):
        '''
        '''
           
        self.projection_historical = projection_historical
        self.reference = reference
        self.normalize_dates = normalize_dates
        
        self.kernel = kernel        
        self.kw_kernel = kw_kernel
        self.group_mappers = group_mappers
        
        if diagnostics_path is None:
            self.diagnostics = False
            self.diagnostics_path = None
        else:
            self.diagnostics = True
            self.diagnostics_path = Path(diagnostics_path)
        
        self.yearly_window = yearly_window
        self.trend_window = trend_window
        self.hydrological_year_month_start = hydrological_year_month_start
          
    def map(self):
        '''
        '''
        
        projection_historical = self.projection_historical.copy()
        reference = self.reference.copy()

        if self.normalize_dates:
            # removes time and sets dates to midnight
            projection_historical.index = self._normalize_dates(projection_historical.index)
            reference.index = self._normalize_dates(reference.index)
        
        valid_idx = pd.to_datetime(pd.concat((self.projection_historical, self.reference), axis=1).dropna().index)
        projection_historical = projection_historical.loc[valid_idx]
        reference = reference.loc[valid_idx]
        
        valid_idx_group = self._group_index(valid_idx)
        projection_historical.index = valid_idx_group
        reference.index = valid_idx_group
        
        self.mapping = self.kernel(**self.kw_kernel).set(to_map=projection_historical,
                                                         reference=reference)
        
    def apply(self, to_map):
        '''
        '''
        
        to_map_ = to_map.copy()
        to_map_dates = to_map_.index
        
        if self.normalize_dates:
            to_map_.index = pd.to_datetime(self._normalize_dates(to_map_dates))
        
        idx_group = self._group_index(to_map_.index)
        to_map_.index = idx_group
        
        mapped = self.mapping.run(to_map_)
        mapped.index = to_map_dates
        
        if self.diagnostics:
            self.plot_diagnostics(mapped, to_map)
      
        return mapped
    
    def _group_index(self, index):
        groups = []
        
        #=======================================================================
        # if len(self.group_mappers)>0:
        #=======================================================================
        for g0 in self.group_mappers:
            groups.append(index.map(g0))
        #=======================================================================
        # else:
        #     groups.append([1]*len(index))
        #=======================================================================
        groups.append(index)
        
        return pd.MultiIndex.from_arrays(groups)
    
    @staticmethod
    def _normalize_dates(datetimes):
        if not isinstance(datetimes, pd.DatetimeIndex):
            datetimes = pd.DatetimeIndex(datetimes)
        
        return datetimes.normalize()
         
    def plot_diagnostics(self, mapped, to_map):
                        
        self.diagnostics_path.mkdir(parents=True, exist_ok=True)
                
        if True:    
            mapped_full = pd.concat((mapped, self.projection_historical), axis=1).bfill(axis=1).iloc[:,0]
            if self.normalize_dates:
                mapped_full.index = pd.to_datetime(self._normalize_dates(mapped_full.index))
            mapped_full.index = self._group_index(mapped_full.index)
            mapped_full.name = 'Mapped'
            
            to_map_full = pd.concat((to_map, self.projection_historical), axis=1).bfill(axis=1).iloc[:,0]
            if self.normalize_dates:
                to_map_full.index = pd.to_datetime(self._normalize_dates(to_map_full.index))
            to_map_full.index = self._group_index(to_map_full.index)
            to_map_full.name = 'To map'
            
            reference = self.reference.copy()
            if self.normalize_dates:
                reference.index = pd.to_datetime(self._normalize_dates(reference.index))
            reference.name = 'Reference'
        
        for i0 in mapped_full.index.get_level_values(0).unique():     
            if isinstance(self.mapping.models[i0-1], QuantileDeltaMapping):
                self.mapping.models[i0-1].plotTrend(mapped_full.loc[i0, :])
                plt.savefig(self.diagnostics_path / f'quantile_trend_{i0}.png')
                plt.close()     
                
                self.mapping.models[i0-1].plotMatching(mapped_full.loc[i0, :], ndates=5)
                plt.savefig(self.diagnostics_path / f'quantile_{i0}.png')
                plt.close()
                
            else:
                ax = self.mapping.models[i0-1].plot(mapped=mapped_full)
                plt.savefig(self.diagnostics_path / f'quantile_{i0}.png')
                plt.close()     
                        
        #### PLOT YEARLY
        _ = plt.figure(figsize=(20,15))
        mapped_full.index = mapped_full.index.get_level_values(-1)
        mapped_full = mapped_full.to_frame()
        
        to_map_full.index = to_map_full.index.get_level_values(-1)
        to_map_full = to_map_full.to_frame()
        
        data_ = pd.concat((reference, to_map_full, mapped_full), axis=1)
        
        rolled = data_.rolling(window=self.yearly_window, center=True, axis=0).mean()
        stacked = rolled.loc[:, ['Reference', 'Mapped']].dropna().stack().reset_index()
        stacked.columns = ['datetime', 'type', 'values']
        stacked.loc[:, 'month'] = stacked['datetime'].dt.month
        sns.lineplot(x='month', y='values', hue=None, style='type', data=stacked, errorbar=('pi', 50), color='k')
        #=======================================================================
        # plt.show(block=False)
        #=======================================================================
          
        stacked = rolled.loc[data_.index<='2099-12-31', ['Mapped']].stack().to_frame()
        stacked.columns=['values']
        stacked.loc[:, 'month'] = stacked.index.get_level_values(0).month
        stacked.reset_index(inplace=True)
        stacked.columns = ['datetime', 'type', 'values', 'month']
        stacked.loc[:,'decade'] = stacked.loc[:,'datetime'].map(lambda x: int(x.year/10)*10)
        stacked = stacked.loc[stacked.loc[:,'decade'].isin([1980, 2000, 2020, 2040, 2060, 2080])]
          
        sns.lineplot(x='month', y='values', hue='decade', style=None, data=stacked, errorbar=('pi', 50), palette='plasma')
        plt.savefig(self.diagnostics_path / f'yearly.png')
        plt.close()     

        #### PLOT EVOLUTION
        data__ = data_.copy()
        if self.hydrological_year_month_start !=1 :
            data__.index -= pd.DateOffset(months=self.hydrological_year_month_start-1)
            data__ = data__.loc[pd.Timestamp(data__.index[0].year+1, 1, 1):pd.Timestamp(data__.index[-1].year-1, 12, 31), :]
        rolled = data__.resample('YS').mean().rolling(window=self.trend_window, center=True, axis=0).mean().dropna(how='all')
        ax = rolled.plot()
        ax.set_xlabel('Hydrological year')
        plt.savefig(self.diagnostics_path / f'trend.png')
        plt.close()     

        #####
        rolled = data__.rolling(window=self.trend_window, center=True, axis=0).std()
        rolled.plot()
        plt.savefig(self.diagnostics_path / f'trend std.png')
        plt.close()                   
    
#===============================================================================
# if __name__=='__main__':
# 
#     location = 'Aguieira'
#     coords = (40.342236, -8.193848)
#     variable = 'temperature'
#     
#     CORDEXProcessedPath = Path('C:/CORDEX/processed')
#     product = 'Iberia_{variable:s}_%s_NCC-NorESM1-M_r1i1p1_GERICS-REMO2015_v1.mr'.format(variable=variable)
#     meteo_hist = MeteoRaster.load(CORDEXProcessedPath / (product % 'historical'))
#     meteo_Iberia01 = MeteoRaster.load('Iberia01_%s.mr' % variable)
# 
# 
#     kw_kernel = {
#         'kw_model': {'trend_window': 365*31,
#                      'transformation': 'additive',
#                      'modified': False,
#                      },
#         'weight_function':lambda x: x**2,
#         'windows': [[(np.arange(-1, 2) + i) % 12 + 1 for i in range(0, 12, 1)],],
#         'model': QuantileDeltaMapping,
#         }
#     group_mappers = [lambda x: x.month]
#     diagnostics = pd.read_csv('diagnostics.csv', sep=';', index_col=0)
#     
#     qm = QuantileMapper(to_map=meteo_hist, reference=meteo_Iberia01, kernel=Multi_Window_Mapper, kw_kernel=kw_kernel, group_mappers=group_mappers, diagnostics=diagnostics)
# 
#     qm.map()
#     
#     model = 'Iberia_temperature_rcp45_NCC-NorESM1-M_r1i1p1_GERICS-REMO2015_v1'
#     rcp45 = MeteoRaster.load(CORDEXProcessedPath / (model + '.mr'))
#     save_path = Path('C:/CORDEX/unbiased/diagnostics') / model
#     rcp45_qm = qm.apply(new_data_to_map=rcp45, save_path=save_path)
#     rcp45_qm.save(Path('C:/CORDEX/unbiased') / (model + '_QDM.mr') )
#     
#     rcp85 = MeteoRaster.load(CORDEXProcessedPath / 'Iberia_temperature_rcp85_NCC-NorESM1-M_r1i1p1_GERICS-REMO2015_v1.mr')
#     rcp85_qm = qm.apply(new_data_to_map=rcp85)
# 
# 
# 
# 
#     
#     ref = meteo_Iberia01.getDataFromLatLon(*coords).stack('Production dates')
#     ref.index = ref.index.droplevel(0)
#     ref.columns = ['Reference']
#      
#     hist = meteo_hist.getDataFromLatLon(*coords).stack('Production dates')
#     hist.index = pd.DatetimeIndex(hist.index.droplevel(0).date)
#     hist.columns = ['Historical']
#      
#     raw45 = rcp45.getDataFromLatLon(*coords).stack('Production dates')
#     raw45.index = pd.DatetimeIndex(raw45.index.droplevel(0).date)
#     raw45.columns = ['RCP45_raw']
#      
#     map45 = rcp45_qm.getDataFromLatLon(*coords).stack('Production dates')
#     map45.index = pd.DatetimeIndex(map45.index.droplevel(0).date)
#     map45.columns = ['RCP45']
#      
#     raw85 = rcp85.getDataFromLatLon(*coords).stack('Production dates')
#     raw85.index = pd.DatetimeIndex(raw85.index.droplevel(0).date)
#     raw85.columns = ['RCP85_raw']
#      
#     map85 = rcp85_qm.getDataFromLatLon(*coords).stack('Production dates')
#     map85.index = pd.DatetimeIndex(map85.index.droplevel(0).date)
#     map85.columns = ['RCP85']
#      
#     data = ref.join(hist, how='outer').join(pd.concat((raw45, map45, raw85, map85), axis=1), how='outer')
#     yearly = data.groupby(data.index.year, axis=0).mean()
#     rolling = yearly.rolling(window=11, center=True, axis=0, min_periods=9).mean()
#     ax = yearly.loc[:, ['Reference', 'Historical', 'RCP45', 'RCP85']].plot(style='.')
#     rolling.loc[:, ['Reference', 'Historical', 'RCP45', 'RCP85']].plot(ax=ax)
#     rolling.loc[:, ['RCP45_raw', 'RCP85_raw']].plot(ax=ax, style=':')
#     
#     pass
#     pass
#===============================================================================