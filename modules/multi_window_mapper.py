'''
Created on 24/04/2023

@author: zepedro
'''

import itertools
import numpy as np
import pandas as pd

class Multi_Window_Mapper(object):
    '''
    Multi-window averaged model application
    '''

    def __init__(self,
                 model,
                 windows=[[[1]]],
                 weight_function=lambda x: x,
                 kw_model = {},
                 groupby = None,
                 
    ):
        
        self.windows = windows
        self.base_model = model
        self.kw_model = kw_model
        
        self.model_characteristcs = [i for i in itertools.product(*self.windows)]
        self.models = [None for i in self.model_characteristcs]
        
        self.weight_function = weight_function
        self.model_weights = [[self._set_weights(v) for v in i] for i in self.model_characteristcs]
        
        self.groupby = groupby
    
    def run(self, *args, **kwargs):
        return self.fit(*args, **kwargs)
        
    def fit(self, reference, to_map=None):
        if isinstance(to_map, type(None)):
            to_map = reference.copy()
        
        joint = pd.concat((reference, to_map), axis=1).dropna()
        index = joint.index

        self._set_index_vs_models(index)
        
        _, _, indexes_vs_models = self._get_index_vs_models(index)
        
        for i0 in range(len(self.models)):
            ref_ = joint.loc[indexes_vs_models.loc[:, i0].values, :].iloc[:,0]
            if self.groupby:
                ref_ = ref_.groupby(self.groupby).mean()

            if ref_.size>0:
                tex_ = joint.loc[indexes_vs_models.loc[:, i0].values, :].iloc[:,1]
                if self.groupby:
                    tex_ = tex_.groupby(self.groupby).mean()
                # where the model is fitted
                self.models[i0] = self.base_model(**self.kw_model)
                self.models[i0].fit(tex_, ref_)

                #===============================================================
                # tmp0 = pd.concat((tex_, ref_), axis=1)
                # tmp0.index = tmp0.index.get_level_values(-1)
                # tmp1 = joint.copy()
                # tmp1.index = tmp1.index.get_level_values(-1)
                # ax = tmp1.plot(style='--')
                # tmp0.plot(ax=ax)
                #===============================================================
        
        return self
    
    def map(self, to_map):
        
        index = to_map.index
        indexes_vs_models, weights, _ = self._get_index_vs_models(index)
        
        to_map_ = to_map.copy()
        to_map_.index = to_map.index.get_level_values(-1)
        
        mapped = indexes_vs_models.copy()*np.nan
        for i0, m0 in enumerate(self.models):
            idx = indexes_vs_models.loc[:, i0].values
            data = to_map_.loc[idx]
            if m0 and data.size>0:
                mapped.loc[idx, i0] = m0.map(data)
        weights_ = self.weight_function(weights)
        mapped_ = (mapped * weights_).sum(axis=1) / (weights_ * np.isfinite(mapped)).sum(axis=1)
        
        if isinstance(to_map.index, pd.MultiIndex):
            mapped_.index = to_map.index
        
        return mapped_
    
    def _set_weights(self, array):
        
        left = np.arange(1, len(array) // 2 + 2)
        right = np.flip(left[:-1])
        
        return np.r_[left, right]
    
    def _set_index_vs_models(self, index):
        

        if isinstance(index.get_level_values(-1), pd.DatetimeIndex):
            index = index.to_frame()
            index.iloc[:,-1] = 0
            index = pd.MultiIndex.from_frame(index)
        
        unique_index = index.drop_duplicates().sort_values()
        
        belonging = [unique_index.get_level_values(i) for i in range(len(self.windows))]
        bool_ = []
        bool_for_map_ = []
        weights_ = []
        for m0, w0 in zip(self.model_characteristcs, self.model_weights):
            bool0_ = []
            bool_for_map0_ = []
            weights0_ = []
            for w1, ww1, b1 in zip(m0, w0, belonging):
                tmp = np.tile(np.expand_dims(w1, axis=0), (len(b1), 1)) == np.tile(np.expand_dims(b1, axis=1), (1, len(w1)))
                bool_for_map0_.append(tmp.any(axis=1))
                bool0_.append(tmp[:, int((tmp.shape[1]-1)/2)])
                
                w1w_ = np.tile(np.expand_dims(ww1, axis=0), (unique_index.size, 1))
                weights0_.append((w1w_*tmp).sum(axis=1))
            bool_.append(np.r_[bool0_].all(axis=0))
            bool_for_map_.append(np.r_[bool_for_map0_].all(axis=0))
            weights_.append(np.r_[weights0_].sum(axis=0))
        index_vs_models_fit = pd.DataFrame(bool_, columns=unique_index).transpose()
        index_vs_models_map = pd.DataFrame(bool_for_map_, columns=unique_index).transpose()
        weights = pd.DataFrame(weights_, columns=unique_index).transpose()
        
        self.index_vs_models_fit = index_vs_models_fit
        self.index_vs_models_map = index_vs_models_map
        self.index_vs_weights = weights

    def _get_index_vs_models(self, index):

        if isinstance(index.get_level_values(-1), pd.DatetimeIndex):
            index = index.to_frame()
            index.iloc[:,-1] = 0
            index = pd.MultiIndex.from_frame(index)

        return self.index_vs_models_map.loc[index, :], self.index_vs_weights.loc[index, :], self.index_vs_models_fit.loc[index, :]

    def set(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.map(*args, **kwargs)
    
if __name__=='__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    
    from quantileMapping import QuantileDeltaMapping
    from matplotlib import pyplot as plt
    
    windows = [[(np.arange(-1, 2) + i) % 12 + 1 for i in range(0, 12, 1)],
               ]
    
    weqm = Multi_Window_Mapper(QuantileDeltaMapping, windows=windows, kw_model={'trend_window': 15}, weight_function=lambda x:x**9)
    dates = pd.date_range('2000-01-01', '2021-06-01', freq='5D')
    s1 = pd.Series(np.random.rand(dates.size)*5, index=dates)
    s1.loc[s1.index.month==10] *= 3
    s1_ = s1.copy()
    s1_.index = s1_.index.month
    s2 = pd.Series(np.random.rand(dates.size), index=dates)
    s2_ = s2.copy()
    s2_.index = s2_.index.month
    weqm.fit(reference=s1_, to_map=s2_)
    
    dates = pd.date_range('2000-01-01', '2099-06-01', freq='5D')
    s3 = pd.Series(np.random.rand(dates.size), index=dates)
    s3 += np.linspace(0, 4, num=s3.shape[0])
    s3_ = s3.copy()
    s3_.index = s3_.index.month
    s3__ = weqm.map(s3_, datetimes=s3.index)
    s3__.index = s3.index
    
    ax = s1.plot(label='Reference')
    s3.plot(label='To map', ax=ax)
    s3__.plot(label='Mapped', ax=ax)
    plt.legend()
    
    plt.show(block=True)
