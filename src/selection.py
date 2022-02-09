# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def train_test_split(df: pd.DataFrame, test_size: float = 0.15):
    ''' Split stream graph into train and test sub stream graphs, according to time dimension. 
    
        Parameters
        ----------
            data : dict
                Dictionary of data representing stream graph :math:`S=(V, E^T, T)`
            test_size : float (default = 0.15)
                Represent proportion of temporal data to include in test set. In other words, test set contains 
                all edges that appeared in S, after t = |T| * (1 - test_size). 
                
        Output
        ------
            Tuple of dictionaries representing training and test stream graphs. '''

    print('Splitting data ...')

    all_t = df['t'].unique()
    test_t = np.quantile(all_t, 1 - test_size)

    df_train = df.loc[df['t'] <= test_t]
    df_test = df.loc[df['t'] > test_t] 

    return df_train, df_test