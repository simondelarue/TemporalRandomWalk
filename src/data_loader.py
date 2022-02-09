# -*- coding: utf-8 -*-

import os
import pandas as pd

from utils import switch_cols_df
from selection import train_test_split

class DataLoader():
    ''' Loads and preprocesses data according to the desired source.

        Parameters
        -----------
            dataset: str
                Name of the dataset to load.
            force_undirected : bool (default = True) 
                If True, for each (t, i, j), adds a triplet (t, j, i). 
            split : bool (default = True)
                If True, performs train-test split on data. 
            in_path: str (default = 'data')
                Local path where to find data. 
            test_size: float (default = 0.15)
                Proportion of test set. '''

    def __init__(self, dataset: str, 
                       force_undirected: bool = True,
                       split: bool = True,
                       in_path: str = 'data',
                       test_size: float = 0.15):

        self.IN_PATH = f'{os.getcwd()}/{in_path}'
        self.name = dataset
        self.dict_df = self.__preprocess(self.__load(), force_undirected, split, test_size)


    def __load(self) -> pd.DataFrame:
        ''' Load data from source file. 
            Output
            -------
                DataFrame '''

        if self.name == 'SF2H':
            return pd.read_csv(f'{self.IN_PATH}/tij_SFHH.dat_', header=None, names=['t', 'i', 'j'], delimiter=' ')
        elif self.name == 'HighSchool':
            return pd.read_csv(f'{self.IN_PATH}/High-School_data_2013.csv', header=None, names=['t', 'i', 'j', 'Ci', 'Cj'], delimiter=' ')
        elif self.name == 'AS':
            return pd.read_pickle(f'{self.IN_PATH}/as-733/as_100.pkl')
        elif self.name == 'ia-contact':
            return pd.read_csv(f'{self.IN_PATH}/ia-contact.edges', header=None, names=['ij', 'wt'], delimiter='\t')
        elif self.name == 'ia-contacts_hypertext2009':
            return pd.read_csv(f'{self.IN_PATH}/ia-contacts_hypertext2009.edges', header=None, names=['i', 'j', 't'], delimiter=',')
        elif self.name == 'ia-enron-employees':
            return pd.read_csv(f'{self.IN_PATH}/ia-enron-employees.edges', header=None, names=['i', 'j', 'c', 't'], delimiter=' ')
        elif self.name == 'fb-forum':
            return pd.read_csv(f'{self.IN_PATH}/fb-forum.edges', header=None, names=['i', 'j', 't'], delimiter=',')


    def __preprocess(self, data_df: pd.DataFrame, force_undirected: bool = True, split: bool = True, test_size: float = 0.15) -> dict:
        ''' Preprocessing consists of reindexing nodes labels from 0 to |V|. Data also may be split into train/test set.
        
            Parameters
            -----------
                data_df: DataFrame
                    Temporal data as a dataframe containing at least triplets (t, i, j).
                force_undirected : bool 
                    If True, for each (t, i, j), adds a triplet (t, j, i).
                split : bool (default = True)
                    If True, performs train-test split on data.
                test_size: float (default = 0.15)
                    Proportion of test set.
                    
            Output
            -------
                Dict of DataFrames with reindexed node labels. '''

        print('Preprocessing data ...')

        res = {}

        if self.name in ['SF2H', 'HighSchool', 'AS', 'ia-contacts_hypertext2009', 'ia-enron-employees', 'fb-forum']:

            # Reindex node labels 
            df_preproc = data_df.copy()
            unique_nodes = set(df_preproc['i'].values) | set(df_preproc['j'].values)

            mapping = {}
            for idx, node in enumerate(unique_nodes):
                mapping[node] = idx

            df_preproc['src'] = df_preproc['i'].apply(lambda x: mapping[x])
            df_preproc['dest'] = df_preproc['j'].apply(lambda x: mapping[x])
        
        elif self.name in ['ia-contact']:

            df_preproc = data_df.copy()
            df_preproc['src'] = df_preproc['ij'].apply(lambda x: int(x.split()[0]))
            df_preproc['dest'] = df_preproc['ij'].apply(lambda x: int(x.split()[1]))
            df_preproc['w'] = df_preproc['wt'].apply(lambda x: int(x.split()[0]))
            df_preproc['t'] = df_preproc['wt'].apply(lambda x: int(x.split()[1]))

        if force_undirected:

            sw_df = switch_cols_df(df_preproc, 'src', 'dest')
            df_preproc = pd.concat([df_preproc, sw_df]).sort_values(['t', 'src', 'dest'])

        if split:
            df_preproc_train, df_preproc_test = train_test_split(df_preproc, test_size)
            res['train'] = df_preproc_train
            res['test'] = df_preproc_test
        else:
            res['train'] = df_preproc
        
        return res


