# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

from collections import defaultdict
from typing import Tuple
import random

from data_loader import DataLoader
from sampling_distribution import sampling_factory


class StreamGraph():
    ''' A stream graph is a dynamic graph that can be defined by :math: `G = (V, E^T, T)`, where :math:`T` provides information about
        the time at which an edge occured between two nodes. Stream graphs are represented as dictionaries of dictionaries of lists. 
        For each node u is associated a dictionary with timestamp as keys and list of nodes as values.

        S : {u : {t0 : [v, w], t1 : [v], ...}, ...}
        
        Parameters
        -----------
            data: DataLoader
                DataLoader object containing a `DataFrame` object with a list triplets :math:`(u, v, t)` for each edge
                between nodes :math:`u` and :math:`v` at time :math:`t`. 
            mask: str
                Either 'train' or 'test'. '''

    def __init__(self, dataloader: DataLoader, mask: str):
        self.data = self._fill_stream_graph(dataloader.dict_df[mask])
        self._number_of_edges = dataloader.dict_df[mask].shape[0]
        

    def _fill_stream_graph(self, df: pd.DataFrame) -> dict:
        ''' Fill stream graph, based on DataFrame containing triplets (u, v, t) corresponding to temporal links between
            nodes u and v occuring at time t. 
            
            Parameters
            ----------
                df : DataFrame
                    Contains triplets (u, v, t). 
                    
            Output
            ------
                Dictionary of dictionaries of list, such that for each node u is associated a dictionary of {t: [v]},
                with t a timestep and v all the nodes that interracted with u at time t. '''

        data = defaultdict(dict)
        for _, row in df.iterrows():
            src, dest, t = row['src'], row['dest'], row['t']

            neighbs = data.get(src, {}).get(t, []) + [dest]
            temporal_neighbs = {t: neighbs}
            data[src].update(temporal_neighbs)

        return data

    def number_of_nodes(self) -> int:
        ''' Returns the number of nodes :math:`|V|` in a stream graph :math:`S = (V, E^T, T)`. 
        
            Output
            ------
                Number of nodes: int '''

        return len(self.data.keys())

    def number_of_edges(self) -> int:
        ''' Returns the number of edges :math:`|E|` in a stream graph :math:`S = (V, E^T, T)`. 
        
            Output
            ------
                Number of edges: int '''
        
        return self._number_of_edges

    def edges(self, sort: str = 'time') -> list:
        ''' Return edges as a list of triplets (u, v, t) forming temporal events between nodes u and v at time t. 
        
            Parameters
            ----------
                sort: str (default = 'time')
                    Sort edges by ascending timesteps. 
                    
            Output
            ------
                List of time-stamped events as triplets (u, v, t). '''

        edges = []
        for node, t_neighbs in self.data.items():
            for t, neighbs in t_neighbs.items():
                for neighb in neighbs:
                    edges.append((node, neighb, t))

        if sort == 'time':
            edges = sorted(edges, key=lambda x: x[2])

        return edges

    def sample_edge(self, strategy: str = 'linear', reverse = None) -> Tuple[int, int, int]:
        ''' Sample temporal edge in stream graph according to strategy. 
        
            Parameters
            ---------
                strategy : {'linear', 'exponential', 'uniform'} str (default = 'linear') 
                    Sampling strategy for node selection during the walk. If 'linear' or 'exponential', sampling is temporally biased
                    towards closest time-related neighbors. 
                reverse : bool (default = False)
                    If True, reverse ranking order, i.e smaller values get greater ranks.
            
            Output
            ------
                Temporal edge as triplet (u, v, t). '''

        edges = self.edges(sort='time')
        timestamps = [e[2] for e in edges]
        
        # Sampling distribution
        sd = sampling_factory(strategy)
        probs = sd.compute_distrib(timestamps, reverse=reverse)

        # Sample edge over distribution
        sampled_t = random.choices(timestamps, weights=probs)[0]
        valid_e = [e for e in edges if e[2]==sampled_t]
        sampled_e = random.choices(valid_e)[0]

        return sampled_e


