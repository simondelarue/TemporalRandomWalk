# -*- coding: utf-8 -*-

import numpy as np

import random
from typing import Tuple

from stream_graph import StreamGraph
from sampling_distribution import sampling_factory


class TemporalRandomWalk():
    ''' A temporal random walk exists from node u to v if there 
        exists a stream of edges made of triplets (u, w, t1), (w, x, t2) ...
        (x, v, tn) with t1 <= t2 <= tn. 
        
        Parameters
        ----------
            sg : StreamGraph
                Stream graph object encoding temporal interactions between nodes.
            start_edge : (int, int)
                Tuple of nodes that starts the temporal random walk.
            t : int
                Time at which starting edge occured.
            strategy : {'linear', 'exponential', 'uniform'} str (default = 'linear') 
                Sampling strategy for node selection during the walk. If 'linear' or 'exponential', sampling is temporally biased
                towards closest time-related neighbors. 
            l : int (default = 5)
                Length of temporal random walk, i.e number of nodes in walk. '''


    def __init__(self, sg: StreamGraph, start_edge: Tuple[int, int], t: int, strategy: str = 'linear', l: int = 5):
        self.sg = sg
        self.walk_ = [*start_edge]
        self.t = t
        self.strategy = strategy
        self.l = l

    # TODO : prevent walk to go back to previous node ?
    def walk(self) -> list:
        ''' Perform temoral random walk and return list of nodes in walk. Next node in walk is selected
            according to strategy, e.g closest time-related node that forms an edge with source node. Note,
            if several nodes are elligible, one of them is selected randomly. '''

        j = self.walk_[-1]
        t = self.t

        # Iterates until length of walk is reached 
        for _ in range(self.l - 2):

            # Compute in-between periods of time between source node and temporal neighbors
            timesteps = np.array(list(self.sg.data.get(j).keys()))
            valid_times = timesteps[timesteps >= t]
            delta_times = valid_times - t

            # Compute probability distribution of in-between period of times, according to strategy.
            # Then, sample timestep in this distribution
            sd = sampling_factory(self.strategy)
            probs = sd.compute_distrib(delta_times, reverse=True) # Sampling is temporally biased towards closest time-related neighbors
            sampled_t = random.choices(valid_times, weights=probs)[0]

            # Randomly chose (uniform) node in events that occured at sampled time
            neighbs = self.sg.data.get(j).get(sampled_t)
            sampled_n = random.choices(neighbs)[0]

            self.walk_.append(sampled_n)

            # Update current nodes and timestamp
            j = sampled_n
            t = sampled_t

        return self.walk_