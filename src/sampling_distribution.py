# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
import scipy.stats


class SamplingDistribution(ABC):
    ''' Parent class for sampling distributions. '''
    def __init__(self, strategy):
        self.strategy = strategy

    @abstractmethod
    def compute_distrib(self, X, reverse):
        pass


class UniformSamplingDistribution(SamplingDistribution):
    ''' Unbiased sampling strategy that corresponds to uniform distribution. '''

    def __init__(self, strategy):
        super().__init__(strategy)

    def compute_distrib(self, X, reverse: bool = None):
        ''' Returns vector of uniform probabilities for each element in X. '''

        l = len(X)
        return np.ones(l) / l


class LinearSamplingDistribution(SamplingDistribution):
    ''' Biased sampling strategy that corresponds to linear distribution. '''

    def __init__(self, strategy):
        super().__init__(strategy)

    def compute_distrib(self, X, reverse: bool = False):
        ''' Map each element in X to consecutive discrete events, then returns vector
            of probabilities for each elements. This corresponds to a temporally biased
            sampling, i.e considering the ascending or descending ordering of X, probabilities
            are temporally biased towards present or past respectively.
            
            Note : X must be in ascending order. 
            
            Parameters
            ----------
                reverse : bool (default = False)
                    If True, reverse ranking order, i.e smaller values get greater ranks. '''
        
        if reverse:
            X = -X
        r = scipy.stats.rankdata(X, 'dense')
        
        return r / np.sum(r)


def sampling_factory(strategy: str = 'linear'):
    ''' Returns SamplingDistribution object according to strategy. 
        
        Parameters
        ----------
            strategy : {'linear', 'exponential', 'uniform'} str (default = 'linear') 
                    Sampling strategy for node selection during the walk. If 'linear' or 'exponential', sampling is temporally biased
                    towards closest time-related neighbors. 
                        
        Output
        ------
            SamplingDistribution object. '''

    if strategy == 'linear':
        return LinearSamplingDistribution(strategy)
    elif strategy == 'uniform':
        return UniformSamplingDistribution(strategy)

# TODO : Add exponential sampling distribution class