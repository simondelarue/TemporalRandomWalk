# -*- coding: utf-8 -*-

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy import stats

import torch

from stream_graph import StreamGraph
from data_loader import DataLoader
from temporal_random_walk import TemporalRandomWalk

def run(data, test_size, strategy):

    # ------ Load Data & preprocessing ------
    dl = DataLoader(data, force_undirected=True, split=True, test_size=test_size)

    # ------ Stream graph ------
    print('\nCreating stream graphs ...')
    sg_train = StreamGraph(dl, mask='train')
    sg_test = StreamGraph(dl, mask='test')

    print('\nTrain graph')
    print(f'# nodes : {sg_train.number_of_nodes()}')
    print(f'# edges : {sg_train.number_of_edges()}')

    print('\nTest graph')
    print(f'# nodes : {sg_test.number_of_nodes()}')
    print(f'# edges : {sg_test.number_of_edges()}')

    # ------ Temporal random walk ------
    strategy = strategy # 'linear' is temporally biased towards the future, i.e we want to maximize chances to select edges that occured recently

    # Sample starting edge
    start_edge = sg_test.sample_edge(strategy=strategy)
    print(f'\nStart edge: {start_edge}')
    
    # Temporal random walk
    trw = TemporalRandomWalk(sg_test, start_edge=start_edge[:2], t=start_edge[-1], strategy='linear', l=20)
    walk = trw.walk()
    print(f'\nTemporal random walk: {walk}')

if __name__=='__main__':

    parser = argparse.ArgumentParser('Preprocessing data')
    parser.add_argument('--data', type=str, help='Dataset name: \{SF2H, HighSchool, ia-contact, ia-contacts_hypertext2009, fb-forum, ia-enron-employees\}', default='SF2H')
    parser.add_argument('--test_size', type=float, help='Size of test dataset (in proportion of total number of edges)', default=0.1)
    parser.add_argument('--strategy', type=str, help='Sampling strategy: \{linear, exponential, uniform\}', default='linear')
    args = parser.parse_args()

    # ------ Parameters ------
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f'Device : {DEVICE}')
    
    # ------ Run model ------
    run(args.data, args.test_size, args.strategy)

