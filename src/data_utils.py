import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import combinations
from tqdm import tqdm
from matplotlib import pyplot
from scipy import stats
from functools import reduce as red
from torch.utils.data import Dataset, DataLoader
from scipy.io import mmread
from collections import Counter
import torch

def generate_synthethic(scRNA, nsamples = 1000, ct=None):

    #get the dictionary of positions
    def build_dfdict(scRNA):

        cells = scRNA.index.tolist()
        ctdict = {x:i for i, x in enumerate(sorted(set(scRNA.index)))}
        dfdict = {}
        for i, ct in enumerate(cells):
            if ctdict[ct] in dfdict:
                dfdict[ctdict[ct]].append(i)
            else:
                dfdict[ctdict[ct]] = [i]

        return dfdict

    #generate the fractions using the dirichlet distribution and combining celltypes
    def generate_fractions_dirichlet(samples, nct):

        #generate random positions vector
        combl = []
        ncombl = sum(map(len,[list(combinations(range(nct),i)) for i in range(1,nct+1)]))
        l = m.ceil(samples/ncombl)

        for i in range(1, nct+1): ## group of combinations
            cmb = list(combinations(range(nct),i))
            for e in cmb:
                combelm = np.zeros((l, nct))
                combelm[:,e] = np.random.dirichlet(alpha = np.ones(len(e)), size = l)
                combl.append(combelm)


        mat = np.vstack(combl)
        assert np.sum(mat.sum(axis=1)) == mat.shape[0]

        return mat

    def gen_expr(x, props, dfd, ncells):

        #get the percentage of each celltype
        props_int = np.int32(props * ncells[:,None])

        #set to 0 if not sampled
        props[props_int == 0] = 0
        props = (props.T/props.sum(axis=1)).T

        #init empty list
        samples_l = []

        for i in tqdm(range(props.shape[0]), desc = 'simulating bulk'):

            sample_l = [x[np.random.choice(dfd[k], size=s)] for k,s in enumerate(props_int[i,:]) if s > 0]
            sample_v = np.vstack(sample_l).sum(axis=0)
            samples_l.append(sample_v)

        samples_v = np.vstack(samples_l)

        return samples_v, props
        
    #get the celltype
    if ct:
       celltypes = ct
    else:
        celltypes = sorted(set(scRNA.index))

    #build dfdict
    dfd = build_dfdict(scRNA)

    #transform to a faster datatype
    scRNA = np.ascontiguousarray(scRNA.values, dtype=np.float32)

    """
    Generate fractions
    """

    #generate proportions
    props = generate_fractions_dirichlet(samples = nsamples, nct = len(celltypes))

    """
    Generate expression for train/test distribution
    """

    #define ncells
    ncells = np.random.randint(100, max(101, scRNA.shape[0]//10), props.shape[0]) ## from 100 to scRNA.shape[0]//10 ncells

    #generate expression for training/test distribution
    xpseudo, ypseudo = gen_expr(x = scRNA, props = props, dfd = dfd, ncells = ncells)

    return xpseudo, ypseudo

def convert_to_float_tensors(*args):
    return [torch.tensor(x).float().cuda() if torch.cuda.is_available() else torch.tensor(x).float() for x in args]

def transform_and_normalize(psbulkrna, bulkrna, log=True):

    ##apply log-t and minmax
    if log:
        psbulkrna = np.log2(psbulkrna+1)
    #psbulkrna_l = psbulkrna - psbulkrna.mean(axis=0)
    psbulkrna_lt = MinMaxScaler(feature_range=(0,1)).fit_transform(psbulkrna.T).T

    ## same with bulk
    if log:
        bulkrna = np.log2(bulkrna+1)
    #bulkrna_l= bulkrna - bulkrna.mean(axis=0)
    bulkrna_lt = MinMaxScaler(feature_range=(0,1)).fit_transform(bulkrna.T).T

    return psbulkrna_lt, bulkrna_lt

def CCCscore(y_pred, y_true, mode='all'):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # pred: shape{n sample, m cell}
    if mode == 'all':
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)

    elif mode == 'avg':
        pass

    ccc_value = 0

    for i in range(y_pred.shape[1]):
      
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_value += ccc

    return ccc_value / y_pred.shape[1]