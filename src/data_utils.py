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
from collections import Counter, defaultdict
import torch
import PyComplexHeatmap as pych

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

    return xpseudo, ypseudo, celltypes

def convert_to_float_tensors(*args):
    return [torch.tensor(x).float() for x in args]

def transform_and_normalize(*args):
    return [MinMaxScaler(feature_range=(0,1)).fit_transform(np.log2(x+1).T).T for x in args]

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


def rename_duplicates(df):

    df = df.copy()
    index_counts = df.index.value_counts()
    renamed_index = []
    
    for index_label in df.index:
        if index_counts[index_label] > 1:
            index_count = index_counts[index_label]
            renamed_label = f'{index_label} ({index_count})'
            renamed_index.append(renamed_label)
            index_counts[index_label] -= 1
        else:
            renamed_index.append(index_label)
    
    df.index = renamed_index

    return df


def select_only_1ct_samples(ytrain):

    ## get only samples that match with single-celltype proportions
    ## to ease the interpretation module

    samples_dict = defaultdict(list)
    for i in range(ytrain.shape[0]):
        for j in range(ytrain.shape[1]):
            if ytrain[i,j] == 1:
                samples_dict[j].append(i)

    return samples_dict

def calculate_scores_batch(explainer, input, reference, output, extra = None):

    ##
    scores = explainer.attribute(
            inputs = input, 
            baselines = reference,
            target = output,
            #additional_forward_args = extra
            )
    
    return scores

def plot_complexheatmap(deeplift_scores_df_kgenes):

    ## normalize
    df = deeplift_scores_df_kgenes / deeplift_scores_df_kgenes.max()
    
    markergenes = pd.DataFrame(df.index)
    markergenes['mk_ct'] = sum(list(map(lambda x: [x]*(len(df.index)//len(df.columns)), df.columns)), [])
    markergenes.index = df.index

    row_ha = pych.HeatmapAnnotation(ct=pych.anno_simple(markergenes.mk_ct, cmap='tab10', add_text=True, legend=False, height = 6), axis=0, verbose=0)
    
    plt.figure(figsize=(10, 10))
    cm = pych.ClusterMapPlotter(data=df,
                                col_cluster=False, row_cluster=False, cmap="viridis",
                                left_annotation = row_ha,
                                row_names_side = 'right',
                                #row_split = markergenes.mk_ct, row_split_gap = 1,
                                #col_split = 1, col_split_gap = 10,
                                col_names_side = 'top',
                                show_rownames=True, show_colnames=True, row_dendrogram=False,
                                label = 'DeepLift Score',
                                xticklabels_kws={'labelrotation':45,'labelsize': 10},
                                yticklabels_kws={'labelsize':8},
                                rasterized=False
                                )
    
    plt.show()
