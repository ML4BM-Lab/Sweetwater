from captum.attr import DeepLift, LayerDeepLift
import os
import data_utils
import numpy as np
import torch
from torch.nn import Linear, ReLU, Sigmoid, Dropout, Softmax
from captum.attr import visualization as viz
import torch.nn as nn
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scanpy as sc
import anndata
import pandas as pd
import scipy
import seaborn as sns
import copy
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import models as mod
import PyComplexHeatmap as pych
from collections import defaultdict
import importlib
importlib.reload(data_utils)

"""
VISUALIZATION
"""

def tsne_deeplift(scRNA, deeplift_scores_df):

    ## get x genes
    topgenes_ct = {ct:0 for ct in deeplift_scores_df.columns}
    for ct in deeplift_scores_df.columns:
        #get top 20 genes
        topkgenes = deeplift_scores_df['B cells'].abs().sort_values(ascending=False).iloc[0:20].index.tolist()
        topgenes_ct[ct] = topkgenes

    # Initialize the t-SNE model
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)

    # Fit the data to the t-SNE model
    X_embedded = tsne.fit_transform(X)

    # Create a scatter plot of the t-SNE visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker='o', s=50)
    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    return

def visualize_score(score_df, name = 'marker_genes', dataset = 'pbmc_GSE107990', mode = 'heatmap'):

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 10))  # Set the size of the heatmap
    sns.set(font_scale=1)  # Adjust the font scale for better readability

    # You can customize the colormap by changing "cmap"
    # You can also customize other parameters like annot, fmt, linewidths, etc.

    if mode == 'cluster':
       sns.clustermap(score_df, col_cluster=False)
    elif mode == 'heatmap':
        sns.heatmap(score_df, cmap="YlGnBu", linewidths=0.5)

    # Set axis labels
    plt.xlabel(f"Celltypes {dataset}")

    # Show the plot
    plt.savefig(os.path.join(f'interpretability/{dataset}_{score_df.shape[0]}genes/{name}_{mode}.png'))
    plt.show()
    clean_plots()

def visualize_correlations(corr_df, score_df, dataset = 'pbmc_GSE107990'):

    # Create a barplot using Seaborn
    sns.set(style="whitegrid")  # Set the style of the plot (optional)

    # Create the barplot
    sns.barplot(x='celltypes', y='corr', data=corr_df)
    
    # Add labels and a title
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.title("Barplot Example")

    # Show the plot
    plt.savefig(os.path.join(f'interpretability/{dataset}_{score_df.shape[0]}genes/corr_barplot.png'))
    plt.show()
    clean_plots()

def plot_complexheatmap(deeplift_scores_df_kgenes):

    df = deeplift_scores_df_kgenes / deeplift_scores_df_kgenes.max()
    
    markergenes = pd.DataFrame(df.index)
    markergenes['mk_ct'] = sum(list(map(lambda x: [x]*(len(df.index)//len(df.columns)), df.columns)), [])
    markergenes.index = df.index

    row_ha = pych.HeatmapAnnotation(ct=pych.anno_simple(markergenes.mk_ct, cmap='tab10', add_text=True, legend=False, height = 10), 
                                        axis=0, verbose=0)
    
    plt.figure(figsize=(30, 30))
    cm = pych.ClusterMapPlotter(data=df,
                                col_cluster=False, row_cluster=False, cmap="viridis",
                                left_annotation = row_ha,
                                row_names_side = 'right',
                                #row_split = markergenes.mk_ct, row_split_gap = 1,
                                #col_split = 1, col_split_gap = 10,
                                col_names_side = 'top',
                                show_rownames=True, show_colnames=True, row_dendrogram=False,
                                label = 'DeepLift Score',
                                xticklabels_kws={'labelrotation':45,'labelsize':20},
                                yticklabels_kws={'labelsize':15},
                                rasterized=False
                                )
    
    plt.savefig(os.path.join(f'interpretability/{name.lower()}/sweetwater/{explstr}/{typeofgenes.replace("+","_").replace(" ","")}/{filename}.pdf'))
    plt.show()
    clean_plots() 

def plot_simpleheatmap(deeplift_scores_df_kgenes): 
    ## plot heatmap
    plt.figure(figsize=(15,15))
    sns.heatmap(deeplift_scores_df_kgenes / deeplift_scores_df_kgenes.max(), cmap='viridis')
    plt.ylabel('Input neuron score', fontsize = 15)
    plt.xlabel('Output neuron score', fontsize = 15)
    plt.savefig(os.path.join(f'interpretability/{name.lower()}/sweetwater/{explstr}/{typeofgenes.replace("+","_").replace(" ","")}/{filename}.png'))
    plt.show()
    clean_plots()

"""
TOOLS
"""

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

"""
COMPUTE MARKER GENES
"""

def clean_plots():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def compute_marker_genes(RNAData, k = 10):

    common_genes = RNAData.common_genes
    scRNAc = copy.deepcopy(RNAData.scRNA)
    celltypes = scRNAc.index.tolist()
    scRNAc.index = list(range(len(scRNAc.index)))

    ## create anndata
    adata = anndata.AnnData(scRNAc)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obs['ct'] = celltypes

    ## apply differential expression to rank genes according to celltype groups
    sc.tl.rank_genes_groups(adata, 'ct', method='wilcoxon')

    ## get ngenes x celltypes dataframe, where each celltype has the genes names sorted by score
    genes_ct_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    #logfold = pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'])
    #pvals_adj = pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj'])
    #markergenes_score = pd.DataFrame(adata.uns['rank_genes_groups']['scores'])

    def build_sorted_matrix(genes_ct_df, scores_df):

        assert all(genes_ct_df.columns == scores_df.columns)

        ## create dict containing, for each cell type, a dictionary with gene_ct -> score_ct mapping
        scores_d = {ct: {} for ct in genes_ct_df.columns}

        for i,ct in enumerate(genes_ct_df.columns):
            for j in range(genes_ct_df.shape[0]):
                gene, score = genes_ct_df.iloc[j,i], scores_df.iloc[j,i]
                scores_d[ct][gene] = score

        ## create dataframe with the same gene order as the scRNA
        df = []

        for ct in genes_ct_df.columns:
            df_ct = []
            for gene in common_genes:
                df_ct.append(scores_d[ct][gene])

            df.append(df_ct)

        ## build marker genes df
        marker_genes_df = pd.DataFrame(df).T
        marker_genes_df.index = common_genes
        marker_genes_df.columns = genes_ct_df.columns

        return marker_genes_df
    
    ## build logfold and pvals
    #log_fold_df = build_sorted_matrix(genes_ct_df, logfold)
    #pvals_adj_df = build_sorted_matrix(genes_ct_df, pvals_adj)

    # get k genes for each celltype
    markergenes = []
    for i in tqdm(range(genes_ct_df.shape[1]), 'looking for unique marker genes for celltypes'):
        ct_k = j = 0
        while ct_k < k:
            mkgene = genes_ct_df.iloc[j,i]
            #if not (mkgene in markergenes):
            markergenes.append(mkgene)
            ct_k += 1
            j+=1

    return markergenes

"""
SWEETWATER INTERPRETATION
"""

def select_only_1ct_samples(ytrprop):

    ## get only samples that match with single-celltype proportions
    ## to easy the interpretation module

    samples_dict = defaultdict(list)
    for i in range(ytrprop.shape[0]):

        for j in range(ytrprop.shape[1]):
            if ytrprop[i,j] == 1:
                samples_dict[j].append(i)

    return samples_dict

def remove_ct(RNAData, kgenes, ctlist):

    celltypes = RNAData.celltypes
    k = len(kgenes)//len(celltypes)
    
    leftgenes = []
    for i,ct in enumerate(ctlist):
        if ct in celltypes:
            leftgenes += kgenes[k*i:(k*(i+1))]

    return leftgenes

def calculate_scores_batch(explainer, input, reference, output, extra = None):

    ##
    scores = explainer.attribute(

                inputs = input, 
                baselines = reference,
                target = output,
                #additional_forward_args = extra
            )
    
    return scores

def select_only_k_var_genes(RNAData, k=20, top=True):

    topkgenes_var = RNAData.scRNA.var().sort_values(ascending=not top).head(k)

    kgenes_triple = []
    genes = RNAData.scRNA.columns
    ## get the positions within the scRNA matrix
    for i in range(RNAData.scRNA.shape[1]):
        if genes[i] in topkgenes_var.index:

            # go through top k genes
            for j,g in enumerate(topkgenes_var.index):
                if g == genes[i]:
                    kgenes_triple.append((g, i, j))

    kgenes_triple = sorted(kgenes_triple, key= lambda x: x[2])
    sorted_genes = list(map(lambda x: x[0], kgenes_triple))
    associated_pos = list(map(lambda x: x[1], kgenes_triple))

    return sorted_genes

"""
SWEETWATER MODELS
"""

class SweetWaterAutoEncoder(torch.nn.Module):
    def __init__(self, num_features, num_classes):

        self.num_features = num_features
        self.num_classes = num_classes

        self.l1size = int(self.num_features//2)
        self.l2size = int(self.num_features//4)
        super().__init__()
        torch.manual_seed(418)

        #aelayers
        self.encl1 = Linear(self.num_features, self.l1size)
        self.encl2 = Linear(self.l1size, self.l2size)

        self.decl1 = Linear(self.l2size, self.l1size)
        self.decl2 = Linear(self.l1size, self.num_features)

        #prop layers
        self.propl1 = Linear(self.l2size, self.l2size)
        self.propl2 = Linear(self.l2size, self.num_classes)

        #activation functions
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()
        self.smm = Softmax(dim=1)
        #self.sm = Sigmoid()

    def forward(self, x, mode='phase3'):
        
        # Apply a final (linear) classifier.
        ## encoder
        ench1 = self.encl1(x)
        ench1 = self.relu1(ench1)
        ench2 = self.encl2(ench1)

        if mode == 'phase3':

            ##proportions inference
            propl1 = self.propl1(ench2)
            propl1 = self.relu2(propl1)
            propl2 = self.propl2(propl1)
            propl2 = self.smm(propl2)

            return propl2

        else:
            
            ##decoder
            dech1 = self.decl1(ench2)
            dech1 = self.relu3(dech1)
            dech2 = self.decl2(dech1)

            return dech2

class SweetWater:

    def __init__(self, data, bulkrna, name, batch_size = 128, epochs = 1000, lr = 0.01, verbose = 0, earlystopping=True):
        
        self.verbose = verbose
        self.xtrain, self.ytrain, self.xtest, self.ytest = data

        """
        Bulkrna
        """
        #define train/test sets just to trigger earlystop if phase2 overfits
        self.bulk_train, self.bulk_test = train_test_split(bulkrna, test_size = 0.2, random_state=13)
        self.bulk_train = self.bulk_train.float()
        self.bulk_test = self.bulk_test.float().cuda()
        
        """
        Define parameters
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr 
        self.earlystopping = earlystopping
        if self.earlystopping:
            self.p1es = mod.EarlyStopper(patience = 10)
            self.p2es = mod.EarlyStopper(patience = 10)
        self.name = name
    
        self.setup()

    def setup(self):

        #model and metrics
        self.aemodel = mod.SweetWaterAutoEncoder(num_features = self.xtrain.shape[1], num_classes = self.ytrain.shape[1]).to('cuda')
        self.mseloss = torch.nn.MSELoss()  
        self.optimizer = torch.optim.Adam(self.aemodel.parameters(), lr = self.lr)

        ## Phase 1: Pseudobulk Alignment
        self.phase1_ds = data_utils.singledataset(self.xtrain)
        self.phase1_dl = DataLoader(self.phase1_ds, batch_size = self.batch_size, shuffle = True)
        self.phase1_ds_test = data_utils.singledataset(self.xtest)
        self.phase1_dl_test = DataLoader(self.phase1_ds_test, batch_size = self.batch_size, shuffle = True)

        ## Phase 2: Bulk Alignment
        self.phase2_ds = data_utils.singledataset(self.bulk_train)
        self.phase2_dl = DataLoader(self.phase2_ds, batch_size = 1, shuffle = True)

        ## Phase 3: Pseudobulk proportions deconvolution
        self.phase3_ds = data_utils.dataset(self.xtrain, self.ytrain)
        self.phase3_dl = DataLoader(self.phase3_ds, batch_size = self.batch_size, shuffle = True)
        self.phase3_ds_test = data_utils.dataset(self.xtest, self.ytest)
        self.phase3_dl_test = DataLoader(self.phase3_ds_test, batch_size = self.batch_size, shuffle = True)

        #define r2 metric
        self.r2 = lambda true, pred : 1 - ((np.square((true - pred)).mean()) / (np.square(true - true.mean(axis=0))).mean())

    def train(self, x, ytrue=None, mode='phase1'):

        self.optimizer.zero_grad()  # Clear gradients.
        if mode != 'phase3': #phase1 or phase 2

            xhat = self.aemodel(x, mode)  # Perform a single forward pass.
            aeloss = self.mseloss(x, xhat) 
            loss = aeloss

        else:

            yhat = self.aemodel(x, mode)
            proploss = self.mseloss(ytrue, yhat)
            loss = proploss
      
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.

        if mode != 'phase3':
            return loss.detach().__float__()
        else:
            return loss.detach().__float__(), yhat 

    @torch.no_grad()
    def test(self, x, ytrue=None, mode='phase1'):

        if mode != 'phase3': #phase1 or phase 2
            xhat = self.aemodel(x, mode)  # Perform a single forward pass.
            aeloss = self.mseloss(x, xhat) 
            loss = aeloss
        else:
            yhat = self.aemodel(x, mode)
            proploss = self.mseloss(ytrue, yhat)
            loss = proploss

        if mode != 'phase3':
            return loss.detach().__float__()
        else:
            return loss.detach().__float__(), yhat 

    def run(self):

        def run_phase1():

            aepseudotrloss, aepseudoteloss = [], []

            with tqdm(total = int(self.epochs)) as pbar:

                for _ in range(int(self.epochs)):

                    btrloss, bteloss = [], []

                    for xpseudo in self.phase1_dl:
                        
                        batch_pseudo_alignment_loss = self.train(x = xpseudo.cuda(), mode='phase1')
                        btrloss.append(batch_pseudo_alignment_loss)
                        
                    for xpseudo_test in self.phase1_dl_test:
                        bteloss.append(self.test(x = xpseudo_test.cuda(), mode='phase1'))

                    test_pseudo_alignment_loss = np.mean(bteloss)
                    
                    aepseudotrloss.append(np.mean(btrloss))
                    aepseudoteloss.append(test_pseudo_alignment_loss)
                    
                    ## early stopping condition
                    if self.earlystopping:
                        if self.p1es.early_stop(aepseudoteloss[-1]):
                            print("Early stopping condition achieved")
                            break
                    
                    if self.verbose:
                        pbar.set_description(f'P1: Train MSE is: {round(aepseudotrloss[-1],6)}, Test MSE is {round(aepseudoteloss[-1],6)}')
                        pbar.update(1)
            
        def run_phase2():

            # define batched and total ae bulk loss
            aebulktrloss, aebulkteloss =  [], []

            with tqdm(total = self.epochs) as pbar:

                for _ in range(self.epochs):

                    btrloss = []

                    for xbulk in self.phase2_dl:
                        
                        batch_pseudo_alignment_loss = self.train(x = xbulk.cuda(), mode='phase2')
                        btrloss.append(batch_pseudo_alignment_loss)
                        
                    aebulktrloss.append(np.mean(btrloss))
                    aebulkteloss.append(self.test(self.bulk_test, mode='phase2'))
                    
                    ## early stopping condition
                    if self.earlystopping:
                        if self.p2es.early_stop(aebulkteloss[-1]):
                            print("Early stopping condition achieved")
                            break
                    
                    if self.verbose:
                        pbar.set_description(f'P2: Train MSE is: {round(aebulktrloss[-1],6)}, Test MSE is: {round(aebulkteloss[-1],6)}')
                        pbar.update(1)
                
        def run_phase3():

            trainproploss, trainpropr2, testproploss, testpropr2 = [], [], [], []

            with tqdm(total = self.epochs) as pbar:
                for _ in range(self.epochs):

                    batched_trproploss, batched_trpropr2 = [], []

                    ## train
                    for xpseudo, bypseudo in self.phase3_dl:
                        
                        bproploss, bypred = self.train(x = xpseudo.cuda(), ytrue = bypseudo.cuda(), mode='phase3')
                        batched_trproploss.append(bproploss)
                        
                        bpropr2 = self.r2(bypseudo.numpy(), bypred.cpu().detach().numpy())
                        batched_trpropr2.append(bpropr2)
                        
                    trainproploss.append(np.mean(batched_trproploss))
                    trainpropr2.append(np.mean(batched_trpropr2))

                    btestproploss, btestr2 = [], []
                    for xpseudo_test, bypseudo_test in self.phase3_dl_test:
                        ## test
                        testproploss_, testypred = self.test(xpseudo_test.cuda(), bypseudo_test.cuda(), mode='phase3')

                        btestproploss.append(testproploss_)
                        btestr2.append(self.r2(bypseudo_test.numpy(), testypred.cpu().detach().numpy()))

                    testproploss.append(np.mean(btestproploss))
                    testpropr2.append(np.mean(btestr2))

                    if self.verbose:
                        pbar.set_description(f'P3: Train MSE {round(trainproploss[-1],6)}, test MSE {round(testproploss[-1],6)}, Train R2 {round(trainpropr2[-1],4)}, Test R2 {round(testpropr2[-1],4)}')
                        pbar.update(1)
                
        ## freeze layers from proportions deconvolution
        for name, param in self.aemodel.named_parameters():
            if 'prop' in name:
                param.requires_grad = False

        #run pseudobulk alignment and reset early stopping
        run_phase1()

        #now run proportions alignment
        run_phase2()

        ### freeze layers from the decoder
        for name, param in self.aemodel.named_parameters():
            if 'decl' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        #run bulk alignment and reset early stopping
        run_phase3()

## define the data parameters
genes_cutoff = 3000
nsamples = 5000
ncells = -1
ngenes_celltype = 500
name = 'pbmc_GSE107990'
modelname = 'Sweetwater'

## load data
RNAData = data_utils.RNADataset(genes_cutoff = genes_cutoff, nsamples = nsamples, ncells=ncells, name=name, model = modelname)
xtrain, ytrprop, xtest, yteprop, xbulkrna, ybulkrna, xbenchmark, ybenchmark, celltypes = RNAData.load_and_process_data()

## define epochs and init sweetwater object
# epochs = round(30000/(xtrain.shape[0]/256))
# sw = SweetWater(data = (xtrain, ytrprop, xtest, yteprop), 
#                 bulkrna = xbulkrna,
#                 name = name, verbose = True, 
#                 lr = 0.00001, batch_size = 256, epochs = epochs)

# # ## train
# sw.run()

# ## get the model
#mlmodel = sw.aemodel

## save it 
modelpath = f'interpretability/{name.lower()}/sweetwater/weights_{xtrain.shape[1]}_genes.pt'
#torch.save(mlmodel.state_dict(), os.path.join(modelpath))

## load the model
model = SweetWaterAutoEncoder(xtrain.shape[1], num_classes = ytrprop.shape[1])
model.load_state_dict(torch.load(os.path.join(modelpath)))
model.eval()

"""
APPLYING DeepLift AND LayerDeepLift
"""

# lets define the inputs, outputs, baselines and additional_forward_args
typeofgenes = 'topkdeepliftscore'
topkvargenes = select_only_k_var_genes(RNAData, k = 25, top=True)
topkmarkergenes = compute_marker_genes(RNAData, k = 10)
bottomkvargenes = select_only_k_var_genes(RNAData, k = 25, top=False)

## use DeepLift model to verify that the first input gene is the one that is playing the most important role on the first output gene
explstr = 'lastlayer'
explainer_lastlayer = DeepLift(model)
#explainer_emblayer = LayerDeepLift(model, layer = model.encl2)

## get dict for dynamic references
singlect_dict = select_only_1ct_samples(ytrprop)

#define samples for static references
input = xtrain ## samples by input genes

"""
Lets define some references
"""
refstr = 'refzeros'
refzeros = torch.zeros(size=input.shape)
reftotalgene = input.mean(axis=1).repeat(input.shape[1],1).T ## total gene expression mean by sample
refeachgene = input.mean(axis=0).repeat(input.shape[0],1) ## each-gene expression mean by sample
refcomb = (input.mean(axis=1) * refeachgene.T).T ## combination of former and latter

"""
Compute scores
"""
output = 0 ## gene i
extra = None
deeplift_scores_l = []

for i in tqdm(range(ytrprop.shape[1])):

    if refstr == 'refdynamic':
        input = xtrain[singlect_dict[i],:]
        refdynamic = torch.zeros(size=xtrain[singlect_dict[i],:].shape)

    ## Calculate scores
    deeplift_score = calculate_scores_batch(eval(f'explainer_{explstr}'), input, eval(refstr), i, extra) ## output gene i
    ## sum across samples
    score_values = deeplift_score.cpu().detach().numpy().sum(axis=0)
    deeplift_scores_l.append(score_values)

# visualize score
deeplift_scores_df = pd.DataFrame(deeplift_scores_l).T
deeplift_scores_df.columns = RNAData.celltypes

if explstr == 'lastlayer':
    deeplift_scores_df.index = RNAData.common_genes

# drop unknowns
#ctlist = ['B cells','Monocytes','NK','T CD4','T CD8', 'mDCs', 'pDCs']
#kgenes = remove_ct(RNAData,kgenes, ctlist)

## filter by top k topkdeepliftscore
if typeofgenes == 'topkdeepliftscore':
    topkdeepliftscore = []
    for ct in deeplift_scores_df.columns:
        topkdeeplift_ct = deeplift_scores_df.loc[:,ct].sort_values(ascending=False).head(ngenes_celltype).index.tolist()
        topkdeepliftscore += topkdeeplift_ct

kgenes = eval(typeofgenes) #+ bottomkgenes
deeplift_scores_df_kgenes = deeplift_scores_df.loc[kgenes, :]
deeplift_scores_df_kgenes.to_csv(os.path.join('interpretability', name.lower(), 'sweetwater', 'lastlayer','topkdeepliftscore',f'score_genes_top{ngenes_celltype}.tsv'),sep='\t')

##define filename
filename = f'corrmat_{xtrain.shape[1]}_sumscores_{refstr}_top{ngenes_celltype}genes'

## plot complex heatmap
plot_complexheatmap(rename_duplicates(deeplift_scores_df_kgenes))
#plot_simpleheatmap(deeplift_scores_df_kgenes)

# ## build corr matrix
# scores_corr = pd.DataFrame(marker_genes_df.corrwith(deeplift_scores_df))
# scores_corr['celltypes'] = scores_corr.index
# scores_corr.columns = ['corr', 'celltypes']
# visualize_correlations(scores_corr, deeplift_scores_df)
