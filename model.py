import importlib
import torch
import models as mod
import plots as pNC
import numpy as np
import pandas as pd
import sys
import os
import torchmetrics
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io import mmread
import data_utils
from scipy.stats.stats import pearsonr
from collections import Counter, defaultdict
import xgboost as xgb
import math as m
from itertools import product
import pickle
from sklearn.metrics import mean_absolute_error
import data_utils
import torch
from torch.nn import Linear, ReLU, Sigmoid, Dropout, Softmax
import torch.nn as nn
import math
import importlib
import numpy as np
from scipy.stats.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import plots as pNC
import math as m

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        print(f"Stablishing Early Stopping with patience {patience}")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
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
        self.relu = ReLU()
        self.smm = Softmax(dim=1)
        self.sm = Sigmoid()

    def forward(self, x, mode):
        
        # Apply a final (linear) classifier.
        ## encoder
        ench1 = self.encl1(x)
        ench1 = self.relu(ench1)
        ench2 = self.encl2(ench1)

        if mode == 'phase3':

            ##proportions inference
            propl1 = self.propl1(ench2)
            propl1 = self.relu(propl1)
            propl2 = self.propl2(propl1)
            propl2 = self.smm(propl2)

            return propl2

        else:
            
            ##decoder
            dech1 = self.decl1(ench2)
            dech1 = self.relu(dech1)
            dech2 = self.decl2(dech1)

            return dech2
        
class singledataset(Dataset):
  def __init__(self,x):
    self.x = x
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx]
  def __len__(self):
    return self.length
  
class dataset(Dataset):
  def __init__(self,x,y):
    self.x = x
    self.y = y
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx], self.y[idx]
  def __len__(self):
    return self.length

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
            self.p1es = EarlyStopper(patience = 10)
            self.p2es = EarlyStopper(patience = 10)
            self.p3es = EarlyStopper(patience = 50)
        self.name = name
    
        self.setup()

    def setup(self):

        #model and metrics
        self.aemodel = SweetWaterAutoEncoder(num_features = self.xtrain.shape[1], num_classes = self.ytrain.shape[1]).to('cuda')
        self.mseloss = torch.nn.MSELoss()  
        self.optimizer = torch.optim.Adam(self.aemodel.parameters(), lr = self.lr)

        ## Phase 1: Pseudobulk Alignment
        self.phase1_ds = singledataset(self.xtrain)
        self.phase1_dl = DataLoader(self.phase1_ds, batch_size = self.batch_size, shuffle = True)
        self.phase1_ds_test = singledataset(self.xtest)
        self.phase1_dl_test = DataLoader(self.phase1_ds_test, batch_size = self.batch_size, shuffle = True)

        ## Phase 2: Bulk Alignment
        self.phase2_ds = singledataset(self.bulk_train)
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

            with tqdm(total = int(self.epochs), mininterval=10) as pbar:

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
                        pbar.set_description(f'P1: Train MSE is: {round(aepseudotrloss[-1],6)}, Test MSE is {round(aepseudoteloss[-1],6)}', refresh=False)
                        pbar.update(1)
            
        def run_phase2():

            # define batched and total ae bulk loss
            aebulktrloss, aebulkteloss =  [], []

            with tqdm(total = self.epochs, mininterval=10) as pbar:

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
                        pbar.set_description(f'P2: Train MSE is: {round(aebulktrloss[-1],6)}, Test MSE is: {round(aebulkteloss[-1],6)}', refresh=False)
                        pbar.update(1)
                
        def run_phase3():

            trainproploss, trainpropr2, testproploss, testpropr2 = [], [], [], []

            with tqdm(total = self.epochs, mininterval=10) as pbar:
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

                    ## early stopping condition
                    if self.earlystopping:
                        if self.p3es.early_stop(testproploss[-1]):
                            print("Early stopping condition achieved")
                            break

                    if self.verbose:
                        pbar.set_description(f'P3: Train MSE {round(trainproploss[-1],6)}, test MSE {round(testproploss[-1],6)}, Train R2 {round(trainpropr2[-1],4)}, Test R2 {round(testpropr2[-1],4)}', refresh=False)
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