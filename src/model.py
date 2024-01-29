import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import models_utils

class SweetWater:

    def __init__(self, data, bulkrna, name, batch_size = 128, epochs = 1000, lr = 0.01, verbose = 0, earlystopping=True):
        
        self.verbose = verbose
        self.xtrain, self.ytrain, self.xtest, self.ytest = data

        """
        Bulkrna
        """

        #define train/test sets just to trigger earlystop if phase2 overfits
        self.bulk_train, self.bulk_test = train_test_split(bulkrna, test_size = 0.2, random_state=13)
        
        """
        Define parameters
        """

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr 
        self.earlystopping = earlystopping
        if self.earlystopping:
            self.p1es = models_utils.EarlyStopper(patience = 10)
            self.p2es = models_utils.EarlyStopper(patience = 10)
            self.p3es = models_utils.EarlyStopper(patience = 50)
        self.name = name
    
        self.setup()

    def setup(self):

        #model and metrics
        model_str = 'cuda' if torch.cuda.is_available() else 'gpu'
        self.aemodel = models_utils.SweetWaterAutoEncoder(num_features = self.xtrain.shape[1], num_classes = self.ytrain.shape[1]).to(model_str)
        self.mseloss = torch.nn.MSELoss()  
        self.optimizer = torch.optim.Adam(self.aemodel.parameters(), lr = self.lr)

        ## Phase 1: Pseudobulk Alignment
        self.phase1_ds = models_utils.singledataset(self.xtrain)
        self.phase1_dl = models_utils.DataLoader(self.phase1_ds, batch_size = self.batch_size, shuffle = True)
        self.phase1_ds_test = models_utils.singledataset(self.xtest)
        self.phase1_dl_test = models_utils.DataLoader(self.phase1_ds_test, batch_size = self.batch_size, shuffle = True)

        ## Phase 2: Bulk Alignment
        self.phase2_ds = models_utils.singledataset(self.bulk_train)
        self.phase2_dl = models_utils.DataLoader(self.phase2_ds, batch_size = 1, shuffle = True)

        ## Phase 3: Pseudobulk proportions deconvolution
        self.phase3_ds = models_utils.dataset(self.xtrain, self.ytrain)
        self.phase3_dl = models_utils.DataLoader(self.phase3_ds, batch_size = self.batch_size, shuffle = True)
        self.phase3_ds_test = models_utils.dataset(self.xtest, self.ytest)
        self.phase3_dl_test = models_utils.DataLoader(self.phase3_ds_test, batch_size = self.batch_size, shuffle = True)

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