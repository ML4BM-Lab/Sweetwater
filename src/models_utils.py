import numpy as np
import torch
from torch.nn import Linear, ReLU, Sigmoid, Dropout, Softmax
from torch.utils.data import Dataset, DataLoader

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