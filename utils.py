import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True

def mixup_data(x, y, device):
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)  # choose an interpolation coefficient lambda at random
    perm = torch.randperm(x.shape[0]).to(
        device
    )  # generate random permutation of the batch
    return (
        lam * x + (1 - lam) * x[perm],
        y,
        y[perm],
        lam,
    )  # return mixed data, raw output, permuted output and lambda


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class my_RRC(torch.nn.Module):
    def __init__(self):

        super().__init__()


    def forward(self, specgram):
        Tau = specgram.shape[-1]
        Upsilon = specgram.shape[-2]

        #t = torch.randint(low=3*Tau//4,high=Tau,size=(1,))
        t = torch.randint(low=3*Tau//4,high=Tau,size=(1,))
        f = torch.randint(low=3*Upsilon//4,high=Upsilon,size=(1,)) 
        #f = Upsilon
        
        rc = torchvision.transforms.RandomCrop(size=(f.numpy()[0],t.numpy()[0]))
        #rc = torchvision.transforms.RandomCrop(size=(f,t.numpy()[0]))

        resize = torchvision.transforms.Resize(size=(Upsilon, Tau))
        
        return resize(rc(specgram))