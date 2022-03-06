import torch
from torch import nn
from torch import optim
import numpy as np
import copy as cp

class PENCIL():
    def __init__(self, all_labels_tensor, n_samples, n_classes, n_epochs, lrs, alpha, beta, gamma, K=10, save_losses=False, use_KL=True):
        '''
        all_labels_tensor: torch tensor, 1-D tensor of labels indexed as in the training dataset object
        n_samples: int, length of training dataset
        n_epochs: list of positive ints, number of epochs of phases in form [n_epochs_i for i in range(3)]
        lrs: list of floats, learnings rates for phases in form [lr_i for i in range(3)]
        alpha: coefficient for lo loss
        beta: coefficient for le loss
        gamma: coefficient for label estimate update
        K: int, learning rate multiplier for label estimate updates
        save_losses: bool, whether to save losses into list of lists of form [[lc,lo,le] for e in *phase 2 epochs*]
        use_KL: bool, whether to use KL loss or crossentropy for phase 3
        '''
        
        self.save_losses = save_losses
        self.use_KL = use_KL
        self.n_epochs = n_epochs
        self.lrs = lrs
        self.n_classes = n_classes
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.K = K
        
        self.CELoss = nn.CrossEntropyLoss()
        self.KLLoss = nn.KLDivLoss(reduction='mean') #PENCIL official implementation uses mean, not batchmean
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self._init_y_tilde(all_labels_tensor)
        self.y_prev = None
        self.losses = []
        
    def _init_y_tilde(self, all_labels_tensor):
        '''
        all_labels_tensor: torch tensor, 1-D tensor of labels indexed as in the training dataset object
        '''
        labels_temp = torch.zeros(all_labels_tensor.size(0), self.n_classes).scatter_(1, all_labels_tensor.view(-1, 1).long(), self.K)
        self.y_tilde = labels_temp.numpy()
        
    def set_lr(self, optimizer, epoch):
        '''
        Call before inner training loop to update lr based on PENCIL phase
        '''
        lr = -1
        if epoch == 0: lr = self.lrs[0] # Phase 1
        elif epoch == self.n_epochs[0]: lr = self.lrs[1] # Phase 2
        elif epoch == self.n_epochs[0]+self.n_epochs[1]: lr = self.lrs[2] # Phase 3 
        elif epoch == self.n_epochs[0]+self.n_epochs[1]+self.n_epochs[2]//3: # Phase 3 first decay
            lr = self.lrs[2]/10
        elif epoch == self.n_epochs[0]+self.n_epochs[1]+2*self.n_epochs[2]//3: # Phase 3 second decay
            lr = self.lrs[2]/100
            
        if lr!=-1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    def get_loss(self, epoch, outputs, labels, indices):
        '''
        outputs: un-normalized logits 
        labels: cuda tensor of noisy labels
        indices: cpu tensor of indices for current batch
        '''
        # Calculate loss based on current phase
        if epoch < self.n_epochs[0]: #Phase 1
            lc = self.CELoss(outputs, labels)
        else:
            self.y_prev = cp.deepcopy(self.y_tilde[indices,:]) #Get unnormalized label estimates
            self.y_prev = torch.tensor(self.y_prev).float()
            self.y_prev = self.y_prev.cuda()
            self.y_prev.requires_grad = True
            # obtain label distributions (y_hat)
            y_h = self.softmax(self.y_prev)
            if epoch<self.n_epochs[0]+self.n_epochs[1] or self.use_KL: # During phase 1. 
                lc = self.KLLoss(self.logsoftmax(self.y_prev),self.softmax(outputs))
            else: # During phase 2 use CE if self.use_KL=False
                lc = self.CELoss(self.softmax(outputs),self.softmax(y_h))
#             lo = self.CELoss(y_h, labels) # lo is compatibility loss
            lo = - torch.mean(torch.mul(self.softmax(y_h), self.logsoftmax(y_h)))
            le = - torch.mean(torch.mul(self.softmax(outputs), self.logsoftmax(outputs))) # le is entropy loss
        # Compute total loss
        if epoch < self.n_epochs[0]:
            loss = lc
        elif epoch < self.n_epochs[0]+self.n_epochs[1]:
            loss = lc + self.alpha * lo + self.beta * le
            if self.save_losses: self.losses.append([lc.item(),lo.item(),le.item()])
        else:
            loss = lc
        return loss
    
    def update_y_tilde(self, epoch, indices):
        '''
        Call this after the backward pass over the loss
        ''' 
        # If in phase 2, update y estimate
        if epoch >= self.n_epochs[0] and epoch < self.n_epochs[0]+self.n_epochs[1]:
            # update y_tilde by back-propagation
            self.y_tilde[indices]+=-self.gamma*self.y_prev.grad.data.cpu().numpy()
