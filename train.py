import time
import copy

import torch
import torch.nn as nn
import torch.optim as optimizer
from sklearn.metrics import recall_score, confusion_matrix, f1_score
import numpy as np
def train_model(net, trn_loader, val_loader, optim, num_epoch=5,
        collect_cycle=30, device='cpu', verbose=True):
    # Initialize:
    # -------------------------------------
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_l1, best_l2 = None, np.iinfo(numpy.uint64).max, np.iinfo(numpy.uint64).max
    loss_fn = nn.MSELoss()

    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        ############## Training ##############
        net.train()
        for input_, label in trn_loader:
            num_itr += 1
            input_, label = input_.to(device), label.to(device)
            
            # zero the parameter gradients
            optim.zero_grad()
            net = net.to(device)

            # forward
            pred = net(input_)
            loss = loss_fn(pred, label)

            # backward + optimize
            loss.backward()
            optim.step()
            
            ###################### End of your code ######################
            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
                
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
                ))

        ############## Validation ##############
        validation_l1loss, validation_l2loss = get_validation_performance(net, loss_fn, val_loader, device)
        val_loss.append(validation_l2loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation L2 Loss: {:.4f}".format(validation_l2loss))
        # update stats
        if validation_l2loss > best_l2:
            best_model = copy.deepcopy(net)
            best_l1, best_l2 = validation_l1loss, validation_l2loss 
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'best_l1': best_l1,
             'best_l2': best_l2,
    }

    return best_model, stats


def get_validation_performance(net, loss_fn, data_loader, device):
    net.eval()
    y_true = [] # true labels
    y_pred = [] # predicted labels

    with torch.no_grad():
        for input_, label in data_loader:
            input_, label = input_.to(device), label.to(device)
            loss = None # loss for this batch
            pred = None # predictions for this batch

            ######## TODO: calculate loss, get predictions #########
            ####### You don't need to average loss across iterations #####
            pred = net(input_)
            loss = loss_fn(pred, label)
            ###################### End of your code ######################

            y_true.append(torch.flatten(label.cpu()))
            y_pred.append(torch.flatten(pred.cpu()))
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    l2, l1 = nn.MSELoss(), nn.L1Loss()
    l1loss = l1(y_pred, y_true)
    l2loss = l2(y_pred, y_true)
    
    return l1loss, l2loss
