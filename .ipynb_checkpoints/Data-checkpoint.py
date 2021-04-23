import torch
import os

def save_model(model, optimizer, loss_func, scheduler, 
                epoch, train_loss, test_loss, 
                BATCH_SIZE, LR, WEIGHT_DECAY,
                INSTRUMENT, ):
    torch.save({
                'model_name': model.__class__.__name__,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_func': loss_func,
                'scheduler': scheduler,

                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,

                'BATCH_SIZE': BATCH_SIZE,
                'LR': LR,
                'WEIGHT_DECAY': WEIGHT_DECAY, 

                'INSTRUMENT' : INSTRUMENT, 

                }, 
            f'model/{model.__class__.__name__}.pkl')

def load_model(path):
    checkpoint = torch.load(path)
    
    model = eval(f'architecture.{checkpoint["model_name"]}().to(device)')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load(path):
    checkpoint = torch.load(path)
    
    model = eval(f'architecture.{checkpoint["model_name"]}().to(device)')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_func = checkpoint['loss_func']
    scheduler = checkpoint['scheduler']

    return model, optimizer, loss_func, scheduler

def load_hyperparam(path):
    checkpoint = torch.load(path)
    
    SIZE = checkpoint['SIZE']
    BATCH_SIZE = checkpoint['BATCH_SIZE']
    LR = checkpoint['LR']
    WEIGHT_DECAY = checkpoint['WEIGHT_DECAY']

    return SIZE, BATCH_SIZE, LR, WEIGHT_DECAY