#pytorch utility imports
import torch
import torchvision
import torchvision.models as models

#neural net imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import external libraries
import os
import pandas as pd
import numpy as np
import tqdm

def train(model, criterion, optimizer, lr_scheduler, loader, num_epoch):
    """ Trains the network for an epoch """
    model.train()
    
    with tqdm.tqdm(loader, desc=f"Epoch {num_epoch}", ncols=30, bar_format='{l_bar}{bar}|', leave=True) as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.unsqueeze(1)
            data, target = data, target
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                
            optimizer.zero_grad()

            data = data.repeat(1, 3, 1, 1)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    lr_scheduler.step()

def evaluate(model, data_loader):
    """ Produces evaluation of network results, given test/validation dataset """    
    model.eval()
    loss = 0
    correct = 0
    test_preds = torch.LongTensor()

    # Required to measure inference time    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((len(data_loader.dataset),1))

    # Main evaluation loop
    for batch_idx, data in enumerate(data_loader):
        # Change behaviour depending on whether the dataset provides labels and images, or just images
        if len(data) == 2:
            data, target = data
            if torch.cuda.is_available():
                target = target.cuda()
        else:
            target = None

        # Adjust size of batch to feed the network
        data = data.unsqueeze(1)
        data = data.repeat(1, 3, 1, 1)        
        
        # Move to GPU, if available, for faster inference
        if torch.cuda.is_available():
            data = data.cuda()
        
        starter.record()    # Measure inference time
        output = model(data)
        ender.record()      # Measure inference time

        # Measure elapsed time.
        torch.cuda.synchronize()    # Neccesary to wait for the GPU to finish computations
        timings[batch_idx] = starter.elapsed_time(ender)

        # Compute predictions
        pred = output.data.max(1, keepdim=True)[1].cpu()
        test_preds = torch.cat((test_preds, pred), dim=0)        

        # Accum loss and accuracy
        try:
            # If target is not None
            loss += F.cross_entropy(output, target, size_average=False).item()
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()
        except TypeError as e:
            # If target is None
            loss, correct = .0, .0

    # Final computations and return values
    loss /= len(data_loader.dataset)
    
    val_error = 100. * (1.- correct / len(data_loader.dataset))
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)

    return test_preds, loss, val_error, mean_syn, std_syn
