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

from dataset import MNIST
from utils import train, evaluate

# Print models available
print(dir(models))


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("torch cuda available:", torch.cuda.is_available())
print("CuDNN enabled:", torch.backends.cudnn.enabled)
print("Device:", device)

# Dataset
dataset = MNIST()
model = torch.load('model.pt')

# Evaluate test set error
test_set_preds, loss, val_error, mean_syn, std_syn = evaluate(model, dataset.val_loader)
print(f'Val loss: {loss:.4f}\tVal error (%): {val_error:.4f}\tInference time (ms): {mean_syn:.4f} (std {std_syn:.4f})\tSaving model: {best_val_error > val_error}')

# Save predictions for validation set
test_set_preds, __, __, mean_syn, std_syn = evaluate(model, dataset.test_loader)
submission_df = pd.read_csv("./data/sample_submission.csv")
submission_df['Label'] = test_set_preds.numpy().squeeze()
submission_df.head()
submission_df.to_csv('submission.csv', index=False)

# -------------------------
def make_predictions(data_loader):
    model.eval()
    test_preds = torch.LongTensor()
    
    for batch_idx, data in enumerate(data_loader):
        data = data.unsqueeze(1)
        
        if torch.cuda.is_available():
            data = data.cuda()

        data = data.repeat(1, 3, 1, 1)
        output = model(data)
        
        preds = output.cpu().data.max(1, keepdim=True)[1]
        test_preds = torch.cat((test_preds, preds), dim=0)

    return test_preds
test_set_preds2 = make_predictions(dataset.test_loader)
submission_df = pd.read_csv("./data/sample_submission.csv")
submission_df['Label'] = test_set_preds2.numpy().squeeze()
submission_df.head()
submission_df.to_csv('submission_aux2.csv', index=False)

