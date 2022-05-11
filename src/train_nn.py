
# Import external libraries
import argparse

# Import torch
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import data reading and other utilities
from dataset import MNIST
from utils import train, evaluate

if __name__ == "__main__":

    # Process command line parameters
    parser = argparse.ArgumentParser(description='Train network for MNIST')
    parser.add_argument('--model_name', choices=dir(models), default='resnet18', help='Model name')
    parser.add_argument('--pretrained', type=bool, default=True, help='Load pretrained weights')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to perform training')
    args = parser.parse_args()

    # Config parameters
    print("Args:", args.model_name, args.epochs, args.pretrained)

    # Detect GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Torch cuda available:", torch.cuda.is_available())
    print("CuDNN enabled:", torch.backends.cudnn.enabled)

    # Dataset
    dataset = MNIST()

    # Model
    model = getattr(models, args.model_name)(pretrained=args.pretrained)

    # Training config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.003)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Move to adequate processor
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Training loop
    best_val_error = 100.

    for n in range(args.epochs):
        train(model, criterion, optimizer, exp_lr_scheduler, dataset.train_loader, n)
        test_set_preds, loss, val_error, mean_syn, std_syn = evaluate(model, dataset.val_loader)
        print(f'Val loss: {loss:.4f}\tVal error (%): {val_error:.4f}\tInference time (ms): {mean_syn:.4f} (std {std_syn:.4f})\tSaving model: {best_val_error > val_error}')
        if best_val_error > val_error:
            best_val_error = val_error
            torch.save(model, f'model_{args.model_name}.pt')
