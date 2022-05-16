
# Import external libraries
import os
import argparse

# Import torch
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Other libs
import mlflow

# Import data reading and other utilities
from dataset import MNIST
from utils import train, evaluate

if __name__ == "__main__":

    # Process command line parameters
    parser = argparse.ArgumentParser(description='Train network for MNIST')
    parser.add_argument('--model_name', choices=dir(models), default='resnet18', help='Model name')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.set_defaults(augment=False)
    parser.add_argument('--pretrain', action='store_true', help='Load pretrain weights')
    parser.add_argument('--no-pretrain', dest='pretrain', action='store_false')
    parser.set_defaults(pretrain=True)
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to perform training')
    parser.add_argument('--l2_reg', type=float, default=0.001, help='L2 weight penalty')    
    args = parser.parse_args()

    # Config parameters
    print("Args:", args)

    # Detect GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Torch cuda available:", torch.cuda.is_available())
    print("CuDNN enabled:", torch.backends.cudnn.enabled)

    # Dataset
    dataset = MNIST(batch_size = 64, augment=args.augment)

    # Model
    model = getattr(models, args.model_name)(pretrained=args.pretrain)

    # Training config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=args.l2_reg)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    # Move to adequate processor
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Training loop
    best_val_error = 100.

    augment_str = "aug" if args.augment else ""
    pretrain_str = "pre" if args.pretrain else ""
    with mlflow.start_run(run_name=f'{args.model_name}_{augment_str}_{pretrain_str}_{args.l2_reg}'):
        for num_epoch in range(args.epochs):
            train(model, criterion, optimizer, lr_scheduler, dataset.train_loader, num_epoch)
            test_set_preds, loss, val_error, mean_syn, std_syn = evaluate(model, dataset.val_loader)
            print(f'Val loss: {loss:.4f}\tVal error (%): {val_error:.4f}\tInference time (ms): {mean_syn:.4f} (std {std_syn:.4f})\tSaving model: {best_val_error > val_error}')
            mlflow.log_metric("loss", loss, num_epoch)
            mlflow.log_metric("error", val_error, num_epoch)
            if best_val_error > val_error:
                best_val_error = val_error
                # We could alternatively log the model in the MLflow experiment
                # This way is more convenient for the purpose of the exercise
                mlflow.pytorch.log_model(model, "model")

# Evaluate final test set error
test_set_preds, loss, val_error, mean_syn, std_syn = evaluate(model, dataset.val_loader)
print(f'FINAL - Val loss: {loss:.4f}\tVal error (%): {val_error:.4f}\tInference time (ms): {mean_syn:.4f} (std {std_syn:.4f})')
