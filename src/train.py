
# Import external libraries
import os
import argparse

# Import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

# Other libs
import mlflow
import kornia.augmentation as K

# Import data reading and other utilities
from dataset import Kaggle_Digit_Recognizer
from utils import train, evaluate

if __name__ == "__main__":

    # Process command line parameters
    parser = argparse.ArgumentParser(description='Train network for MNIST')
    parser.add_argument('--model_name', choices=dir(models), default='resnet18', help='Model name')
    parser.add_argument('--no-pretrain', dest='pretrain', action='store_false')
    parser.set_defaults(pretrain=True)
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to perform training')
    parser.add_argument('--l2_reg', type=float, default=0.001, help='L2 weight penalty')    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')        
    parser.add_argument('--augment_affine', type=float, default=8., help='Strength of affine augmentation (%, 0 to disable)')    
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

    # Affine augmentation
    transform = torch.nn.Sequential(
        K.RandomAffine(degrees=(180. / 100.)*args.augment_affine, p=0.5),
        K.RandomAffine(degrees=0, translate=(args.augment_affine / 100, args.augment_affine / 100), p=0.5),
        K.RandomAffine(degrees=0, scale=(1.-args.augment_affine / 100, 1.+args.augment_affine / 100), p=0.5),
        ) if args.augment_affine > 0. else None

    # Dataset
    dataset = Kaggle_Digit_Recognizer(transform=transform)

    # Debug code to store augmented images to disk
    '''
    import cv2
    index=4
    for i in range(10):
        cv2.imwrite("img_"+str(index)+"_"+str(i)+".png", (128*dataset.train_tensor[index][0]).cpu().numpy())
    quit()
    '''

    # Generate Pytorch data loaders
    batch_size = 64
    train_loader = DataLoader(dataset.train, batch_size=args.batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(dataset.val, batch_size=args.batch_size, num_workers=2, shuffle=False)
    test_loader = DataLoader(dataset.test, batch_size=args.batch_size, num_workers=2, shuffle=False)

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

    with mlflow.start_run(run_name=f'{args.model_name}_{args.augment_affine}_{args.pretrain}_{args.l2_reg}'):
        for num_epoch in range(args.epochs):
            train(model, criterion, optimizer, lr_scheduler, train_loader, num_epoch)
            test_set_preds, loss, val_error, mean_syn, std_syn = evaluate(model, val_loader)
            print(f'Val loss: {loss:.4f}\tVal error (%): {val_error:.4f}\tInference time (ms): {mean_syn:.4f} (std {std_syn:.4f})\tSaving model: {best_val_error > val_error}')
            mlflow.log_metric("loss", loss, num_epoch)
            mlflow.log_metric("error", val_error, num_epoch)
            if best_val_error > val_error:
                best_val_error = val_error
                # We could alternatively log the model in the MLflow experiment
                # This way is more convenient for the purpose of the exercise
                mlflow.pytorch.log_model(model, "model")

# Evaluate final test set error
test_set_preds, loss, val_error, mean_syn, std_syn = evaluate(model, val_loader)
print(f'FINAL - Val loss: {loss:.4f}\tVal error (%): {val_error:.4f}\tInference time (ms): {mean_syn:.4f} (std {std_syn:.4f})')
