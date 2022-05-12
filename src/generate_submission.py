
# Import external libraries
import pandas as pd
import argparse

# Import torch
import torch

# Import data reading and other utilities
from dataset import MNIST
from utils import evaluate

if __name__ == "__main__":

    # Process command line parameters
    parser = argparse.ArgumentParser(description='Train network for MNIST')
    parser.add_argument('model_file', type=str, help='Model filename')
    args = parser.parse_args()

    # Config parameters
    print("Args:", args.model_file)

    # Dataset
    dataset = MNIST()
    model = torch.load(args.model_file)

    # Evaluate test set error
    test_set_preds, loss, val_error, mean_syn, std_syn = evaluate(model, dataset.val_loader)
    val_accuracy = (1. - val_error / 100.)
    print(f'Val loss: {loss:.4f}\tVal accuracy (%): {val_accuracy:.4f}\tInference time (ms): {mean_syn:.4f} (std {std_syn:.4f})')

    # Save predictions for validation set
    test_set_preds, __, __, __, __ = evaluate(model, dataset.test_loader)
    submission_df = pd.read_csv("./data/sample_submission.csv")
    submission_df['Label'] = test_set_preds.numpy().squeeze()
    submission_df.head()
    submission_df.to_csv('submission.csv', index=False)
