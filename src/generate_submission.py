
# Import external libraries
import pandas as pd
import argparse

# Import torch
import torch
from torch.utils.data import DataLoader

# Import data reading and other utilities
from dataset import Kaggle_Digit_Recognizer
from utils import evaluate

if __name__ == "__main__":

    # Process command line parameters
    parser = argparse.ArgumentParser(description='Train network for MNIST')
    parser.add_argument('model_file', type=str, help='Model filename')
    args = parser.parse_args()

    # Config parameters
    print("Args:", args.model_file)

    # Dataset
    dataset = Kaggle_Digit_Recognizer()
    val_loader = DataLoader(dataset.val, batch_size=64, num_workers=2, shuffle=False)
    test_loader = DataLoader(dataset.test, batch_size=64, num_workers=2, shuffle=False)

    model = torch.load(args.model_file)

    # Evaluate test set error
    test_set_preds, loss, val_error, mean_syn, std_syn = evaluate(model, val_loader)
    val_accuracy = (1. - val_error / 100.)
    print(f'Val loss: {loss:.4f}\tVal accuracy (%): {val_accuracy:.4f}\tInference time (ms): {mean_syn:.4f} (std {std_syn:.4f})')

    # Save predictions for validation set
    test_set_preds, __, __, __, __ = evaluate(model, test_loader)
    submission_df = pd.read_csv("./data/sample_submission.csv")
    submission_df['Label'] = test_set_preds.numpy().squeeze()
    submission_df.head()
    submission_df.to_csv('submission.csv', index=False)
