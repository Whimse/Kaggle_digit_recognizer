
# Import external libraries
import numpy as np
import pandas as pd
import argparse

if __name__ == "__main__":

    # Process command line parameters
    parser = argparse.ArgumentParser(description='Train network for MNIST')
    parser.add_argument('prediction_files', type=str, nargs='+', help='Model filename')
    args = parser.parse_args()

    # Config parameters
    print("Args:", args.prediction_files)

    predictions = []
    for submission_file in args.prediction_files:
        print (f"Processing {submission_file}")

        # Save predictions for validation set
        submission_df = pd.read_csv(submission_file)
        predictions.append(submission_df['Label'].to_numpy())
    predictions = np.array(predictions).transpose()

    # Get most frequent value per row
    most_frequent_values = []
    for row in predictions:
        most_frequent_values.append(np.bincount(row).argmax())

    # Print some of the non-coherent predictions in the ensemble and their consensuated prediction
    #idxs = np.where(np.any(np.tile(predictions[:,0], (6,1)).transpose() != predictions, axis=1))[0]
    #for idx in idxs[:10]:
    #    print(idx, predictions[idx], most_frequent_values[idx])

    # Save consensuated predictions
    submission_df = pd.read_csv("./data/sample_submission.csv")
    submission_df['Label'] = most_frequent_values
    submission_df.head()
    submission_df.to_csv('submission_ensemble.csv', index=False)




