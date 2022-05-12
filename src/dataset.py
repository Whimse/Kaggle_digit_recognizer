# Import external libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Import Torch
import torch
from torch.utils.data import DataLoader, TensorDataset

class MNIST:
    """ Class to generate MNIST dataset readers for the Digit Recognizer Kaggle competition """
    def __init__(self):
        # We could expose this as a class constructor parameter
        batch_size = 64

        # Read data from files
        input_folder_path = "../data/"
        train_df = pd.read_csv("./data/train.csv")
        test_df = pd.read_csv("./data/test.csv")

        train_labels = train_df['label'].values
        train_images = (train_df.iloc[:,1:].values).astype('float32')
        test_images = (test_df.iloc[:,:].values).astype('float32')

        # Generate train and validation split
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                            stratify=train_labels, random_state=123,
                                                                            test_size=0.20)

        # Reshape images
        train_images = train_images.reshape(train_images.shape[0], 28, 28)
        val_images = val_images.reshape(val_images.shape[0], 28, 28)
        test_images = test_images.reshape(test_images.shape[0], 28, 28)

        # Normalize and convert to tensor
        train_images_tensor = torch.tensor(train_images)/255.0
        train_labels_tensor = torch.tensor(train_labels)
        train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
        val_images_tensor = torch.tensor(val_images)/255.0
        val_labels_tensor = torch.tensor(val_labels)
        val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)
        test_images_tensor = torch.tensor(test_images)/255.0

        # Generate Pytorch data loaders
        self.train_loader = DataLoader(train_tensor, batch_size=batch_size, num_workers=2, shuffle=True)
        self.val_loader = DataLoader(val_tensor, batch_size=batch_size, num_workers=2, shuffle=False)
        self.test_loader = DataLoader(test_images_tensor, batch_size=batch_size, num_workers=2, shuffle=False)
