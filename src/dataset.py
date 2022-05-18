# Import external libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Import Torch
import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x).squeeze()

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class Kaggle_Digit_Recognizer:
    """ Class to generate MNIST dataset readers for the Digit Recognizer Kaggle competition """
    def __init__(self, transform=None):

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

        # Reshape and normalize images
        train_images = train_images.reshape(train_images.shape[0], 28, 28)/255.0
        val_images = val_images.reshape(val_images.shape[0], 28, 28)/255.0
        test_images = test_images.reshape(test_images.shape[0], 28, 28)/255.0

        # Convert images to Pytorch tensor
        train_images_tensor = torch.tensor(train_images)
        train_labels_tensor = torch.tensor(train_labels)
        self.train = CustomTensorDataset((train_images_tensor, train_labels_tensor), transform=transform)

        val_images_tensor = torch.tensor(val_images)
        val_labels_tensor = torch.tensor(val_labels)
        self.val = CustomTensorDataset((val_images_tensor, val_labels_tensor))

        self.test = torch.tensor(test_images)
