#import external libraries
import pandas as pd
from sklearn.model_selection import train_test_split

#pytorch utility imports
import torch
from torch.utils.data import DataLoader, TensorDataset
#import torchvision.transforms as transforms

class MNIST:
    def __init__(self):
        input_folder_path = "../data/"
        train_df = pd.read_csv("./data/train.csv")
        test_df = pd.read_csv("./data/test.csv")

        train_labels = train_df['label'].values
        train_images = (train_df.iloc[:,1:].values).astype('float32')
        test_images = (test_df.iloc[:,:].values).astype('float32')

        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                            stratify=train_labels, random_state=123,
                                                                            test_size=0.20)

        train_images = train_images.reshape(train_images.shape[0], 28, 28)
        val_images = val_images.reshape(val_images.shape[0], 28, 28)
        test_images = test_images.reshape(test_images.shape[0], 28, 28)

        #train
        train_images_tensor = torch.tensor(train_images)/255.0
        train_labels_tensor = torch.tensor(train_labels)
        train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)

        #val
        val_images_tensor = torch.tensor(val_images)/255.0
        val_labels_tensor = torch.tensor(val_labels)
        val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)

        #test
        test_images_tensor = torch.tensor(test_images)/255.0

        self.train_loader = DataLoader(train_tensor, batch_size=16, num_workers=2, shuffle=True)
        self.val_loader = DataLoader(val_tensor, batch_size=16, num_workers=2, shuffle=False)
        self.test_loader = DataLoader(test_images_tensor, batch_size=16, num_workers=2, shuffle=False)

'''
# https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
from torchvision import transforms
	transform = transforms.Compose([
	 transforms.Resize(256),
	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
	 transforms.Normalize(
	 mean=[0.485, 0.456, 0.406],
	 std=[0.229, 0.224, 0.225]
	 )])

img_t = transform(img)
'''
