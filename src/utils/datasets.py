import os

from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as T

from scipy.io import loadmat
from easydict import EasyDict

class PARDataset(Dataset):
    def __init__ (self, 
        image_root, 
        image_list,
        label,
        attribute_name=None,
        transform=None):
        super().__init__()

        self.root = image_root
        self.image_list = image_list
        self.label = label

        self.transform = transform
        self.attribute_name = attribute_name

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        sample_img_list = self.image_list[index].tolist()
        sample_img_list = [i.item() for i in sample_img_list]

        for img_file in sample_img_list:
            image = os.path.join(self.root, img_file)
            image = Image.open(image).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            try:
                images = torch.cat((images, image.unsqueeze(0)), dim=0)
            except:
                images = image.unsqueeze(0) 

        return images, label

    def get_attribute_name(self):
        return self.attribute_name


class ConcatedFeatures(Dataset):
    def compare(self, X_MNet, X_RC):
        for i in range(X_MNet.shape[0]):
            for j in range(X_MNet.shape[1]):
                if np.abs(X_MNet[i][j]-0.5) < np.abs(X_RC[i][j]-0.5):
                    X_MNet[i][j] = X_RC[i][j]
        return X_MNet

    def __init__(self, X_MNet, Y_train, Y, mode='MNet'):
        self.X = None
        self.Y = None
        self.X_RC = None
        if mode == 'concat':
            self.X = X_MNet
            self.X_RC = Y_train
            self.Y = Y

    def __getitem__(self, index):
        X = self.X[index]
        X_RC = self.X_RC[index]
        Y = self.Y[index]
        return X, X_RC, Y

    def __len__(self):
        return len(self.X)


def get_dataset(settings):
    dataset_name = settings['datasets']['dataset_name']
    batch_size = settings['training']['batch_size']
    num_workers = settings['training']['num_workers']

    if dataset_name == 'RAPv1':
        training_data, test_data = get_rapv1(settings)
        num_attr = 51
        attribute_name = test_data.get_attribute_name()

        # define the data loaders
        training_dataloader = torch.utils.data.DataLoader(training_data, 
            batch_size=batch_size, 
            shuffle=True, num_workers=num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_data,
            batch_size=batch_size, 
            shuffle=False, num_workers=num_workers)

    elif dataset_name == 'RAPv2':
        raise ValueError('Not implemented yet')
    elif dataset_name == 'HICO':
        raise ValueError('Not implemented yet')
    elif dataset_name == 'PA-100K':
        raise ValueError('Not implemented yet')
    elif dataset_name == 'PETA':
        raise ValueError('Not implemented yet')
    elif dataset_name == 'WIDER':
        raise ValueError('Not implemented yet')
    else:
        raise ValueError('Dataset not supported')

    dataset_info = {
        'num_attr': num_attr,
        'attribute_name': attribute_name
    }

    return training_dataloader, test_dataloader, dataset_info


def get_rapv1(settings, partition_index=0):
    """
    parse RAP dataset
    """
    data_root = settings['datasets']['data_root']
    # Load the annotation data
    annotation_dir = os.path.join(data_root, 'RAP_annotation')
    img_dir = os.path.join(data_root, 'RAP_dataset')

    group_order = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
               26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 1, 2, 3, 0, 4, 5, 6, 7, 8, 43, 44,
               45, 46, 47, 48, 49, 50]

    annotation = loadmat(os.path.join(annotation_dir, 'RAP_annotation.mat'))
    annotation = annotation['RAP_annotation'][0,0] # np.void
    # partition, label, attribute_chinese, attribute_eng, position, imgage_name, attribute_exp
    partition = annotation[0]
    label = annotation[1][:np.arange(51)]           # (41585, 92)
    attribute_name = annotation[3]  # (92, 1)
    img_file = annotation[5]        # (41585, 1)

    train_partition = partition[partition_index,0][0,0][0][0] - 1 # (33268)
    test_partition = partition[partition_index,0][0,0][1][0] - 1 # (8317)
    
    # define the transforms
    height = 256 # Default 256
    width = 192 # Default 192
    normalize = T.Normalize(mean=[0.1385, 0.1377, 0.1319], std=[0.2432, 0.2357, 0.2331])
    train_transform = T.Compose([
        T.Resize((height, width)),
        #T.Pad(10),
        #T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    test_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    # parse data
    training_data = PARDataset(
        image_root=img_dir,
        image_list=img_file[train_partition][:,0],
        label=label[train_partition],
        transform=train_transform
    )
    test_data = PARDataset(
        image_root=img_dir,
        image_list=img_file[test_partition][:,0],
        label=label[test_partition],
        transform=test_transform
    )

    return training_data, test_data, attribute_name

