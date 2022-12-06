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
        transform=None):
        super().__init__()

        self.root = image_root
        self.image_list = image_list
        self.label = label

        self.transform = transform

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


def get_dataset(settings):
    dataset_name = settings['datasets']['dataset_name']

    if dataset_name == 'RAPv1':
        num_attr = 92
        training_data, test_data = get_rapv1(settings)

        # define the data loaders
        training_dataloader = torch.utils.data.DataLoader(training_data, 
            batch_size=settings['training']['batch_size'], 
            shuffle=True, num_workers=settings['training']['num_workers'])
        test_dataloader = torch.utils.data.DataLoader(test_data,
            batch_size=settings['training']['batch_size'], 
            shuffle=False, num_workers=settings['training']['num_workers'])

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

    return training_dataloader, test_dataloader


def get_rapv1(settings, partition_index=0):
    """
    parse RAP dataset
    """
    data_root = settings['datasets']['data_root']
    # Load the annotation data
    annotation_dir = os.path.join(data_root, 'RAP_annotation')
    img_dir = os.path.join(data_root, 'RAP_dataset')

    annotation = loadmat(os.path.join(annotation_dir, 'RAP_annotation.mat'))
    annotation = annotation['RAP_annotation'][0,0] # np.void
    # partition, label, attribute_chinese, attribute_eng, position, imgage_name, attribute_exp
    partition = annotation[0]
    label = annotation[1]           # (41585, 92)
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

