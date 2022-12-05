import torch
from torch.utils.data import Dataset
from torchvision import datasets


def get_dataset(settings):
    dataset_name = settings['datasets']['dataset_name']
    if dataset_name == 'RAPv1':
        training_data, test_data = get_rapv1(settings)

        # define the data loaders
        training_dataloader = torch.utils.data.DataLoader(training_data, 
            batch_size=settings['training']['batch_size'], 
            shuffle=True, num_workers=settings['training']['num_workers'])
        test_dataloader = torch.utils.data.DataLoader(test_data,
            batch_size=settings['training']['batch_size'], 
            shuffle=False, num_workers=settings['training']['num_workers'])



def get_rapv1(settings):
    pass