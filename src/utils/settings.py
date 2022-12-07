import os
import yaml

import torch

def make_dir(path):
    os.makedirs(path, exist_ok=True)

def get_settings(setting_file='settings.yaml'):
    with open(setting_file, 'r') as f:
        return yaml.load(f)


def get_dir(root_dir, exp_name):
    """
    Get directory path
    """
    root_dir = os.path.join(root_dir, exp_name)
    make_dir(root_dir)

    model_dir = os.path.join(root_dir, 'models')
    log_dir = os.path.join(root_dir, 'logs')

    make_dir(model_dir)
    make_dir(log_dir)

    return model_dir, log_dir
    

def get_device(settings):
    """
    Get device
    """
    if settings['training']['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:'+settings['training']['device']  if torch.cuda.is_available() else 'cpu')

    return device