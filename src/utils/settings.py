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


def get_optimizer(setting, backbone_model):
    optimizer_name = setting['training']['optimizer']['name'].lower()
    
    param_groups =[{'params': backbone_model.module.finetune_params(),
                    'lr': setting['training']['lr_scheduler']['lr_ft'],
                    'weight_decay': setting['training']['optimizer']['weight_decay']},
                    {'params': backbone_model.module.fresh_params(),
                     'lr': setting['training']['lr_scheduler']['lr_new'],
                     'weight_decay': setting['training']['optimizer']['weight_decay']}]
    
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=setting['training']['optimizer']['momentum'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError('Optimizer is not supported yet.')

    return optimizer


def get_scheduler(setting, optimizer):
    # scheduler
    scheduler_name = setting['training']['lr_scheduler']['name'].lower()
    if scheduler_name == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
    elif scheduler_name == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=setting['training']['lr_scheduler']['lr_step'], gamma=0.1)
    elif scheduler_name == 'annealing_cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR_with_Restart(optimizer, 
            T_max=setting['training']['lr_scheduler']['T_max'], 
            T_mult=setting['training']['lr_scheduler']['T_mult'],
            eta_min=setting['training']['lr_scheduler']['eta_min'])
    elif scheduler_name == 'warmup_cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineLRScheduler(optimizer,
            t_initial=setting['training']['lr_scheduler']['t_initial'],
            lr_min=1e-5,
            warmup_lr_init=1e-4,
            warmup_t=setting['training']['lr_scheduler']['warmup_t'],)
    else:
        lr_scheduler = None

    return lr_scheduler