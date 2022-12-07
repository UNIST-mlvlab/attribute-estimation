import argparse
import datetime
import os
import pickle

from collections import defaultdict

import numpy as np
import torch

import src.batch_engine as batch_engine

import src.models.base_block as base_block
import src.models.model_factory as factory

from src.utils.datasets import get_dataset
import src.utils.logger as logger
import src.utils.settings as settings
import src.utils.toolkits as toolkits


def argument_parser():
    parser = argparse.ArgumentParser(description='Attribute Recognition Framework',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--setting', type=str, default='config.yaml',
        help='Path to the setting file')

    parser.add_argument('-T', '--training', action='store_true',
        help='Training mode')
    parser.add_argument('-V', '--verbose', action='store_true',
        help='Print the training process')

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


def train_main(setting, verbose=False):
    exp_name = setting['name']

    current_time = toolkits.time_stamp()
    model_dir, log_dir = settings.get_dir(setting['exp_dir'], exp_name)
    model_PATH = os.path.join(model_dir, 'ckpt_'+current_time +'.pth')
    result_PATH = os.path.join(model_dir, 'result_'+current_time +'.pkl')

    writter = None
    if verbose:
        writter = logger.get_writter(log_dir, exp_name)

    training_dataloader, test_dataloader, dataset_info = get_dataset(setting)    

    # build backbone
    backbone, c_output = factory.build_backbone(setting['models']['backbone'])
    
    classify_dict = setting['models']['classifier']
    classifier = factory.build_classifier(classify_dict['name'])(
        nattr = dataset_info['num_attr'],
        c_in = c_output,
        bn=classify_dict['bn'],
        pool=classify_dict['pool'],
        scale=classify_dict['scale'],
    )

    backbone_model = base_block.FeatClassifier(backbone, classifier, bn_wd=True)
    device = settings.get_device(setting)
    backbone_model = backbone_model.to(device)

    if setting['training']['distributed']:
        raise ValueError('Distributed training is not supported yet.') 
    else:
        backbone_model = torch.nn.DataParallel(backbone_model)

    # loss and criteria
    loss_dict = setting['training']['loss']
    criteria = factory.build_loss(loss_dict['name'])(
        sample_weight=[1,]
        scale=1,
        size_num=True,
        tb_writer=writer
    )

    model_ema = None

    # optimizer
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

    # training
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    for epoch in range(setting['training']['backbone_epochs']):
        lr = optimizer.param_groups[1]['lr']

        train_loss, train_gt, train_probs, train_imgs, _, train_loss_mtr = batch_engine.batch_trainer(
            setting,
            epoch=epoch,
            model=backbone_model,
            model_ema=model_ema,
            train_loader=training_dataloader,
            criterion=criteria,
            optimizer=optimizer,
            loss_w=[1,],
            scheduler=lr if scheduler_name == 'annealing_cosine' else None
        )

        valid_loss, valid_gt, valid_probs, valid_imgs, _, valid_loss_mtr = batch_engine.valid_trainer(
            setting,
            epoch=epoch,
            model=backbone_model,
            valid_loader=test_dataloader,
            criterion=criteria,
            loss_w=[1,]
        )

        if scheduler_name == 'plateau':
            lr_scheduler.step(metrics=valid_loss)
        elif scheduler_name == 'multistep':
            lr_scheduler.step()
        elif scheduler_name == 'warmup_cosine':
            lr_scheduler.step(epoch=epoch+1)

        train_result = toolkits.get_pedestrian_metrics(train_gt, train_probs, index=None)
        valid_result = toolkits.get_pedestrian_metrics(valid_gt, valid_probs, index=None)

        if verbose:
            print(f'Evaluation on train set, train losses {train_loss}\n',
                    'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        train_result.ma, np.mean(train_result.label_f1),
                        np.mean(train_result.label_pos_recall),
                        np.mean(train_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
                        train_result.instance_f1))

            print(f'Evaluation on test set, valid losses {valid_loss}\n',
                    'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        valid_result.ma, np.mean(valid_result.label_f1),
                        np.mean(valid_result.label_pos_recall),
                        np.mean(valid_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                        valid_result.instance_f1))

            print(f'{toolkits.time_stamp()}')
            print('-' * 60)

        if verbose:
            logger.tb_visualizer_pedes(writter, lr, epoch, train_loss, valid_loss, train_result, valid_result)

        cur_metric = valid_result.ma
        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = epoch
            logger.save_ckpt(backbone_model, model_PATH, epoch, maximum)

        result_list[epoch] = {
            'train_result': train_result,  # 'train_map': train_map,
            'valid_result': valid_result,  # 'valid_map': valid_map,
            'train_gt': train_gt, 'train_probs': train_probs,
            'valid_gt': valid_gt, 'valid_probs': valid_probs,
            'train_imgs': train_imgs, 'valid_imgs': valid_imgs
        }

    with open(result_PATH.replace(':', '.'), 'wb') as f:
        pickle.dump(result_list, f)

    return maximum, best_epoch

def test_main(setting, verbose=False):
    pass


if __name__ == '__main__':
    args = argument_parser()

    toolkits.set_seed(args.seed)
    setting = settings.get_settings(args.setting)
    verbose = args.verbose

    if args.training:
        train_main(setting, verbose)
    else:
        test_main(setting, verbose)