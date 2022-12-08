import argparse
import os
import pickle

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import DataLoader

import src.batch_engine as batch_engine

from src.models.fusion import FusionModel
import src.models.base_block as base_block
import src.models.model_factory as factory

from src.utils.datasets import get_dataset, ConcatedFeatures
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
    backbone_main(setting, verbose)
    recommender_main(setting, verbose)
    fusion_main(setting, verbose)


def test_main(setting, verbose=False):
    pass


def backbone_main(setting, verbose=False):
    exp_name = setting['name'] + '_backbone'

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

    # optimizer and scheduler
    optimizer = settings.get_optimizer(setting, backbone_model)
    lr_scheduler = settings.get_scheduler(setting, optimizer)
    scheduler_name = setting['training']['lr_scheduler']['name'].lower()

    # training
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    for epoch in range(setting['training']['backbone_epochs']):
        lr = optimizer.param_groups[1]['lr']

        train_loss, train_gt, train_probs, train_imgs = batch_engine.batch_trainer(
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

        valid_loss, valid_gt, valid_probs, valid_imgs = batch_engine.valid_trainer(
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
            logger.result_printting(train_loss, train_result, valid_loss, valid_result)
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

    with open(result_PATH, 'wb') as f:
        pickle.dump(result_list, f)

    return maximum, best_epoch


def recommender_main(setting, verbose=False):
    label_data = np.loadtxt("SWIN_TRAIN_PAR_GT.csv", delimiter=',')


def fusion_main(setting, verbose=False):
    exp_name = setting['name'] + '_fusion'

    _, log_dir = settings.get_dir(setting['exp_dir'], exp_name)

    writter = None
    if verbose:
        writter = logger.get_writter(log_dir, exp_name)

    Reco_train = np.loadtxt('SWIN_TRAIN_SAR_sclaed.csv', delimiter=',')
    # Reco_test = np.loadtxt('SWIN_PAR_SAR_TEST.csv', delimiter=',')

    ST_train = np.loadtxt('SWIN_TRAIN_PAR_PRED.csv', delimiter=',')
    ST_test = np.loadtxt('SWIN_PAR_PRED.csv', delimiter=',')

    Y_train = np.loadtxt('SWIN_TRAIN_PAR_GT.csv', delimiter=',')
    Y_test = np.loadtxt('SWIN_PAR_GT.csv', delimiter=',')

    train_context = ConcatedFeatures(ST_train, Y_train, Y_train, mode='concat')
    test_context = ConcatedFeatures(ST_test, Y_train, Y_test, mode='concat')

    fusion_setting = setting['training']['fusion']

    epochs = fusion_setting['epochs']
    batch_size = fusion_setting['batch_size']
    lr = fusion_setting['learning_rate']
    num_workers = setting['training']['num_workers']

    device = settings.get_device(setting)

    train_dataloader = DataLoader(train_context, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers)
    test_dataloader = DataLoader(test_context, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers)

    labels = train_context.Y
    label_ratio = labels.mean(0) if setting['training']['loss']['sample_weight'] else None

    model = FusionModel()
    criteria = factory.build_loss()(
        sample_weight=label_ratio,
        scale=1,
        size_sum=True,
        tb_writer=writter
    )
    criteria = criteria.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(epochs):
        batch_num = 0
        pred_list = np.zeros((1,51))
        gt_list = np.zeros((1,51))

        for data, data_rc, label in train_dataloader:
            data, data_rc, label = data.to(device), data_rc.to(device), label.to(device)

            Reco_pred = []

            for idx in range(len(data)):
                bin_data = data[idx].unsqueeze(0)
                bin_data = bin_data.clone().detach().cpu().numpy() > 0.5
                bin_data = bin_data.astype(int)

                hamming = cdist(bin_data, Y_train, 'hamming')
                smallest_index = np.argmin(hamming)

                Reco_pred.append(Reco_train[smallest_index].tolist())

            Reco_pred = torch.tensor(Reco_pred).cuda()
            Reco_pred = torch.sigmoid(Reco_pred)
            input = torch.cat((Reco_pred, data), dim=1)
            pred = model(input)

            loss, _ = criteria.forward(pred, label)
            loss = loss[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_num += 1

            pred = pred.detach().cpu().numpy()
            pred_list = np.append(pred_list, pred, axis=0)
            gt_list = np.append(gt_list, label.detach().cpu().numpy(), axis=0)

        train_result = toolkits.get_pedestrian_metrics(gt_list[1:], pred_list[1:], threshold=0.5)

        # validation step
        pred_list = np.zeros((1,51))
        gt_list = np.zeros((1,51))

        with torch.no_grad():
            for data, data_rc, label in test_dataloader:
                data, data_rc, label = data.to(device), data_rc.to(device), label.to(device)

                Reco_pred = []
                for i in range(len(data)):
                    bin_data = data[i].unsqueeze(0)
                    bin_data = bin_data.clone().detach().cpu().numpy() > 0.5
                    bin_data = bin_data.astype(int)

                    hamming = cdist(bin_data, Y_train, 'hamming')
                    smallest_index = np.argmin(hamming)

                    Reco_pred.append(Reco_train[smallest_index].tolist())

                Reco_pred = torch.tensor(Reco_pred).cuda()
                Reco_pred = torch.sigmoid(Reco_pred)
                input = torch.cat((Reco_pred, data), dim=1)
                pred = model(input)

                batch_num += 1
                val_loss, _ = criteria.forward(pred, label)

                pred = pred.detach().cpu().numpy()
                pred_list = np.append(pred_list, pred, axis=0)
                gt_list = np.append(gt_list, label.detach().cpu().numpy(), axis=0)

        valid_result = toolkits.get_pedestrian_metrics(gt_list[1:], pred_list[1:], threshold=0.5)
        if verbose:
            logger.result_printting(-1.0, train_result, -1.0, valid_result)
        
        scheduler.step(val_loss[0])


if __name__ == '__main__':
    args = argument_parser()

    toolkits.set_seed(args.seed)
    setting = settings.get_settings(args.setting)
    verbose = args.verbose

    if args.training:
        train_main(setting, verbose)
    else:
        test_main(setting, verbose)