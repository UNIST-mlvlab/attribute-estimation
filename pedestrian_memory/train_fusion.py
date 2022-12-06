import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import binarize, normalize
from vit_pytorch.efficient import ViT
import math
import torch.optim as optim
from linformer import Linformer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from vit_pytorch.deepvit import DeepViT
from torch.utils.data import DataLoader, Dataset
import math
from easydict import EasyDict
from sklearn.preprocessing import binarize, MinMaxScaler

from scipy.spatial.distance import cdist
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmcv.cnn import get_model_complexity_info
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

from configs import cfg, update_config
from dataset.multi_label.coco import COCO14
from dataset.augmentation import get_transform
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_ema import ModelEmaV2
from optim.adamw import AdamW
from scheduler.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from scheduler.cosine_lr import CosineLRScheduler
from tools.distributed import distribute_bn
from tools.vis import tb_visualizer_pedes
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from dataset.pedes_attr.pedes import PedesAttr
from models.base_block import FeatClassifier
from models.model_factory import build_loss, build_classifier, build_backbone

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool
from models.backbone import swin_transformer, resnet, bninception, vit
# from models.backbone.tresnet import tresnet
from losses import bceloss, scaledbceloss
from models import base_block

from models.model_factory import build_loss, build_classifier, build_backbone
import argparse
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool
from configs import cfg, update_config

def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5, index=None, cfg=None):
    """
    index: evaluated label index
    """
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    if index is not None:
        pred_label = pred_label[:, index]
        gt_label = gt_label[:, index]

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    # instance_f1 = np.mean(instance_f1)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result

class Dist(nn.Module):
    def __init__(self):
        super(Dist, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(102, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 51),
        )

    def forward(self, x):
        x = torch.tensor(x)
        self.logits = self.linear_relu_stack(x.type(torch.FloatTensor).cuda())
        return self.logits

class concated_feaures(Dataset):
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

class Weighted_BCELoss(object):
    """
        Weighted_BCELoss was proposed in "Multi-attribute learning for pedestrian attribute recognition in surveillance scenarios"[13].
    """
    def __init__(self):
        super(Weighted_BCELoss, self).__init__()
        self.weights = None
        self.weights = torch.Tensor([0.311434,
                                     0.009980,
                                     0.430011,
                                     0.560010,
                                     0.144932,
                                     0.742479,
                                     0.097728,
                                     0.946303,
                                     0.048287,
                                     0.004328,
                                     0.189323,
                                     0.944764,
                                     0.016713,
                                     0.072959,
                                     0.010461,
                                     0.221186,
                                     0.123434,
                                     0.057785,
                                     0.228857,
                                     0.172779,
                                     0.315186,
                                     0.022147,
                                     0.030299,
                                     0.017843,
                                     0.560346,
                                     0.000553,
                                     0.027991,
                                     0.036624,
                                     0.268342,
                                     0.133317,
                                     0.302465,
                                     0.270891,
                                     0.124059,
                                     0.012432,
                                     0.157340,
                                     0.018132,
                                     0.064182,
                                     0.028111,
                                     0.042155,
                                     0.027558,
                                     0.012649,
                                     0.024504,
                                     0.294601,
                                     0.034099,
                                     0.032800,
                                     0.091812,
                                     0.024552,
                                     0.010388,
                                     0.017603,
                                     0.023446,
                                     0.128917]).cuda()
    def forward(self, output, target, epoch):
        EPS = 1e-12
        if self.weights is not None:
            cur_weights = torch.exp(target + (1 - target * 2) * self.weights)
            loss = cur_weights *  (target * torch.log(output + EPS)) + ((1 - target) * torch.log(1 - output + EPS))
        else:
            loss = target * torch.log(output + EPS) + (1 - target) * torch.log(1 - output + EPS)
        return torch.neg(torch.mean(loss))


'''dataload'''
def main():

    Reco_train = np.loadtxt('SWIN_TRAIN_SAR_sclaed.csv', delimiter=',')
    # Reco_test = np.loadtxt('SWIN_PAR_SAR_TEST.csv', delimiter=',')

    ST_train = np.loadtxt('SWIN_TRAIN_PAR_PRED.csv', delimiter=',')
    ST_test = np.loadtxt('SWIN_PAR_PRED.csv', delimiter=',')

    Y_train = np.loadtxt('SWIN_TRAIN_PAR_GT.csv', delimiter=',')
    Y_test = np.loadtxt('SWIN_PAR_GT.csv', delimiter=',')

    train_context = concated_feaures(ST_train, Y_train, Y_train, mode='concat')
    test_context = concated_feaures(ST_test, Y_train, Y_test, mode='concat')

    '''main'''
    batch_size = 50
    test_batch = 50
    epochs = 500
    lr = 1e-3
    gamma = 0.7
    seed = 50

    device = 'cuda'

    train_loader = DataLoader(
        dataset=train_context,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=test_context,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    writer = None
    if cfg.VIS.TENSORBOARD.ENABLE:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)

        writer_dir = os.path.join(exp_dir, cfg.NAME, 'runs', current_time)
        writer = SummaryWriter(log_dir=writer_dir)

    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/pedes_baseline/rapv1.yaml",

    )

    parser.add_argument("--debug", type=str2bool, default="true")
    parser.add_argument('--local_rank', help='node rank for distributed training', default=0,
                        type=int)
    parser.add_argument('--dist_bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

    args = parser.parse_args()
    update_config(cfg, args)

    labels = train_context.Y
    label_ratio = labels.mean(0) if cfg.LOSS.SAMPLE_WEIGHT else None

    criterion = build_loss(cfg.LOSS.TYPE)(
        sample_weight=label_ratio, scale=cfg.CLASSIFIER.SCALE, size_sum=cfg.LOSS.SIZESUM, tb_writer=writer)
    criterion = criterion.cuda()

    model = Dist().cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(epochs):
        batch_num = 0
        pred_list = np.zeros((1,51))
        gt_list = np.zeros((1,51))
        for data, data_rc, label in train_loader:
            data = data.to(device)
            data_rc = data_rc.to(device)
            label = label.to(device)

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

            loss, lossmtr = criterion.forward(pred, label)
            loss = loss[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_num += 1

            pred = pred.detach().cpu().numpy()
            pred_list = np.append(pred_list, pred, axis=0)
            gt_list = np.append(gt_list, label.detach().cpu().numpy(), axis=0)

        result = get_pedestrian_metrics(gt_list[1:], pred_list[1:], threshold=0.5)
        print(result.label_ma)
        print(result.ma)
        print(result.instance_acc)
        print(result.instance_prec)
        print(result.instance_recall)
        print(result.instance_f1)

        '''val'''

        pred_list = np.zeros((1,51))
        gt_list = np.zeros((1,51))

        with torch.no_grad():
            for data, data_rc, label in valid_loader:
                data = data.to(device)
                data_rc = data_rc.to(device)
                label = label.to(device)

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
                val_loss, val_lossmtr = criterion.forward(pred, label)

                pred = pred.detach().cpu().numpy()
                pred_list = np.append(pred_list, pred, axis=0)
                gt_list = np.append(gt_list, label.detach().cpu().numpy(), axis=0)

        result = get_pedestrian_metrics(gt_list[1:], pred_list[1:], threshold=0.5)
        print(result.label_ma)
        print(result.ma)
        print(result.instance_acc)
        print(result.instance_prec)
        print(result.instance_recall)
        print(result.instance_f1)
        scheduler.step(val_loss[0])

if __name__ == '__main__':
    main()