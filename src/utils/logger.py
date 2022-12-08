import os

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import src.utils.toolkits as toolkits

def get_writter(log_dir, exp_name):
    current_time = toolkits.time_stamp()
    writer_dir = os.path.join(log_dir, exp_name, 'runs', current_time)
    os.makedirs(writer_dir, exist_ok=True)
    
    writter = SummaryWriter(writer_dir)

    return writter


def save_ckpt(model, ckpt_files, epoch, metric):
    """
    Note:
        torch.save() reserves device type and id of tensors to save.
        So when loading ckpt, you have to inform torch.load() to load these tensors
        to cpu or your desired gpu, if you change devices.
    """

    if not os.path.exists(os.path.dirname(os.path.abspath(ckpt_files))):
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_files)))

    save_dict = {'state_dicts': model.state_dict(),
                 'state_dict_ema': unwrap_model(model).state_dict(),
                 'epoch': f'{toolkits.time_stamp()} in epoch {epoch}',
                 'metric': metric,}

    torch.save(save_dict, ckpt_files)


def tb_visualizer_pedes(tb_writer, lr, epoch, train_loss, valid_loss, train_result, valid_result):
    tb_writer.add_scalars('train/lr', {'lr': lr}, epoch)
    tb_writer.add_scalars('train/losses', {'train': train_loss,
                                         'test': valid_loss}, epoch)

    tb_writer.add_scalars('train/perf', {'ma': train_result.ma,
                                         'pos_recall': np.mean(train_result.label_pos_recall),
                                         'neg_recall': np.mean(train_result.label_neg_recall),
                                         'Acc': train_result.instance_acc,
                                         'Prec': train_result.instance_prec,
                                         'Rec': train_result.instance_recall,
                                         'F1': train_result.instance_f1}, epoch)

    tb_writer.add_scalars('test/perf', {'ma': valid_result.ma,
                                        'pos_recall': np.mean(valid_result.label_pos_recall),
                                        'neg_recall': np.mean(valid_result.label_neg_recall),
                                        'Acc': valid_result.instance_acc,
                                        'Prec': valid_result.instance_prec,
                                        'Rec': valid_result.instance_recall,
                                        'F1': valid_result.instance_f1}, epoch)


def result_printting(train_loss, train_result, valid_loss, valid_result):
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