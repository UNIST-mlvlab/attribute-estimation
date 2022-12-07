import os

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