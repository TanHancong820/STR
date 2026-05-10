from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from Optim import ScheduledOptim

from dataloader import AVE_Weakly_Dataset
from weakly_model import psp_net
from measure import compute_acc

import os
import time
import warnings
warnings.filterwarnings("ignore")
import argparse
import pdb



parser = argparse.ArgumentParser(description='Weakly supervised AVE Localization')

# Data specifications
parser.add_argument('--model_name', type=str, default='PSP', help='model name')
parser.add_argument('--dir_video', type=str, default="./data/video_clip_feature_1frame_vitl14_gat_residual.h5", help='visual features')
parser.add_argument('--dir_audio', type=str, default='./data/audio_embedding.h5', help='audio features')
parser.add_argument('--dir_labels', type=str, default='./data/mil_labels.h5', help='labels of AVE dataset')
parser.add_argument('--prob_dir_labels', type=str, default='./data/prob_label.h5', help='audio-visual corresponces labels (normalized) of AVE dataset')

parser.add_argument('--dir_video_bg', type=str, default="./data/visual_noisy.h5", help='dataset directory')
parser.add_argument('--dir_audio_bg', type=str, default='./data/audio_noisy.h5', help='dataset directory')
parser.add_argument('--dir_labels_bg', type=str, default='./data/labels_noisy.h5', help='dataset directory')
parser.add_argument('--dir_labels_gt', type=str, default='./data/right_labels.h5', help='dataset directory')

parser.add_argument('--dir_order_train', type=str, default='./data/train_order.h5', help='indices of training samples')
parser.add_argument('--dir_order_val', type=str, default='./data/val_order.h5', help='indices of validation samples')
parser.add_argument('--dir_order_test', type=str, default='./data/test_order.h5', help='indices of testing samples')

parser.add_argument('--nb_epoch', type=int, default=300, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--save_epoch', type=int, default=5, help='number of epoch for saving models')
parser.add_argument('--check_epoch', type=int, default=5, help='number of epoch for checking accuracy of current models during training')
parser.add_argument('--threshold', type=float, default=0.095, help='key-parameter for pruning process')
parser.add_argument('--clip_lambda', type=float, default=1, help='weight for CLIP similarity supervision')

parser.add_argument('--trained_model_path', type=str, default=None, help='pretrained model')
parser.add_argument('--train', action='store_true', default=False, help='train a new model')


FixSeed = 123
random.seed(FixSeed)
np.random.seed(FixSeed)
torch.manual_seed(FixSeed)
torch.cuda.manual_seed(FixSeed)


def train(args, net_model, optimizer):
    AVEData = AVE_Weakly_Dataset(video_dir=args.dir_video, video_dir_bg=args.dir_video_bg, audio_dir=args.dir_audio, \
                        audio_dir_bg=args.dir_audio_bg, label_dir=args.dir_labels, prob_label_dir=args.prob_dir_labels, label_dir_bg=args.dir_labels_bg, \
                        label_dir_gt = args.dir_labels_gt, order_dir=args.dir_order_train, batch_size=args.batch_size, status = "train")

    nb_batch = AVEData.__len__() // args.batch_size
    print(AVEData.__len__())
    print('nb_batch:', nb_batch)
    epoch_l = []
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(args.nb_epoch):

        net_model.train() #
        epoch_loss = 0
        epoch_loss_cls = 0
        # epoch_loss_clip = 0
        epoch_loss_category = 0
        n = 0
        start = time.time()
        SHUFFLE_SAMPLES = True
        for i in range(nb_batch):
            audio_inputs, video_inputs, labels, prob_labels, sd_mask_batch = AVEData.get_batch(i, SHUFFLE_SAMPLES)
            # labels: [bs, 29]
            # video: (bs, 10, 7, 7, 512)
            SHUFFLE_SAMPLES = False
            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            labels = labels.cuda()
            prob_labels = prob_labels.cuda()
            sd_mask_batch = sd_mask_batch.cuda()  # ★ 稀疏/稠密标签
            # av_match = av_match.cuda()

            net_model.zero_grad()

            scores_avg, out_prob, sim_va = net_model(audio_inputs, video_inputs, sparse_dense_mask=sd_mask_batch)

            loss_cls_prob = nn.BCELoss()(scores_avg, prob_labels)
            
#             # 是否有“任何事件”（视频级）
#             video_has_event = (prob_labels.sum(dim=1) > 0).float()  # [B]

#             # 全局 CLIP loss
#             loss_clip = nn.BCEWithLogitsLoss()(sim_avg, video_has_event)

            # loss = loss_cls_prob + args.clip_lambda * loss_clip
            loss = loss_cls_prob

            epoch_loss += loss.cpu().data.numpy()
            # epoch_loss_clip += loss_clip.item()
            epoch_loss_cls += loss_cls_prob.item()
            loss.backward()
            optimizer.step_lr()
            n = n + 1

        SHUFFLE_SAMPLES = True

        # if (epoch+1) % 80 == 0 and epoch < 200:
        #     optimizer.update_lr()

        end = time.time()
        epoch_l.append(epoch_loss)

        print("=== Epoch {%s} | lr: {%.6f} | "
    "Total Loss: {%.4f} | "
    "CLS Loss: {%.4f} | "
    % (
        str(epoch), optimizer._optimizer.param_groups[0]['lr'], epoch_loss / n, epoch_loss_cls / n
    )
)


        if epoch % args.save_epoch == 0 and epoch != 0:
            val_acc = val(args, net_model)
            print('val accuracy:', val_acc, 'epoch=', epoch)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                print('best val accuracy: {} *******************************'.format(best_val_acc))
                # torch.save(net_model, model_name + "_" + str(epoch) + "_weakly.pt")
        if epoch % args.check_epoch == 0 and epoch != 0:
            test_acc = test(args, net_model)
            print('test accuracy:', test_acc, 'epoch=', epoch)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                print('best test accuracy: {} ================================='.format(best_test_acc))
                torch.save(net_model, model_name + "_" + str(epoch) + "_weakly.pt")

    print('[best val accuracy]: ', best_val_acc)
    print('[best test accuracy]: ', best_test_acc)


def val(args, net_model):
    net_model.eval()

    AVEData = AVE_Weakly_Dataset(
        video_dir=args.dir_video,
        video_dir_bg=args.dir_video_bg,
        audio_dir=args.dir_audio,
        audio_dir_bg=args.dir_audio_bg,
        label_dir=args.dir_labels,
        prob_label_dir=args.prob_dir_labels,
        label_dir_bg=args.dir_labels_bg,
        label_dir_gt=args.dir_labels_gt,
        order_dir=args.dir_order_val,
        batch_size=402,
        status="val"
    )

    nb_batches = len(AVEData) // AVEData.batch_size
    remainder = len(AVEData) % AVEData.batch_size
    if remainder > 0:
        nb_batches += 1

    all_labels = []
    all_preds = []

    for i in range(nb_batches):
        batch_size = AVEData.batch_size
        if i == nb_batches - 1 and remainder > 0:
            batch_size = remainder

        audio_inputs, video_inputs, labels, sd_mask_batch = AVEData.get_batch(i)

        audio_inputs = audio_inputs.cuda()
        video_inputs = video_inputs.cuda()
        sd_mask_batch = sd_mask_batch.cuda()
        labels_np = labels.numpy()

        scores_avg, x_labels, sim_va = net_model(audio_inputs, video_inputs, sparse_dense_mask=sd_mask_batch)
        x_labels = F.softmax(x_labels, dim=-1).cpu().data.numpy()

        all_labels.append(labels_np)
        all_preds.append(x_labels)

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    acc = compute_acc(all_labels, all_preds, len(AVEData))
    print('val accuracy', acc)
    return acc


def test(args, net_model, model_path=None):
    if model_path is not None:
        net_model = torch.load(model_path)
        print(">>> [Testing] Load pretrained model from " + model_path)

    net_model.eval()
    AVEData = AVE_Weakly_Dataset(
        video_dir=args.dir_video,
        video_dir_bg=args.dir_video_bg,
        audio_dir=args.dir_audio,
        audio_dir_bg=args.dir_audio_bg,
        label_dir=args.dir_labels,
        prob_label_dir=args.prob_dir_labels,
        label_dir_bg=args.dir_labels_bg,
        label_dir_gt=args.dir_labels_gt,
        order_dir=args.dir_order_test,
        batch_size=402,
        status="test"
    )

    nb_batches = len(AVEData) // AVEData.batch_size
    remainder = len(AVEData) % AVEData.batch_size
    if remainder > 0:
        nb_batches += 1

    all_labels = []
    all_preds = []

    for i in range(nb_batches):
        batch_size = AVEData.batch_size
        if i == nb_batches - 1 and remainder > 0:
            batch_size = remainder

        audio_inputs, video_inputs, labels, sd_mask_batch = AVEData.get_batch(i)

        audio_inputs = audio_inputs.cuda()
        video_inputs = video_inputs.cuda()
        sd_mask_batch = sd_mask_batch.cuda()
        labels_np = labels.numpy()

        scores_avg, x_labels, sim_va = net_model(audio_inputs, video_inputs, sparse_dense_mask=sd_mask_batch)
        x_labels = F.softmax(x_labels, dim=-1).cpu().data.numpy()

        all_labels.append(labels_np)
        all_preds.append(x_labels)

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    acc = compute_acc(all_labels, all_preds, len(AVEData))
    print('test accuracy', acc)
    return acc



if __name__ == "__main__":
    args = parser.parse_args()
    print("args:", args)

    # model and optimizer
    model_name = args.model_name
    if model_name == "PSP":
        net_model = psp_net(2048, 512, 256, 29)
    else:
        raise NotImplementedError
    net_model.cuda()


    # base_parameters = []
    # for name, parameter in net_model.named_parameters():
    #     if parameter.requires_grad:
    #         print(name)
    #         base_parameters.append(parameter)

    optimizer = optim.Adam(net_model.parameters(), lr=1e-4)
    optimizer = ScheduledOptim(optimizer)

    # train or test
    if args.train:
        train(args, net_model, optimizer)
    else:
        test_acc = test(args, net_model, model_path=args.trained_model_path)
        print("[test] accuracy: ", test_acc)