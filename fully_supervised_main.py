from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataloader import AVE_Fully_Dataset
from fully_model import psp_net
from measure import compute_acc, AVPSLoss
from Optim import ScheduledOptim

import warnings
warnings.filterwarnings("ignore")
import argparse
import pdb

# -------------------- argparse --------------------
parser = argparse.ArgumentParser(description='Fully supervised AVE localization')

# ================= sparse_dense_labels.npy =================
# sd_label_all = np.load('./data/auto_sparse_dense_labels.npy')  # shape [4143], sparse=0, dense=1
# ==========================================================

# data
parser.add_argument('--model_name', type=str, default='PSP', help='model name')
parser.add_argument('--dir_video', type=str, default="./data/video_clip_feature_1frame_vitl14_gat_residual.h5", help='visual features')
parser.add_argument('--dir_audio', type=str, default='./data/audio_embedding.h5', help='audio features')
parser.add_argument('--dir_labels', type=str, default='./data/right_labels.h5', help='labels of AVE dataset')

parser.add_argument('--dir_order_train', type=str, default='./data/train_order.h5', help='indices of training samples')
parser.add_argument('--dir_order_val', type=str, default='./data/val_order.h5', help='indices of validation samples')
parser.add_argument('--dir_order_test', type=str, default='./data/test_order.h5', help='indices of testing samples')

parser.add_argument('--nb_epoch', type=int, default=300,  help='number of epoch')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--save_epoch', type=int, default=5, help='number of epoch for saving models')
parser.add_argument('--check_epoch', type=int, default=5, help='number of epoch for checking accuracy of current models during training')
parser.add_argument('--LAMBDA', type=float, default=1, help='weight for balancing losses')
parser.add_argument('--threshold', type=float, default=0.099, help='key-parameter for pruning process')
parser.add_argument('--clip_lambda', type=float, default=1, help='weight for CLIP similarity supervision')
parser.add_argument('--use_kd', action='store_true', default=True, help='enable knowledge distillation inside psp_net')
parser.add_argument('--kd_lambda', type=float, default=0.6, help='weight for KD loss')
parser.add_argument('--kd_T', type=float, default=4.0, help='temperature T used inside KD (used by model if use_kd)')


parser.add_argument('--trained_model_path', type=str, default=None, help='pretrained model')
parser.add_argument('--train', action='store_true', default=False, help='train a new model')

# -------------------- 固定随机种子 --------------------
FixSeed = 123
random.seed(FixSeed)
np.random.seed(FixSeed)
torch.manual_seed(FixSeed)
torch.cuda.manual_seed(FixSeed)



# -------------------- 训练函数（自动组合 KD + CE + AVPS + CLIP） --------------------
def train(args, net_model, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AVEData = AVE_Fully_Dataset(
    video_dir=args.dir_video,
    audio_dir=args.dir_audio,
    label_dir=args.dir_labels,
    order_dir=args.dir_order_train,
    batch_size=args.batch_size,
    status='train'  # <-- 会加载 train_sparse_dense_labels.npy
)

    nb_batch = AVEData.__len__() // args.batch_size
    print('nb_batch:', nb_batch)

    epoch_l, best_val_acc, best_test_acc, best_epoch = [], 0, 0, 0
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(args.nb_epoch):
        net_model.train()
        epoch_loss = epoch_loss_cls = epoch_loss_avps = epoch_loss_clip = epoch_loss_kd = 0.0
        n_samples = 0
        SHUFFLE_SAMPLES = True

        for i in range(nb_batch):
            audio_inputs, video_inputs, labels, segment_label_batch, segment_avps_gt_batch, sd_mask_batch = \
                AVEData.get_batch(i, SHUFFLE_SAMPLES)
            SHUFFLE_SAMPLES = False

            audio_inputs = audio_inputs.to(device)
            video_inputs = video_inputs.to(device)
            labels = labels.to(device)
            segment_label_batch = segment_label_batch.to(device)
            segment_avps_gt_batch = segment_avps_gt_batch.to(device)

            if isinstance(sd_mask_batch, torch.Tensor):
                sd_label_batch = sd_mask_batch.long().to(device)
            else:
                sd_label_batch = torch.from_numpy(np.asarray(sd_mask_batch)).long().to(device)

            optimizer._optimizer.zero_grad()

            # ---------- 前向 ---------- 
            outputs = net_model(audio_inputs, video_inputs, labels=segment_label_batch, sparse_dense_mask=sd_label_batch)

            fusion, out, cross_att, loss_kd, sim_va = outputs

            # ---------- CE Loss ----------
            loss_cls = ce_loss(out.permute(0, 2, 1), segment_label_batch)

            # ---------- AVPS Loss ----------
            cross_att_sigmoid = torch.sigmoid(cross_att)
            loss_avps = F.binary_cross_entropy(cross_att_sigmoid, segment_avps_gt_batch.float())

            # ---------- CLIP Loss ----------
#             # audio: (bs, T, 2048) → (bs, T, 256)
#             audio_hidden = F.relu(net_model.audio_proj_in(audio_inputs))
#             audio_hidden = net_model.ln_audio_in(audio_hidden)
            
#             # audio → CLIP space: (bs, T, 512)
#             audio_frames_clip = F.normalize(net_model.clip_similarity.proj_a(net_model.audio_to_clip(audio_hidden)),dim=-1)
#             # video → CLIP space: (bs, T, 512)
#             video_clip_space = F.normalize(net_model.clip_similarity.proj_v(video_inputs),dim=-1)

#             # similarity: (bs, T, T)
#             sim = torch.bmm(audio_frames_clip, video_clip_space.transpose(1, 2))
#             sim = sim.mean(dim=2) / 0.2
            y = (segment_avps_gt_batch > 0).float()
            loss_clip = F.binary_cross_entropy_with_logits(sim_va, y)

            # ---------- 自动合并 KD loss，前10 epoch不开KD ----------
            if epoch < 10:
                loss = loss_cls + args.LAMBDA * loss_avps + args.clip_lambda * loss_clip
            else:
                loss = loss_cls + args.LAMBDA * loss_avps + args.clip_lambda * loss_clip + loss_kd

            # ---------- 统计 ----------
            epoch_loss += loss.item() * audio_inputs.size(0)
            epoch_loss_cls += loss_cls.item() * audio_inputs.size(0)
            epoch_loss_avps += loss_avps.item() * audio_inputs.size(0)
            epoch_loss_clip += loss_clip.item() * audio_inputs.size(0)
            epoch_loss_kd += loss_kd.item() * audio_inputs.size(0)
            n_samples += audio_inputs.size(0)

            # ---------- 反向传播 ----------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), max_norm=5.0)
            optimizer.step_lr()

        SHUFFLE_SAMPLES = True
        epoch_l.append(epoch_loss / n_samples)

        acc = compute_acc(labels.cpu().numpy(), out.cpu().data.numpy(), labels.shape[0])
        print(
            "=== Epoch {%d} lr: {%.6f} | Loss: [{%.4f}] "
            "loss_cls: [{%.4f}] | loss_avps: [{%.4f}] | "
            "loss_clip: [{%.4f}] | loss_kd: [{%.4f}] | training_acc %.4f"
            % (
                epoch,
                optimizer._optimizer.param_groups[0]['lr'],
                epoch_loss / n_samples,
                epoch_loss_cls / n_samples,
                epoch_loss_avps / n_samples,
                epoch_loss_clip / n_samples,
                epoch_loss_kd / n_samples,
                acc,
            )
        )

        # ---------- Validation ----------
        if epoch % args.save_epoch == 0 and epoch != 0:
            val_acc = val(args, net_model)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                print('best val accuracy: {}'.format(best_val_acc))

        # ---------- Test ----------
        if epoch % args.check_epoch == 0 and epoch != 0:
            test_acc = test(args, net_model)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                print('best test accuracy: {}'.format(best_test_acc))
                torch.save(net_model.state_dict(), args.model_name + "_" + str(epoch) + "_fully.pt")

    print('[best val accuracy]: ', best_val_acc)
    print('[best test accuracy]: ', best_test_acc)


# -------------------- 验证函数 --------------------
def val(args, net_model):
    net_model.eval()
    AVEData = AVE_Fully_Dataset(
    video_dir=args.dir_video,
    audio_dir=args.dir_audio,
    label_dir=args.dir_labels,
    order_dir=args.dir_order_val,
    batch_size=args.batch_size,
    status='val'  # <-- 会加载 val_sparse_dense_labels.npy
)


    acc_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for i in range(AVEData.__len__() // args.batch_size):
            audio_inputs, video_inputs, labels, _, _, sd_mask_batch = AVEData.get_batch(i)

            audio_inputs = audio_inputs.to(device)
            video_inputs = video_inputs.to(device)
            labels = labels.to(device)

            if isinstance(sd_mask_batch, torch.Tensor):
                sd_label_batch = sd_mask_batch.long().to(device)
            else:
                sd_label_batch = torch.from_numpy(np.asarray(sd_mask_batch)).long().to(device)

            # ---------- 前向传播 ----------
            outputs = net_model(audio_inputs, video_inputs, sparse_dense_mask=sd_label_batch)

            fusion, out, cross_att, loss_kd, sim_va = outputs

            # ---------- 计算准确率 ----------
            acc_list.append(compute_acc(labels.cpu().numpy(), out.cpu().data.numpy(), labels.shape[0]))

    val_acc = np.mean(acc_list)
    print('[val] acc:', val_acc)
    return val_acc


# -------------------- 测试函数 --------------------
def test(args, net_model, model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 加载模型 ----------
    if model_path is not None:
        sd = torch.load(model_path)
        try:
            net_model.load_state_dict(sd)
            print(">>> [Testing] Load pretrained state_dict from " + model_path)
        except Exception:
            net_model = torch.load(model_path)
            print(">>> [Testing] Load pretrained full model from " + model_path)

    net_model.eval()

    AVEData = AVE_Fully_Dataset(
    video_dir=args.dir_video,
    audio_dir=args.dir_audio,
    label_dir=args.dir_labels,
    order_dir=args.dir_order_test,
    batch_size=args.batch_size,
    status='test'  # <-- 会加载 test_sparse_dense_labels.npy
)


    acc_list = []

    with torch.no_grad():
        for i in range(AVEData.__len__() // args.batch_size):
            audio_inputs, video_inputs, labels, _, _, sd_mask_batch = AVEData.get_batch(i)

            audio_inputs = audio_inputs.to(device)
            video_inputs = video_inputs.to(device)
            labels = labels.to(device)

            if isinstance(sd_mask_batch, torch.Tensor):
                sd_label_batch = sd_mask_batch.long().to(device)
            else:
                sd_label_batch = torch.from_numpy(np.asarray(sd_mask_batch)).long().to(device)

            # ---------- 前向传播 ----------
            outputs = net_model(audio_inputs, video_inputs, sparse_dense_mask=sd_label_batch)

            fusion, out, cross_att, loss_kd, sim_va = outputs

            # ---------- 计算准确率 ----------
            acc_list.append(compute_acc(labels.cpu().numpy(), out.cpu().data.numpy(), labels.shape[0]))

    test_acc = np.mean(acc_list)
    print('[test] acc:', test_acc)
    return test_acc


# -------------------- Main --------------------
if __name__ == "__main__":
    args = parser.parse_args()
    print("args: ", args)

    if args.model_name == "PSP":
        # pass use_kd flag and KD params to model so model internal forward can compute loss_kd when enabled
        net_model = psp_net(a_dim=2048, v_dim=512, category_num=29,
                            use_kd=args.use_kd, kd_lambda=args.kd_lambda, kd_T=args.kd_T)
    else:
        raise NotImplementedError

    net_model.cuda()
    base_optimizer = optim.Adam(net_model.parameters(), lr=1e-4)
    optimizer = ScheduledOptim(base_optimizer)

    if args.train:
        train(args, net_model, optimizer)
    else:
        test_acc = test(args, net_model, model_path=args.trained_model_path)
        print("[test] accuracy: ", test_acc)