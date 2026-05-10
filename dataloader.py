"""AVE dataset"""
import numpy as np
import torch
import h5py
import pickle
import random
from itertools import product
import os
import pdb
import torch.utils.data as data
import logging
logger = logging.getLogger('MMAction')

ave_dataset = ['bell', 'Male', 'Bark', 'aircraft', 'car', 'Female', 'Helicopter',
    'Violin', 'Flute', 'Ukulele', 'Fry food', 'Truck', 'Shofar', 'Motorcycle',
    'guitar', 'Train', 'Clock', 'Banjo', 'Goat', 'Baby', 'Bus',
    'Chainsaw', 'Cat', 'Horse', 'Toilet', 'Rodents', 'Accordion', 'Mandolin', 'background']
STANDARD_AVE_DATASET = ['Church bell', 'Male speech, man speaking', 'Bark', 'Fixed-wing aircraft, airplane', 'Race car, auto racing', \
                    'Female speech, woman speaking', 'Helicopter', 'Violin, fiddle', 'Flute', 'Ukulele', 'Frying (food)', 'Truck', 'Shofar', \
                    'Motorcycle', 'Acoustic guitar', 'Train horn', 'Clock', 'Banjo', 'Goat', 'Baby cry, infant cry', 'Bus', 'Chainsaw',\
                    'Cat', 'Horse', 'Toilet flush', 'Rodents, rats, mice', 'Accordion', 'Mandolin']


class AVE_Fully_Dataset(data.Dataset):
    """Data preparation for fully supervised PSP with VGGish-Audio + CLIP-Video"""

    def __init__(self, video_dir, audio_dir, label_dir, order_dir, batch_size, status):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.status = status

        # ====== Load Features ======
        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]   # (N,10,128)
        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]           # (N,10,29)
        with h5py.File(video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]   # (N,10,512)

        # ====== Load sample order (global index) ======
        with h5py.File(order_dir, 'r') as hf:
            order = hf['order'][:]
        self.lis = order.tolist()
        self.list_copy = self.lis.copy()

        # ====== Load sparse / dense labels (GLOBAL INDEX → 0/1) ======
        # ★ MOD: load 2D npy and convert to dict
        if status == 'train':
            sd_array = np.load("./data/train_sparse_dense_labels.npy")
        elif status == 'val':
            sd_array = np.load("./data/val_sparse_dense_labels.npy")
        else:  # test
            sd_array = np.load("./data/test_sparse_dense_labels.npy")

        # sd_array shape: (K, 2) -> [sid, label]
        self.sd_label_dict = {
            int(sid): int(label)
            for sid, label in sd_array
        }

        # ====== Allocate batch memory ======
        self.video_batch = np.zeros([batch_size, 10, 512], dtype=np.float32)
        self.audio_batch = np.zeros([batch_size, 10, 2048], dtype=np.float32)
        self.label_batch = np.zeros([batch_size, 10, 29], dtype=np.float32)

        # teacher hard label for KD (bs,10)
        self.segment_label_batch = np.zeros([batch_size, 10], dtype=np.int64)

        # AVPS loss ground truth
        self.segment_avps_gt_batch = np.zeros([batch_size, 10], dtype=np.float32)

        # sparse / dense label (bs,)
        self.sd_label_batch = np.zeros(batch_size, dtype=np.int64)

    def __len__(self):
        return len(self.lis)

    # -------------------------------------------------
    # Original AVPS GT computation (UNCHANGED)
    # -------------------------------------------------
    def get_segment_wise_relation(self, batch_labels):
        # batch_labels: [bs, 10, 29]
        bs, seg_num, category_num = batch_labels.shape
        all_seg_idx = list(range(seg_num))
        for i in range(bs):
            col_sum = np.sum(batch_labels[i].T, axis=1)
            category_bg_cols = col_sum.nonzero()[0].tolist()
            category_bg_cols.sort() # [category_label_idx, 28(background_idx, optional)]

            category_col_idx = category_bg_cols[0]
            category_col = batch_labels[i, :, category_col_idx]
            same_category_row_idx = category_col.nonzero()[0].tolist()
            if len(same_category_row_idx) != 0:
                self.segment_avps_gt_batch[i, same_category_row_idx] = 1 / (len(same_category_row_idx))

        for i in range(bs):
            row_idx, col_idx = np.where(batch_labels[i] == 1)
            self.segment_label_batch[i, row_idx] = col_idx


    # -------------------------------------------------
    # Get batch (KEEP STRUCTURE UNCHANGED)
    # -------------------------------------------------
    def get_batch(self, idx, shuffle_samples=False):
        if shuffle_samples:
            random.shuffle(self.list_copy)

        select_ids = self.list_copy[
            idx * self.batch_size: (idx + 1) * self.batch_size
        ]

        for i in range(self.batch_size):
            sid = select_ids[i]  # GLOBAL INDEX

            self.video_batch[i] = self.video_features[sid]
            self.audio_batch[i] = self.audio_features[sid]
            self.label_batch[i] = self.labels[sid]

            # ★ MOD: lookup sparse / dense label by global index
            if sid in self.sd_label_dict:
                self.sd_label_batch[i] = self.sd_label_dict[sid]
            else:
                # safety fallback (default dense)
                self.sd_label_batch[i] = 1

            # KD teacher hard label
            self.segment_label_batch[i] = self.labels[sid].argmax(-1)

        # AVPS GT
        self.get_segment_wise_relation(self.label_batch)

        return (
            torch.from_numpy(self.audio_batch).float(),
            torch.from_numpy(self.video_batch).float(),
            torch.from_numpy(self.label_batch).float(),
            torch.from_numpy(self.segment_label_batch).long(),
            torch.from_numpy(self.segment_avps_gt_batch).float(),
            torch.from_numpy(self.sd_label_batch).long()
        )

    
class AVE_Weakly_Dataset(object):
    """Weakly supervised AVE Dataset with sparse/dense labels (for normal samples only)."""

    def __init__(self, video_dir, video_dir_bg, audio_dir, audio_dir_bg,
                 label_dir, prob_label_dir, label_dir_bg, label_dir_gt,
                 order_dir, batch_size, status='train'):
        
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.video_dir_bg = video_dir_bg
        self.audio_dir_bg = audio_dir_bg
        self.status = status
        self.batch_size = batch_size
        
        # ---------- 与官方相同的样本ID管理 ----------
        with h5py.File(order_dir, 'r') as hf:
            train_l = hf['order'][:]  # 与官方相同的变量名
        self.lis = train_l
        self.list_copy = self.lis.copy().copy().tolist()  # 与官方相同的副本机制

        # ---------- 加载正样本特征 ----------
        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]     # (All,10,2048)
        with h5py.File(video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]     # (All,10,512)
        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]             # (All,29)
        with h5py.File(prob_label_dir, 'r') as hf:
            self.prob_labels = hf['avadataset'][:]        # (All,29)

        # 与官方相同的索引选择
        self.video_features = self.video_features[train_l, :, :]
        self.audio_features = self.audio_features[train_l, :, :]
        self.labels = self.labels[train_l, :]
        self.prob_labels = self.prob_labels[train_l, :]

        print('video_features.shape', self.video_features.shape)
        print('audio_features.shape', self.audio_features.shape)

       # ---------- 加载负样本（背景噪音） ----------
        if status == "train":

            with h5py.File(label_dir_bg, 'r') as hf:
                self.negative_labels = hf['avadataset'][:]   # (ng_num,29)

            with h5py.File(audio_dir_bg, 'r') as hf:
                self.negative_audio_features = hf['avadataset'][:]   # (ng_num,10,2048)

            with h5py.File(video_dir_bg, 'r') as hf:
                self.negative_video_features = hf['avadataset'][:]   # (ng_num,10,512)

            ng_num = self.negative_audio_features.shape[0]

            # ------------------------------------------------
            # ★ 特征已经对齐，不需要pad和pool
            # ------------------------------------------------
            neg_audio_pad = self.negative_audio_features
            neg_video_pooled = self.negative_video_features

            # 与官方相同的拼接逻辑
            size = self.audio_features.shape[0] + ng_num

            # ---------- 音频拼接 ----------
            audio_train_new = np.zeros((size, 10, 2048), dtype=np.float32)
            audio_train_new[0:self.audio_features.shape[0], :, :] = self.audio_features
            audio_train_new[self.audio_features.shape[0]:size, :, :] = neg_audio_pad
            self.audio_features = audio_train_new

            # ---------- 视频拼接 ----------
            video_train_new = np.zeros((size, 10, 512), dtype=np.float32)
            video_train_new[0:self.video_features.shape[0], :, :] = self.video_features
            video_train_new[self.video_features.shape[0]:size, :, :] = neg_video_pooled
            self.video_features = video_train_new

            # ---------- 标签拼接 ----------
            y_train_new = np.zeros((size, 29), dtype=np.float32)
            y_train_new[0:self.labels.shape[0], :] = self.labels
            y_train_new[self.labels.shape[0]:size, :] = self.negative_labels
            self.labels = y_train_new

            # ---------- 概率标签 ----------
            prob_y_train_new = np.zeros((size, 29), dtype=np.float32)
            prob_y_train_new[0:self.prob_labels.shape[0], :] = self.prob_labels
            prob_y_train_new[self.prob_labels.shape[0]:size, :] = self.negative_labels
            self.prob_labels = prob_y_train_new

            # ---------- 样本ID扩展 ----------
            self.list_copy.extend(list(range(8000, 8000+ng_num, 1)))

            print(f"训练数据: 正样本 {len(train_l)} 个, 负样本 {ng_num} 个")
            print(f"音频特征形状: {self.audio_features.shape}")
            print(f"视频特征形状: {self.video_features.shape}")
            print(f"标签形状: {self.labels.shape}")

        else:
            # 验证/测试集
            with h5py.File(label_dir_gt, 'r') as hf:
                self.labels = hf['avadataset'][:]
                self.labels = self.labels[train_l, :, :]
            print(f"{status}数据: {len(train_l)} 个样本")

        # ---------- 加载稀疏/密集标签（仅正常样本） ----------
        if status == 'train':
            sd_array = np.load('./data/train_sparse_dense_labels.npy')
        elif status == 'val':
            sd_array = np.load('./data/val_sparse_dense_labels.npy')
        else:
            sd_array = np.load('./data/test_sparse_dense_labels.npy')

        self.sd_label_dict = {int(row[0]): int(row[1]) for row in sd_array}

        # ---------- batch buffers ----------
        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 512]))  # 3D视频特征
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 2048]))  # 2048维音频
        
        # 新增：稀疏/密集标签batch
        self.sd_label_batch = np.zeros(batch_size, dtype=np.int64)
        # 新增：CLIP 用的 AV 对齐标签
        # self.av_match_label_batch = np.zeros(batch_size, dtype=np.int64)
        
        if status == "train":
            self.label_batch = np.float32(np.zeros([self.batch_size, 29]))
            self.prob_label_batch = np.float32(np.zeros([self.batch_size, 29]))
        else:
            self.label_batch = np.float32(np.zeros([self.batch_size, 10, 29]))

    def __len__(self):
        return len(self.labels)  # 与官方相同，返回标签数量

    def get_batch(self, idx, shuffle_samples=False):
        # 与官方相同的索引查找逻辑
        self.list_copy_copy = self.list_copy.copy().copy()  # 固定参考列表
        
        if shuffle_samples:
            random.shuffle(self.list_copy)
        
        select_ids = self.list_copy[idx * self.batch_size : (idx + 1) * self.batch_size]

        for i in range(self.batch_size):
            id = select_ids[i]
            real_id = self.list_copy_copy.index(id)  # 与官方相同的查找方式
            
            self.video_batch[i, :, :] = self.video_features[real_id, :, :]  # 3D视频
            self.audio_batch[i, :, :] = self.audio_features[real_id, :, :]  # 2048维音频
            
            if self.status == "train":
                self.label_batch[i, :] = self.labels[real_id, :]
                self.prob_label_batch[i, :] = self.prob_labels[real_id, :]
            else:
                self.label_batch[i, :, :] = self.labels[real_id, :, :]
            
            # ★ 新增：稀疏/密集标签分配
            if id >= 8000:  # 负样本
                self.sd_label_batch[i] = 1  # 背景噪音固定为dense
            else:  # 正常样本
                self.sd_label_batch[i] = self.sd_label_dict.get(id, 1)  # 默认dense
                
            # # -------- CLIP AV match --------
            # # 0 = aligned, 1 = mismatched
            # self.av_match_label_batch[i] = 1 if id >= 8000 else 0

        if self.status == 'train':
            return (
                torch.from_numpy(self.audio_batch).float(),
                torch.from_numpy(self.video_batch).float(),
                torch.from_numpy(self.label_batch).float(),
                torch.from_numpy(self.prob_label_batch).float(),
                torch.from_numpy(self.sd_label_batch).long()  # ★ 新增返回值
            )
        else:
            return (
                torch.from_numpy(self.audio_batch).float(),
                torch.from_numpy(self.video_batch).float(),
                torch.from_numpy(self.label_batch).float(),
                torch.from_numpy(self.sd_label_batch).long()  # ★ 新增返回值
            )