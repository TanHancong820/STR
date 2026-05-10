import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchaudio
import sys
import os
import numpy as np

# --------------------
# 初始化辅助函数
# --------------------
def init_layers(layers):
    for layer in layers:
        if isinstance(layer, (list, tuple)):
            for l in layer:
                if hasattr(l, 'weight') and l.weight is not None:
                    nn.init.xavier_uniform_(l.weight)
                if hasattr(l, 'bias') and l.bias is not None:
                    l.bias.data.fill_(0.0)
        else:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.fill_(0.0)

# --------------------
# Sparse Audio Enhancement (unchanged)
# --------------------
def enhance_sparse_audio(audio_seq, min_period=2, max_period=8):
    audio_np = audio_seq.detach().cpu().numpy()
    T, D = audio_np.shape
    norm = np.linalg.norm(audio_np, axis=1)
    mask = (norm < 1e-6)
    if mask.sum() == 0:
        return audio_seq
    energy = norm
    if energy.max() > 0:
        energy = energy / (energy.max() + 1e-6)
    best_period = None
    best_corr = -1
    for p in range(min_period, min(max_period, max(1, T//2))):
        corr = np.dot(energy[:-p], energy[p:]) if p < T else 0.0
        if corr > best_corr:
            best_corr = corr
            best_period = p
    if best_period is not None and best_corr > 0.05:
        period = best_period
        audio_np_filled = audio_np.copy()
        for t in np.where(mask)[0]:
            if t - period >= 0:
                audio_np_filled[t] = audio_np[t - period]
            else:
                valid_idx = np.where(~mask)[0]
                if len(valid_idx) > 0:
                    audio_np_filled[t] = np.mean(audio_np[valid_idx], axis=0)
                else:
                    audio_np_filled[t] = 0
        return torch.tensor(audio_np_filled, device=audio_seq.device, dtype=audio_seq.dtype)
    audio_np_filled = audio_np.copy()
    idx = np.arange(T)
    valid_idx = np.where(~mask)[0]
    if len(valid_idx) == 0:
        return audio_seq * 0
    for d in range(D):
        audio_np_filled[mask, d] = np.interp(idx[mask], valid_idx, audio_np[valid_idx, d])
    return torch.tensor(audio_np_filled, device=audio_seq.device, dtype=audio_seq.dtype)


def enhance_sparse_audio_v2(audio_seq: torch.Tensor, min_period: int = 2, max_period: int = 8,
    energy_percentile: float = 85.0,
    max_replay_per_source: int = 2,
    replay_mix: float = 0.7
):
    """
    改进点：
    1. 关键帧 = 能量变化显著（而非是否为 0）
    2. 重放源帧有频次上限
    3. 使用软重放（mix）避免过拟合
    """

    device = audio_seq.device
    dtype = audio_seq.dtype

    audio_np = audio_seq.detach().cpu().numpy()
    T, D = audio_np.shape

    # ---------- 1. 能量与变化量 ----------
    energy = np.linalg.norm(audio_np, axis=1)
    if energy.max() > 0:
        energy = energy / (energy.max() + 1e-6)

    delta_energy = np.abs(energy - np.roll(energy, 1)) #delta_energy 是形状为 [T,] 的数组，表示每个帧相对于前一帧的能量变化幅度。
    delta_energy[0] = 0.0

    # 关键帧 = 能量突变点
    threshold = np.percentile(delta_energy, energy_percentile) #计算能量变化量的「分位数阈值」，即 85% 的帧的能量变化量都低于这个值，剩下 15% 是突变帧。
    key_mask = delta_energy > threshold

    # 若没有明显关键帧，直接返回
    if key_mask.sum() == 0:
        return audio_seq

    # ---------- 2. 周期搜索（限制范围） ----------
    best_period = None
    best_corr = -1.0

    max_p = min(max_period, T // 2)
    for p in range(min_period, max_p + 1):
        corr = np.dot(energy[:-p], energy[p:])
        if corr > best_corr:
            best_corr = corr
            best_period = p

    # 周期不可靠则不重放
    if best_period is None or best_corr < 0.05:
        return audio_seq

    # ---------- 3. 可控重放 ----------
    filled = audio_np.copy()
    replay_counter = np.zeros(T, dtype=np.int32)

    for t in np.where(key_mask)[0]: #遍历每个关键帧 t
        src = t - best_period #计算关键帧 t 对应的「源帧」（即最佳周期前的那个帧）
        if src < 0:
            continue

        # 限制同一源帧被重放的次数
        if replay_counter[src] >= max_replay_per_source:
            continue

        filled[t] = (
            replay_mix * audio_np[src]
            + (1.0 - replay_mix) * audio_np[t]
        )
        replay_counter[src] += 1

    return torch.tensor(filled, device=device, dtype=dtype)



# --------------------
# 简单 Self-Attention
# --------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super(SelfAttention, self).__init__()
        self.phi = nn.Linear(dim, hidden_dim)
        self.theta = nn.Linear(dim, hidden_dim)
        self.g = nn.Linear(dim, hidden_dim)
        init_layers([self.phi, self.theta, self.g])

    def forward(self, x):
        bs, T, dim = x.shape
        phi = self.phi(x)
        theta = self.theta(x)
        g = self.g(x)
        att = torch.bmm(phi, theta.permute(0,2,1)) / (dim ** 0.5)
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, g)
        return out, att


# --------------------
# AVGA Temporal
# --------------------
class AVGA(nn.Module):
    """Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """
    def __init__(self, a_dim=256, v_dim=256, hidden_size=256, map_size=1):
        super(AVGA, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(a_dim, hidden_size)
        self.affine_video = nn.Linear(v_dim, hidden_size)
        self.affine_v = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_h = nn.Linear(map_size, 1, bias=False)

        init.xavier_uniform_(self.affine_v.weight)
        init.xavier_uniform_(self.affine_g.weight)
        init.xavier_uniform_(self.affine_h.weight)
        init.xavier_uniform_(self.affine_audio.weight)
        init.xavier_uniform_(self.affine_video.weight)

    def forward(self, audio, video):
        # audio: [bs, 10, 256]
        # video: [bs, 10, 1, 1，256]
        V_DIM = video.size(-1)
        v_t = video.view(video.size(0) * video.size(1), -1, V_DIM) # [bs*10, 1, 256]
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t)) # [bs*10, 1, 256]
        a_t = audio.view(-1, audio.size(-1)) # [bs*10, 256]
        a_t = self.relu(self.affine_audio(a_t)) # [bs*10, 256]
        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2) # [bs*10, 1, 1] + [bs*10, 1, 1]

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2) # [bs*10, 1]
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map, [bs*10, 1, 1]
        c_t = torch.bmm(alpha_t, V).view(-1, V_DIM) # [bs*10, 1, 256]注意力权重对原始视频特征 V 加权求和
        video_t = c_t.view(video.size(0), -1, V_DIM) # attended visual features, [bs, 10, 256]
        return video_t


# --------------------
# Bi-LSTM wrapper
# --------------------
class LSTM_A_V(nn.Module):
    def __init__(self, a_dim, v_dim, hidden_dim=256, num_layers=1):
        super(LSTM_A_V, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_audio = nn.LSTM(a_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.lstm_video = nn.LSTM(v_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size, device):
        num_dirs = 2
        h0_a = torch.zeros(self.num_layers*num_dirs, batch_size, self.hidden_dim, device=device)
        c0_a = torch.zeros(self.num_layers*num_dirs, batch_size, self.hidden_dim, device=device)
        h0_v = torch.zeros(self.num_layers*num_dirs, batch_size, self.hidden_dim, device=device)
        c0_v = torch.zeros(self.num_layers*num_dirs, batch_size, self.hidden_dim, device=device)
        return (h0_a, c0_a), (h0_v, c0_v)

    def forward(self, a_fea, v_fea):
        bs = a_fea.size(0)
        device = a_fea.device
        hidden_a, hidden_v = self.init_hidden(bs, device)
        self.lstm_audio.flatten_parameters()
        self.lstm_video.flatten_parameters()
        lstm_audio, _ = self.lstm_audio(a_fea, hidden_a)
        lstm_video, _ = self.lstm_video(v_fea, hidden_v)
        return lstm_audio, lstm_video


# --------------------
# CLIPSimilarity (compat)
# --------------------
class CLIPSimilarity(nn.Module):
    def __init__(self, v_dim=512, a_dim=512):
        super(CLIPSimilarity, self).__init__()
        self.proj_v = nn.Linear(v_dim, 512, bias=False)
        self.proj_a = nn.Linear(a_dim, 512, bias=False)
        init_layers([self.proj_v, self.proj_a])

    def forward(self, audio_global, video_frames):
        pa = F.normalize(self.proj_a(audio_global), dim=-1)
        pv = F.normalize(self.proj_v(video_frames), dim=-1)
        sim = torch.sum(pa.unsqueeze(1) * pv, dim=-1)
        return sim


# --------------------
# Sparse/Dense classifier
# --------------------
# class SparseDenseClassifier(nn.Module):
#     def __init__(self, a_dim=512, v_dim=512, hidden=256):
#         super(SparseDenseClassifier, self).__init__()
#         self.fc1 = nn.Linear(a_dim + v_dim, hidden)
#         self.fc2 = nn.Linear(hidden, 2)
#         init_layers([self.fc1, self.fc2])

#     def forward(self, audio_global, video_global):
#         x = torch.cat([audio_global, video_global], dim=-1)
#         x = F.relu(self.fc1(x))
#         logits = self.fc2(x)
#         return logits


# --------------------
# 分类头
# --------------------
class Classify(nn.Module):
    def __init__(self, hidden_dim=256, category_num=29):
        super(Classify, self).__init__()
        self.L1 = nn.Linear(hidden_dim, 64)
        self.L2 = nn.Linear(64, category_num)
        init_layers([self.L1, self.L2])

    def forward(self, feature):
        out = F.relu(self.L1(feature))
        out = self.L2(out)
        return out

# --------------------
# CrossAttention (lightweight)
# --------------------
class CrossAttention(nn.Module):
    """
    兼容旧版 PyTorch 的 Cross-Transformer（无 batch_first）
    输入 shape: (B, T, D)
    输出 shape: (B, T, D)
    """
    def __init__(self, embed_dim=512, n_heads=16, n_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'v_self_attn': nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout),
                'a_self_attn': nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout),
                'cross_attn_v2a': nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout),
                'cross_attn_a2v': nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout),
                'norm_v1': nn.LayerNorm(embed_dim),
                'norm_v2': nn.LayerNorm(embed_dim),
                'norm_a1': nn.LayerNorm(embed_dim),
                'norm_a2': nn.LayerNorm(embed_dim),
            })
            self.layers.append(layer)

    def forward(self, v, a):
        """
        v, a: (B, T, D)
        """
        B, T, D = v.shape

        # 转换为 MultiheadAttention 接收的格式: (T, B, D)
        v_t = v.transpose(0, 1)
        a_t = a.transpose(0, 1)

        for layer in self.layers:
            # --- Video Self-Attention ---
            v_self, _ = layer['v_self_attn'](v_t, v_t, v_t)
            v_t = v_t + v_self
            v_t = layer['norm_v1'](v_t)

            # --- Audio Self-Attention ---
            a_self, _ = layer['a_self_attn'](a_t, a_t, a_t)
            a_t = a_t + a_self
            a_t = layer['norm_a1'](a_t)

            # --- Video-to-Audio Cross Attention ---
            a_cross, _ = layer['cross_attn_v2a'](a_t, v_t, v_t)
            a_t = a_t + a_cross
            a_t = layer['norm_a2'](a_t)

            # --- Audio-to-Video Cross Attention ---
            v_cross, _ = layer['cross_attn_a2v'](v_t, a_t, a_t)
            v_t = v_t + v_cross
            v_t = layer['norm_v2'](v_t)

        # 转回 (B, T, D)
        v_out = v_t.transpose(0, 1)
        a_out = a_t.transpose(0, 1)
        return v_out, a_out


# --------------------
# 主模型 psp_net (第二阶段版本: 保持原逻辑, 增加 AudioAttentionPool + TemporalCrossTransformer)
# --------------------
class psp_net(nn.Module):
    def __init__(self, a_dim=2048, v_dim=512, hidden_dim=256, category_num=29,
                 device=None, use_kd=True, kd_lambda=0.6, kd_T=4):
        super(psp_net, self).__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.category_num = category_num
        self.hidden_dim = hidden_dim
        self.use_kd = use_kd
        self.kd_lambda = kd_lambda
        self.kd_T = kd_T

        # ---------------- projections ----------------
        self.audio_proj_in = nn.Linear(a_dim, hidden_dim)
        self.video_proj_in = nn.Sequential(
            nn.Linear(v_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.audio_proj_perseg = nn.Linear(a_dim, hidden_dim)
        self.audio_proj_global = nn.Linear(a_dim, hidden_dim)
        self.video_proj = nn.Linear(v_dim, hidden_dim)

        self.audio_proj = nn.Linear(a_dim, 512)
        self.audio_to_clip = nn.Linear(hidden_dim, 512)
        # 定义全局可学习 mix
        # self.replay_mix = nn.Parameter(torch.tensor(0.7))

        # AVGA (map_size=1 for per-frame vector)
        self.avga = AVGA(a_dim=hidden_dim, v_dim=hidden_dim, hidden_size=hidden_dim, map_size=1)

        # LSTM wrapper
        self.lstm_a_v = LSTM_A_V(a_dim=hidden_dim, v_dim=hidden_dim, hidden_dim=hidden_dim)

        # CLIP sim and sparse/dense classifier
        self.clip_similarity = CLIPSimilarity(v_dim=512, a_dim=512)

        self.video_adapter_for_sd = nn.Linear(v_dim, 512)
        self.video_frame_adapter = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.ln_audio_in = nn.LayerNorm(hidden_dim)
        self.ln_video_in = nn.LayerNorm(hidden_dim)
        self.ln_lstm_out = nn.LayerNorm(hidden_dim * 2)

        self.alpha_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Sigmoid activation to make sure alpha is between 0 and 1
        )

        fused_dim = hidden_dim * 2
        self.L1 = nn.Linear(fused_dim, 64, bias=False)
        self.L2 = nn.Linear(64, category_num, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avps_fc = nn.Linear(fused_dim, 1)

        # Student head for KD
        self.video_cls_head = nn.Linear(hidden_dim * 2, category_num)

        # 可学习缩放参数（PSP 残差缩放）
        # self.alpha_v = nn.Parameter(torch.tensor(0.5))
        # self.alpha_a = nn.Parameter(torch.tensor(0.5))

        # Temporal Cross-Transformer
        self.cross_attention = CrossAttention(embed_dim=hidden_dim*2, n_heads=16, n_layers=3, dropout=0.1)

        # init layers
        init_layers([self.audio_proj_in, *[m for m in self.video_proj_in if isinstance(m, nn.Linear)],
                     self.audio_proj_perseg, self.audio_proj_global, self.video_proj,
                     self.audio_proj, self.audio_to_clip,
                     *[m for m in self.video_frame_adapter if isinstance(m, nn.Linear)],
                     self.video_adapter_for_sd,
                     self.L1, self.L2,
                     self.video_cls_head])

        self.to(self.device)

    def forward(self, audio, video, labels=None, sparse_dense_mask=None):
        device = self.device
        audio = audio.to(device)
        video = video.to(device)

        bs = audio.shape[0]
        T = audio.shape[1]  # As the time dimension is fixed at 10, no need for conditional check

        # ---------- audio reshape ----------
        #audio_global_raw = audio.mean(dim=1)  # Use mean of all time steps for global audio features
        audio_perseg_raw = audio  # No need for condition since it's (bs, 10, 2048)

        # ---------- global projections ----------
        #audio_global_proj = F.normalize(self.audio_proj(audio_global_raw), dim=-1)
        #video_global_raw = video.mean(dim=1)  # Use mean of all time steps for global video features

        audio_perseg_in = F.relu(self.audio_proj_in(audio_perseg_raw))

        bs_v, T_v, v_dim = video.shape
        video_flat = video.reshape(-1, v_dim)
        video_proj_flat = self.video_proj_in(video_flat)
        video_in = video_proj_flat.reshape(bs_v, T_v, -1)

        video_for_lstm = torch.zeros_like(video_in)

        audio_perseg_in = self.ln_audio_in(audio_perseg_in)
        video_in = self.ln_video_in(video_in)

        # ---------- DENSE branch (use AVGA) ----------
        use_avga_mask = (sparse_dense_mask != 0)
        dense_idx = torch.nonzero(use_avga_mask, as_tuple=False).squeeze(1).to(device=device, dtype=torch.long)

        if dense_idx.numel() > 0:
            audio_dense_seq = audio_perseg_in.index_select(0, dense_idx)
            video_dense_in = video_in.index_select(0, dense_idx)
            video_dense_maps = video_dense_in.view(video_dense_in.size(0), video_dense_in.size(1), 1, 1, video_dense_in.size(2))
            video_dense_reweighted = self.avga(audio_dense_seq, video_dense_maps)
            idx_exp = dense_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, T, video_dense_reweighted.size(-1))
            video_for_lstm = video_for_lstm.scatter(0, idx_exp, F.relu(video_dense_reweighted))

        # ---------- SPARSE branch ----------
        sparse_idx = torch.nonzero(~use_avga_mask, as_tuple=False).squeeze(1).to(device=device, dtype=torch.long)
        
        if sparse_idx.numel() > 0:
            new_audio_slices = []
            for j in range(sparse_idx.numel()):
                idx = sparse_idx[j].item()
                sample_audio = audio_perseg_in[idx]
                feat_mean = sample_audio.mean(dim=0)
                alpha_raw = self.alpha_predictor(feat_mean)
                alpha = alpha_raw  # Modify alpha using Sigmoid to ensure it stays in [0, 1]

                # forward 时取值（保证 0~1）
                # mix = torch.sigmoid(self.replay_mix).item()

                # 调用增强函数
                enhanced = enhance_sparse_audio(sample_audio)
                if enhanced.device != device:
                    enhanced = enhanced.to(device)

                # new_slice = alpha.view(1, 1) * enhanced + (1.0 - alpha.view(1, 1)) * sample_audio
                new_slice = alpha * enhanced + (1.0 - alpha) * sample_audio
                new_audio_slices.append(new_slice.unsqueeze(0))

            new_audio_slices = torch.cat(new_audio_slices, dim=0)
            idx_exp = sparse_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, T, self.hidden_dim)
            audio_perseg_in = audio_perseg_in.scatter(0, idx_exp, new_audio_slices)

            video_sparse = video_in.index_select(0, sparse_idx)
            idx_exp_v = sparse_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, T, video_sparse.size(-1))
            video_for_lstm = video_for_lstm.scatter(0, idx_exp_v, video_sparse)


        # ---------- LSTM Fusion ----------
        lstm_audio, lstm_video = self.lstm_a_v(audio_perseg_in, video_for_lstm)

        lstm_audio = self.ln_lstm_out(lstm_audio)
        lstm_video = self.ln_lstm_out(lstm_video)

        # ---------- CrossAttention ----------
        cross_video, cross_audio = self.cross_attention(lstm_video, lstm_audio)
        lstm_audio_enh = lstm_audio + cross_audio
        lstm_video_enh = lstm_video + cross_video

        lstm_audio = lstm_audio_enh
        lstm_video = lstm_video_enh

        # ---------- CLIP similarity ----------
        audio_clip_global = audio_perseg_in.mean(dim=1)          # (bs, hidden_dim)
        audio_clip_global = F.normalize(self.audio_to_clip(audio_clip_global), dim=-1)
        
        video_clip_frames = F.normalize(self.video_frame_adapter(video_for_lstm), dim=-1)

        sim_va = self.clip_similarity(audio_clip_global, video_clip_frames)
        sim_att_va = F.softmax(sim_va, dim=-1)
        sim_combined = sim_att_va

        V_prime = lstm_video * sim_combined.unsqueeze(-1)
        A_prime = lstm_audio * sim_combined.unsqueeze(-1)

        # V_psp = lstm_video + self.alpha_v * V_prime
        # A_psp = lstm_audio + self.alpha_a * A_prime
        fused_per_seg = 0.5 * (V_prime + A_prime)

        # ---------- Fusion head ----------
        fusion = fused_per_seg  # 直接作为 fusion 表示

        out = self.relu(self.L1(fusion))   # [bs, T, 64]
        out = self.L2(out)                 # [bs, T, category_num]
        cross_att = self.avps_fc(fused_per_seg).squeeze(-1)

        # ---------- KD ----------
        if self.use_kd:
            feat_dim = V_prime.size(-1)
            V_flat = V_prime.reshape(-1, feat_dim)
            student_logits = self.video_cls_head(V_flat)

            if labels is None:
                loss_kd = torch.tensor(0.0, device=device)
            else:
                labels = labels.to(device)
                labels_flat = labels.reshape(-1).long()
                loss_kd = F.cross_entropy(student_logits, labels_flat)
                loss_kd = loss_kd * self.kd_lambda
        else:
            loss_kd = torch.tensor(0.0, device=device)

        return fusion, out, cross_att, loss_kd, sim_va
        # return fusion, out, cross_att, loss_kd, sim_va, A_prime, V_prime





