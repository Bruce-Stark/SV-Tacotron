from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from torch import nn
import numpy as np
import torch


# 梅尔过滤器组
mel_window_length = 25  # 单位：毫秒
mel_window_step = 10    # 单位：毫秒
mel_n_channels = 40
# 采样率
sampling_rate = 16000
# 音频音量标准化
audio_norm_target_dBFS = -30
# 话语中的谱图帧数
partials_n_frames = 160     # 1600 ms
inference_n_frames = 80     # 800 ms
# 静音检测算法（VAD）
vad_window_length = 30  # 单位：毫秒
vad_moving_average_width = 8
vad_max_silence_length = 6
# 模型参数
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3
# 训练参数
learning_rate_init = 1e-4
speakers_per_batch = 64
utterances_per_speaker = 10


class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        
        # 网络结构
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        # 相似度矩阵中的权重和偏移量
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # 损失
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self):
        # 梯度标识
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
        # 梯度裁剪（防止梯度爆炸）
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    # 一批话语的频谱嵌入
    def forward(self, utterances, hidden_init=None):
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        return embeds

    # 相似度矩阵
    def similarity_matrix(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # 论文中的e向量
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # 论文中的c向量
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # 相似度矩阵
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        # 参考论文中s向量的计算公式
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    # Loss和EER
    def loss(self, embeds):

        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # roc曲线的fpr和tpr
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer
