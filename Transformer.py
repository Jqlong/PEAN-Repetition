import copy

import numpy as np
import torch
from torch.nn import Softmax
import torch.nn.functional as F

'''Transformer模型的实现'''

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.dim_model = config.embedding_size
        self.hidden = 1024
        self.last_hidden = 512

        self.num_head = config.trf_heads
        self.num_encoder = config.trf_layers

        # 位置编码
        self.position_embedding = Positional_Encoding(self.dim_model, config.pad_num, config.dropout, config.device)
        # 堆叠多个编码器
        self.encoder = Encoder(self.dim_model, self.num_head, self.hidden, config.dropout)
        self.encoders = torch.nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)])
        self.tanh = torch.nn.Tanh()  # 激活函数

    def forward(self, x):
        # print('x的维度', x.shape)  # x的维度 torch.Size([1, 400])
        out = self.position_embedding(x)
        for encoder in self.encoders:
            out, alpha = encoder(out)

        out = out.view(out.size(0), -1)
        return out


class Positional_Encoding(torch.nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        # embed：嵌入维度   pad_size：输入序列的最大长度，字节数
        super(Positional_Encoding, self).__init__()
        self.device = device
        # 生成位置编码矩阵 pe，其形状为 [pad_size, embed]
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # # 偶数位置的编码使用 sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数位置的编码使用 cos
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = x + torch.nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(torch.nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # K.permute(0, 2, 1) 将键向量 K 的最后两个维度进行转置
        if scale:
            attention = attention * scale

        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context, attention


class Multi_Head_Attention(torch.nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head  # 每个头的维度
        self.fc_Q = torch.nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = torch.nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = torch.nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()  # 计算注意力
        self.fc = torch.nn.Linear(num_head * self.dim_head, dim_model)  # 多头注意力输出后的全连接层，将多头的输出重新映射回 dim_model。
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_model)  # 对输出进行层归一化，使得每层的输出具有相同的分布，帮助梯度更稳定地传播。

    def forward(self, x):
        batch_size = x.size(0)  # 形状为 [batch_size, seq_len, dim_model]
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, - 1, self.dim_head)  # [batch_size * num_head, seq_len, dim_head]
        K = K.view(batch_size * self.num_head, - 1, self.dim_head)
        V = V.view(batch_size * self.num_head, - 1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context, alpha = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差链接
        out = self.layer_norm(out)
        return out, alpha


class Position_wise_Feed_Forward(torch.nn.Module):
    # 前馈网络
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = torch.nn.Linear(dim_model, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        # dim_model:序列嵌入的维度
        # num_head:注意力头的数量
        # hidden：前向神经网络中隐藏层大小
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        # 前向神经网络，用于对每个位置的表示进行非线性变换。
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        # x：输入数据，形状为 [batch_size, seq_len, dim_model]
        out, alpha = self.attention(x)
        out = self.feed_forward(out)
        return out, alpha
