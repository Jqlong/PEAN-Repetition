import copy

import numpy as np
import torch


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
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # # 偶数位置的编码使用 sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数位置的编码使用 cos
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = x + torch.nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Multi_Head_Attention:
    def __init__(self, dim_model, num_head, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0


class Position_wise_Feed_Forward:
    pass


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
