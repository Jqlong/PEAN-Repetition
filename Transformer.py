import copy

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


class Positional_Encoding(torch.nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])


class Encoder(torch.nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()