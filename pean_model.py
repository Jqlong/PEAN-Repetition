import torch
import Transformer as TRF

'''PEAN模型的实现'''


# 构建模型
class PEAN(torch.nn.Module):
    def __init__(self, config):
        super(PEAN, self).__init__()
        self.config = config
        self.emb_size = config.embedding_size

        #
        self.length_embedding = torch.nn.Embedding(num_embeddings=2000, embedding_dim=config.length_emb_size, padding_idx=0)  # 嵌入层
        self.lenLSTM = torch.nn.LSTM(input_size=config.length_emb_size, hidden_size=config.lenlstmhidden_size, num_layers=config.num_layers, bidirectional=True,
                                     batch_first=True, dropout=config.dropout)
        # 原始字节 嵌入
        self.emb = torch.nn.Embedding(num_embeddings=config.n_vocab, embedding_dim=self.emb_size, padding_idx=0)

        # Transformer 模型用于 raw 特征
        self.TRF = TRF.Model(config=config)
        self.fc01 = torch.nn.Linear(self.emb_size * config.pad_num, config.num_classes)

        # LSTM 用于 length 特征
        self.length_embedding = torch.nn.Embedding(2000, config.length_emb_size, padding_idx=0)
        self.lenlstm = torch.nn.LSTM(config.length_emb_size, config.lenlstmhidden_size, config.num_layers,
                               bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc02 = torch.nn.Linear(config.lenlstmhidden_size * 2, config.num_classes)

        # 最终融合的全连接层
        self.fc = torch.nn.Linear((self.emb_size * config.pad_num + config.lenlstmhidden_size * 2), config.num_classes)

    def forward(self, x):
        # 输入的是一个列表
        config = self.config
        traffic_bytes_idss = x[0]  # raw数据
        length_seq = x[1]  # length数据

        # Transformer 用于 raw 特征提取
        hidden_feature = torch.Tensor(config.pad_num, len(traffic_bytes_idss), self.emb_size).to(config.device)
        # 这里要改一下
        for i in range(config.pad_num):
            packet_emb = self.emb((traffic_bytes_idss[:, i, :]))
            packet_feature = torch.mean(packet_emb, dim=1)  # [batch_size,emb_size]  packet-level embedding
            hidden_feature[i, :, :] = packet_feature
        hidden_feature = hidden_feature.permute(1, 0, 2)
        out1 = self.TRF(hidden_feature)

        # for i in range(config.pad_num):
        #     out_trf = self.TRF(traffic_bytes_idss[:, i, :])  # Transformer 处理每个数据包
        #     hidden_feature[i, :, :] = out_trf
        # hidden_feature = hidden_feature.permute(1, 0, 2)
        # out1 = hidden_feature.view(hidden_feature.size(0), -1)

        # LSTM 用于 length 特征提取
        input = self.length_embedding(length_seq).reshape(-1, config.pad_len_seq, config.length_emb_size)
        output, (final_hidden_state, final_cell_state) = self.lenlstm(input)
        out2 = output[:, -1, :]  # 提取最后一个时刻的输出作为特征

        # 进行特征融合
        out1_classification = self.fc01(out1)
        out2_classification = self.fc02(out2)
        middle_layer = torch.cat((out1, out2), dim=1)

        # 最终分类
        final_output = self.fc(middle_layer)

        return final_output, out1_classification, out2_classification


