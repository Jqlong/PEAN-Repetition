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
        # 字节索引在261之间，
        self.emb = torch.nn.Embedding(num_embeddings=config.n_vocab, embedding_dim=self.emb_size, padding_idx=0)

        # Transformer 模型用于 raw 特征
        self.TRF = TRF.Model(config=config)
        self.fc01 = torch.nn.Linear(self.emb_size * config.pad_num, config.num_classes)

        # LSTM 用于 length 特征
        self.length_embedding = torch.nn.Embedding(2000, config.length_emb_size, padding_idx=0)

        # length_emb_size：这是输入到 LSTM 的每个时间步的特征维度  32
        # lenlstmhidden_size：这是 LSTM 隐藏层的输出维度  128
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

        # print('len(traffic_bytes_idss)', len(traffic_bytes_idss))  # 等于batch大小

        # Transformer 用于 raw 特征提取

        # 创建一个张量，用于存储之后经过Transformer编码后的特征
        # 数据包数量 x 批量大小 x 嵌入层的维度
        hidden_feature = torch.Tensor(config.pad_num, len(traffic_bytes_idss), self.emb_size).to(config.device)
        # print('hidden_feature', hidden_feature.shape)  # hidden_feature torch.Size([10, 3, 128])
        # 这里要改一下
        for i in range(config.pad_num):  # 遍历每个数据包，对数据包进行嵌入操作
            # 取出第i个数据包的所有数据，[batch_size, config.pad_num, packet_length] 的张量，表示一批流中每个流的数据包序列。
            # 通过self.emb，每个字节（范围为261）都被映射为128维的向量，
            packet_emb = self.emb((traffic_bytes_idss[:, i, :]))
            # print('packet_emb的维度', packet_emb.shape)  torch.Size([3, 400, 128])

            # 对嵌入向量进行平均池化操作，计算每个数据包的嵌入的平均值。
            # 沿着数据包长度维度进行平均，取平均值  packet_feature torch.Size([3, 128])
            packet_feature = torch.mean(packet_emb, dim=1)  # [batch_size,emb_size]  packet-level embedding
            # print('packet_feature', packet_feature.shape)

            # 将数据包特征向量存储到hidden_feature中
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
        # print('input的形状', input.shape)  # torch.Size([3, 10, 32])
        output, (final_hidden_state, final_cell_state) = self.lenlstm(input)
        # print('output形状', output.shape)  # torch.Size([3, 10, 256])

        # 最后一个实践部的输出，可以作为整个序列的表示
        out2 = output[:, -1, :]  # 提取最后一个时刻的输出作为特征
        # print('out2的形状', out2.shape)  # torch.Size([3, 256])。

        # 进行特征融合
        out1_classification = self.fc01(out1)
        out2_classification = self.fc02(out2)
        middle_layer = torch.cat((out1, out2), dim=1)
        # print('middle_layer的形状', middle_layer.shape)  torch.Size([3, 1536])

        # 最终分类
        final_output = self.fc(middle_layer)

        return final_output, out1_classification, out2_classification


