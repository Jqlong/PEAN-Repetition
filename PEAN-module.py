import torch

# 构建模型
class PEAN(torch.nn.Module):
    def __init__(self, config):
        super(PEAN, self).__init__()
        self.config = config
        self.mode = config.mode
        self.emb_size = config.embedding_size

        # 长度序列
        self.length_embedding = torch.nn.Embedding(num_embeddings=2000, embedding_dim=config.length_emb_size, padding_idx=0)  # 嵌入层
        self.lenLSTM = torch.nn.LSTM(input_size=config.length_emb_size, hidden_size=config.lenlstmhidden_size, num_layers=config.num_layers, bidirectional=True,
                                     batch_first=True, dropout=config.dropout)
        # 原始字节 嵌入
        self.emb = torch.nn.Embedding(num_embeddings=config.n_vocab, embedding_dim=self.emb_size, padding_idx=0)




