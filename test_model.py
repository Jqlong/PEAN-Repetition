import torch
from matplotlib import pyplot as plt
from transformers import BertTokenizer

from Transformer import Positional_Encoding
import Transformer as trf
import seaborn as sns
import utils


class Config:
    def __init__(self):
        self.dataset = 'example_dataset'
        self.vocab_path = 'config/vocab.txt'
        self.train_path = './TrafficData/train.txt'
        self.batch_size = 3
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer(vocab_file='config/vocab.txt', max_len=10, max_seq_length=8)
        self.pad_len_seq = 10

        self.embed = 400
        self.dropout = 0.5
        self.pad_size = 10


config = Config()


position_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)

dataloader = utils.create_dataloader(config, config.train_path, config.batch_size)
for batch in dataloader:
    traffic_bytes_idss = batch['traffic_bytes_idss'].to(config.device).float()
    seq_lens = batch['seq_lens'].to(config.device)
    masks = batch['masks'].to(config.device)
    length_seq = batch['length_seq'].to(config.device)
    length_seq = torch.reshape(length_seq, (-1, config.pad_len_seq, 1)).to(config.device)
    labels = batch['label'].to(config.device)

    pos_embed = position_embedding(traffic_bytes_idss)
    print("Positional Encoding 输出维度:", pos_embed.shape)
    plt.figure(figsize=(5, 5))
    sns.heatmap(pos_embed)
    plt.title("Sinusoidal Function")
    plt.xlabel("hidden dimension")
    plt.ylabel("sequence length")


    # print(labels)

    # 训练过程代码...
    # print(traffic_bytes_idss.shape, seq_lens.shape, masks.shape, length_seq.shape, labels.shape)

# 1. 测试 Positional_Encoding 输出

