import torch
from transformers import BertTokenizer


class Config:
    def __init__(self):
        self.dataset = 'example_dataset'
        self.vocab_path = 'config/vocab.txt'
        self.train_path = './TrafficData/train.txt'
        self.batch_size = 3
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer(vocab_file='config/vocab.txt', max_len=10, max_seq_length=8)
        self.pad_len_seq = 10
        self.embedding_size = 128
        self.length_emb_size = 32
        self.lenlstmhidden_size = 128
        self.num_layers = 2
        self.dropout = 0.5
        self.n_vocab = 261
        self.trf_heads = 8
        self.trf_layers = 2

        self.pad_num = 10

        self.class_list = [x.strip() for x in open('./TrafficData/class').readlines()]
        self.num_classes = len(self.class_list)
        self.learning_rate = 1e-3
        self.epochs = 100
        self.load = False
