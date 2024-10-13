import torch
from transformers import BertTokenizer


class Config:
    def __init__(self):
        self.dataset = 'example_dataset'
        self.vocab_path = 'config/vocab.txt'
        self.train_path = './TrafficData/train.txt'
        self.batch_size = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer(vocab_file='config/vocab.txt', max_len=10, max_seq_length=8)
        self.pad_len_seq = 10