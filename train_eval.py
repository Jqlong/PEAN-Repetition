import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import utils

# 用于模型的训练和测试
def train(config, model, train_loader, optimizer, criterion, device):

    # 加载模型
    if config.load:
        print("\n\nloading model from: {}".format(config.save_path))
        with open(config.print_path, 'a') as f:
            f.write("loading model from: {}\n".format(config.save_path))
        model.load_state_dict(torch.load(config.save_path))

    model.train()  # 训练模型

    total_loss = 0
    total_preds = []
    total_labels = []

    for batch in train_loader:
        pass



