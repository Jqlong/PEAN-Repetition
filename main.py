import torch
import numpy as np
import utils
from config import Config


# 数据集

def perpare_dataset(config):
    # 准备数据
    train_data = utils.create_dataloader(config, config.train_path, config.batch_size)
    print("训练数据集的长度：", len(train_data))


if __name__ == '__main__':
    config = Config()
    perpare_dataset(config)  # 准备数据

