import torch
import numpy as np
import utils
from config import Config
import pean_model

# 采用k折交叉验证
def get_k_fold_data(k, i, dataset):
    fold_size = len(dataset) // k
    X_train, X_valid = None, None

    # 将数据划分为训练集和验证集
    for j in range(k):
        X_part = dataset[j * fold_size: (j + 1) * fold_size]

        # 第 i 折用作验证集，其他用作训练集
        if j == i:
            X_valid = X_part
        else:
            if X_train is None:
                X_train = X_part
            else:
                X_train += X_part

    return X_train, X_valid


def main():
    # 加载配置
    config = Config()

    # 加载数据集
    dataset = utils.MyDataset(config, config.train_path)  # 数据集
    k = 5

    train_loader = utils.create_dataloader(config, config.train_path, config.batch_size)
    val_loader = utils.create_dataloader(config, config.test_path, config.batch_size)
    test_loader = utils.create_dataloader(config, config.test_path, config.batch_size)


if __name__ == '__main__':
    main()

