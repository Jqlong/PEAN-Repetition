from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from utils import create_dataloader, MyDataset
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
    dataset = MyDataset(config, config.train_path)  # 数据集
    k = 10  # 采用10折交叉验证

    for i in range(k):
        print(f"fold {i + 1} / {k}")
        # 划分训练集和测试集
        train_data, test_data = get_k_fold_data(k, i, dataset)

        # 创建数据加载器
        train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
        test_val = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)

        # 初始化模型
        model = pean_model.PEAN(config).to(config.device)



if __name__ == '__main__':
    main()

