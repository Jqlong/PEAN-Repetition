import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import numpy as np

from train_eval import train, evaluate, test
from utils import create_dataloader, MyDataset
from config import Config
import pean_model


# 采用k折交叉验证
def get_k_fold_data(k, i, dataset):
    fold_size = len(dataset) // k
    indices = list(range(len(dataset)))  # 创建索引列表
    train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]  # 训练集索引
    val_indices = indices[i * fold_size: (i + 1) * fold_size]  # 验证集索引

    # 使用 Subset 来构建训练集和验证集
    X_train = Subset(dataset, train_indices)
    X_valid = Subset(dataset, val_indices)

    return X_train, X_valid


def main():
    # 加载配置
    config = Config()

    # 加载数据集
    dataset = MyDataset(config, config.train_path)  # 假设使用整个数据集，之后用于K折划分
    k = 10  # K折交叉验证中的折数

    acc_list = []
    k_list = []
    # K折交叉验证
    for i in range(k):
        k_list.append(i)
        print(f"Fold {i + 1}/{k}")

        # 划分训练集和验证集
        train_data, val_data = get_k_fold_data(k, i, dataset)

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

        # 初始化模型
        model = pean_model.PEAN(config).to(config.device)

        # 定义优化器和损失函数
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.001)
        criterion = CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992)

        # 训练与验证
        for epoch in range(config.epochs):
            train_loss, train_acc = train(config, model, train_loader, optimizer, criterion, config.device)
            val_loss, val_acc = evaluate(config, model, val_loader, criterion, config.device)
            # epoch_list.append(epoch)
            # val_acc_list.append(val_acc)

            print(
                f"Epoch [{epoch + 1}/{config.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val "
                f"Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 在每个折训练完后可以测试一下模型的性能
        test_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
        # test_loader = create_dataloader(config, config.test_path, config.batch_size)
        test_acc = test(config, model, test_loader, criterion, config.device)
        acc_list.append(test_acc)
        print(f"Test Accuracy for fold {i + 1}: {test_acc:.4f}")

    plt.plot(k_list, acc_list)
    plt.xlabel('k')
    plt.ylabel('acc')
    plt.show()


if __name__ == "__main__":
    main()
