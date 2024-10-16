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
        optimizer.zero_grad()

        # 传数据
        # return (traffic_bytes_idss, length_seq, seq_lens, masks), label

        traffic_bytes_idss = batch["traffic_bytes_idss"].to(device)  # 原始字节
        length_seq = batch["length_seq"].to(device)  # 长度序列
        label = batch["label"].to(device)  # 标签

        # print("traffic_bytes_idss维度", traffic_bytes_idss.shape)  # torch.Size([1, 10, 400])

        # 前向传播
        output, out1_classification, out2_classification = model([traffic_bytes_idss, length_seq])

        # 计算损失
        loss = criterion(output, label)
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()

# 预测值
        preds = torch.argmax(output, dim=1)
        total_preds.extend(preds.cpu().numpy())
        total_labels.extend(label.cpu().numpy())

    # 计算平均损失与准确率
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(total_labels, total_preds)

    return avg_loss, acc

def evaluate(config, model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch in val_loader:
            traffic_bytes_idss = batch["traffic_bytes_idss"].to(device)
            length_seq = batch["length_seq"].to(device)
            label = batch["label"].to(device)

            # 前向传播
            output, out1_classification, out2_classification = model([traffic_bytes_idss, length_seq])

            # 计算损失
            loss = criterion(output, label)
            total_loss += loss.item()

            # 预测值
            preds = torch.argmax(output, dim=1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(label.cpu().numpy())

    # 计算平均损失与准确率
    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(total_labels, total_preds)

    return avg_loss, acc


def test(config, model, test_loader, criterion, device):
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch in test_loader:
            traffic_bytes_idss = batch["traffic_bytes_idss"].to(device)
            length_seq = batch["length_seq"].to(device)
            label = batch["label"].to(device)

            # 前向传播
            output, out1_classification, out2_classification = model([traffic_bytes_idss, length_seq])

            # 预测值
            preds = torch.argmax(output, dim=1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(label.cpu().numpy())

    acc = accuracy_score(total_labels, total_preds)
    return acc


