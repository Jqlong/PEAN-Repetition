# PEAN-Repetition

# 1. 项目介绍

对论文《A Novel Multimodal Deep Learning Framework  for Encrypted Traffic Classification》的代码进行复现。

**源代码链接**：https://github.com/Lin-Dada/PEAN

# 2. 具体工作

对原代码进行了一些精简和改进：

1. 数据集处理方面：(utils.py)

   - 原始代码使用build_dataset加载数据集，对数据进行处理，使用DatasetIterater构建训练数据集。在此基础上，本次对数据处理方面进行了一些改进，将其改造为标准的Dataset和DataLoader结构。

   - 使用MyDataset(Dataset)类继承Dataset，load_dataset函数用于处理数据，并生成缓存文件，返回的结构为：

     ```python
     return {
         "traffic_bytes_idss": torch.LongTensor(traffic_bytes_idss),
         "seq_lens": torch.LongTensor(seq_lens),
         "masks": torch.LongTensor(masks),
         "length_seq": torch.LongTensor(length_seq),
         # 确保 label 是一个标量
         "label": torch.tensor(label, dtype=torch.long)
     }
     ```

     其中"traffic_bytes_idss"的格式为：数据包字节对应的索引；

     ​		“seq_lens”：为数据包的长度，格式为：[400, 400, 400, 148, 264, 400, 400, 0, 0, 0]，0表示当前会话长度不够10个数据包

     ​		“length_seq”：表示数据包的长度序列，格式为：[537, 1360, 1278, 146, 262, 502, 426, 0, 0, 0]，在进行截断前进行长度获取

     ​		”label“：表示类别

   - 使用torch.utils.data.DataLoader构建训练和测试集，

   - 至此，数据处理工作完成

2. Transformer模型方面：(未进行改动)

3. PEAN模型方面：(pean_model.py)

   - 去除了pretrain（后续再考虑加上）;
   - 对原始字节，使用Transformer
   - 对长度序列，使用LSTM
   - 去除了其他选项

4. 训练和测试方面：(train_eavl.py)

   - 进行了很多的精简，单纯进行了训练、损失计算、反向传播和权重更新等任务。
   - 不同的是，在训练一个轮次后进行了验证，在训练完整个模型后，进行了测试
   - 对划分数据集方面，直接使用原始的get_k_fold_data函数会有问题，因为dataset变化了，变成了切片对象，而不是单个索引，所以需要进行一些改进。

# 3. 对数据形状进行分析

1. 原始字节维度变化分析：
   - 首先DataLoader加载的mini-batch，其中traffic_bytes_idss的维度为： **torch.Size([3, 10, 400])**，表示为batch x pad_num x pad_length，分别表示批量大小、数据包的数量、每个数据包的字节数。
   - 通过emb嵌入层torch.nn.Embedding(num_embeddings=config.n_vocab, embedding_dim=self.emb_size, padding_idx=0)，将原始字节转换为高纬的嵌入向量，维度变为**torch.Size([3, 400, 128])**，表示每个数据包的字节被映射为128维的向量。
   - 通过对嵌入向量进行平均池化操作，将维度变为**torch.Size([3, 128])**，将数据包特征向量存储到hidden_feature中，形状为[config.pad_num, batch_size, emb_size]->**torch.Size([10, 3, 128])**
   - 对 `hidden_feature` 进行维度置换，交换张量的维度顺序，形状变为：[batch_size, config.pad_num, emb_size]->**torch.Size([3, 10, 128])**，输入到Transformer中

