import os
import torch
from torch.utils.data import Dataset
import pickle

from tqdm import tqdm


# 构建数据集
class MyDataset(Dataset):
    def __init__(self, path):
        # 缓存，后续直接读取
        cache_dir = './DataCache/'  # 缓存文件夹
        # 缓存文件  sni_whs_10_400_10
        cached_dataset_file = cache_dir + 'train.txt'

        # if os.path.exists(cached_dataset_file):  # 如果存在缓存文件
        #     print("加载缓存文件夹{}".format(cached_dataset_file))
        #     with open(cached_dataset_file, 'rb') as handle:
        #         contents = pickle.load(handle)
        #         return contents
        # else:
        #     pass

        print("构建训练数据集......")
        contents = []
        with open(path, 'r') as f:
            for line in tqdm(f):  # 显示进度条
                if not line:  # 如果当前行为空，继续下一行
                    continue

                item = line.split('\t')  # 制表符隔开




        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


