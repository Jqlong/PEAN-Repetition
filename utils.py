import os
import torch
from torch.utils.data import Dataset
import pickle
from transformers import BertTokenizer
tokenizer = BertTokenizer(vocab_file='./confit/vocab.txt', max_len=10, max_seq_length=8)


from tqdm import tqdm


UNK, PAD, CLS, SEP = '[UNK]', '[PAD]', '[CLS]','[SEP]'

# 构建数据集
class MyDataset(Dataset):
    def __init__(self, path, pad_num=10, pad_length=400, pad_len_seq=10):
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
        # 数据集格式：数据包\t数据包\t数据包长度\t类别
        contents = []
        with open(path, 'r') as f:
            for line in tqdm(f):  # 显示进度条
                if not line:  # 如果当前行为空，继续下一行
                    continue

                item = line.split('\t')  # 获取数据包的原始字节 长度序列 类别
                # print(len(item))  # 现实的是一共有多少项
                flow = item[0:-2]  # 取流量数据-->数据包
                # print(len(flow))  # 数据包数量
                if len(flow) < 2:
                    # 数据包数量小于2
                    continue
                if len(flow) > 10: # 数据包个数大于10
                    flow = flow[0: 10]  # 取前10个数据包
                length_seq = item[-2].strip().split(' ')  # 长度序列
                # print(length_seq)
                length_seq = list(map(int, length_seq))  # 转换为整数列表
                # print(length_seq)

                labels = item[-1]  # 类别

                masks = []  # 创建掩码
                seq_lens = []
                traffic_bytes_idss = []

                for packet in flow:  # 遍历流中的每个数据包
                    traffic_bytes = tokenizer.tokenize(packet)  # 对每一行的每一个数据包进行分词处理，
                    # print(traffic_bytes)
                    # print(len(traffic_bytes))
                    if len(traffic_bytes) <= pad_length:  # 对长度小于400的进行处理
                        # 如果数据包长度小于400字节
                        traffic_bytes = [CLS] + traffic_bytes + [SEP]  # 填充开始和结束
                        # print(traffic_bytes)
                        # print(len(traffic_bytes))
                    else:
                        traffic_bytes = [CLS] + traffic_bytes
                        traffic_bytes[pad_length - 1] = SEP  # 截断

                    seq_len = len(traffic_bytes)
                    # print('原始len', seq_len)
                    mask = []
                    traffic_bytes_ids = tokenizer.convert_tokens_to_ids(traffic_bytes)  # 将字节转化为vocab.txt对应的索引
                    # print(traffic_bytes_ids)

                    # 处理每个数据包的长度
                    if pad_length:
                        if len(traffic_bytes) < pad_length:  # 小于400
                            # 进行填充 1表示是真实token 0表示是填充
                            mask = [1] * len(traffic_bytes_ids) + [0] * (pad_length - len(traffic_bytes))  # [1,1,1,0,0,0]
                            traffic_bytes_ids += ([0] * (pad_length - len(traffic_bytes)))  # 后面的数据全为0  这时候全部变为400字节
                            # print(len(traffic_bytes_ids))
                            # print(traffic_bytes_ids)
                        else:
                            # 长度大于400 进行截断
                            mask = [1] * pad_length
                            traffic_bytes_ids = traffic_bytes_ids[:pad_length]
                            seq_len = pad_length
                            # print("截断len", seq_len)
                    traffic_bytes_idss.append(traffic_bytes_ids)
                    seq_lens.append(seq_len)  # 400和小于400的长度序列  最大10个  [131, 400, 400, 340, 73, 400, 400, 227]
                    # print(seq_lens)
                    masks.append(mask)
                    # print(masks)

                    # 处理每个数据包的长度序列
                    if pad_len_seq:  # 处理的都是真实长度，并非填充和截断后的长度
                        if len(length_seq) < pad_len_seq:
                            length_seq += [0] * (pad_len_seq - len(length_seq))  # 后面的都为0
                        else:
                            length_seq = length_seq[:pad_len_seq]  # 截断长度

                # 此时已经处理完每个流的每个数据包
                # print(len(traffic_bytes_idss))  # 表示一个流的数据包数量
                if pad_num:  # 如果一个流的长度不够10个数据包
                    if len(traffic_bytes_idss) < pad_num:  # 对长度不够10的流进行处理
                        len_tmp = len(traffic_bytes_idss)  # 保存原始长度
                        # print(len_tmp)
                        mask = [0] * pad_length  # 先全为零
                        # print(mask)
                        traffic_bytes_ids = [1] + [0] * (pad_length - 2) + [2]  # 构建一个空的数据包 长度为400
                        # print(traffic_bytes_ids)
                        # print(len(traffic_bytes_ids))

                        seq_len = 0
                        for i in range(pad_num - len_tmp):
                            # 需要填充几个400字节的数据包
                            masks.append(mask)  # 这是一个二维列表 每一维代表一个数据包 1代表真实字节，0表示填充字节
                            # print(masks)
                            traffic_bytes_idss.append(traffic_bytes_ids)
                            seq_lens.append(seq_len)
                    else:
                        # 大于10个数据包
                        traffic_bytes_idss = traffic_bytes_idss[:pad_num]  # 截断
                        masks = masks[:pad_num]
                        seq_lens = seq_lens[:pad_num]
                print(seq_lens)

                # traffic_bytes_idss：数据包字节对应的索引,
                # seq_lens：数据包的长度  [400, 400, 400, 148, 264, 400, 400, 0, 0, 0]
                # masks
                # length_seq,
                # int(labels)
                contents.append((traffic_bytes_idss, seq_lens, masks, length_seq, int(labels)))  # 处理完一行，也就是一个流




                # print(seq_lens)


        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


MyDataset('./TrafficData/train.txt')
