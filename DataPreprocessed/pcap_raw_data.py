# from scapy.all import *
#
# def extract_raw_data(pcap_file_path, output_file_path):
#     # 使用scapy打开pcap文件
#     packets = rdpcap(pcap_file_path)
#
#     # 打开输出文件，以二进制写入模式
#     with open(output_file_path, 'wb') as output_file:
#         # 遍历每个数据包，提取原始数据并写入输出文件
#         for packet in packets:
#             if Raw in packet:  # 检查数据包中是否有原始数据
#                 raw_data = bytes(packet[Raw])  # 获取原始数据部分
#                 output_file.write(raw_data)  # 将原始数据写入输出文件
#                 print(raw_data)
#
#     print(f"原始数据提取完成，保存在 {output_file_path}")
#
#
# if __name__ == '__main__':
#     # 指定输入的 pcap 文件路径
#     # input_pcap_file = 'facebook_video2a.pcap.TCP_23-195-220-179_443_131-202-240-150_47976.pcap'
#     input_pcap_file = "G:\\Users\\22357\\Desktop\\thesis\\dataset\\NonVPN-PCAPs-01\\2_Session\AllLayers\\facebook_video2a-ALL\\facebook_video2a.pcap.TCP_23-195-220-179_443_131-202-240-150_47976.pcap"
#     output_txt_file = "G:\\Users\\22357\\Desktop\\thesis\\dataset\\NonVPN-PCAPs-01\\2_Session\AllLayers\\facebook_video2a-ALL\\facebook_video2a.pcap.TCP_23-195-220-179_443_131-202-240-150_47976.txt"
#
#     # 提取 TCP 流的原始数据并打印输出
#     extract_raw_data(input_pcap_file, output_txt_file)
import os
from scapy.all import *
from tqdm import tqdm  # 导入 tqdm 库

# 定义类别和对应的数字的字典（全部转换为小写）
categories = {
    "BitTorrent": 0,
    "Facetime": 1,
    "FTP": 2,
    "Gmail": 3,
    "MySQL": 4,
    "Outlook": 5,
    "Skype": 6,
    "SMB": 7,
    "Weibo": 8,
    "WorldOfWarcraft": 9,
}


# 输入文件
def extract_transport_protocol_flow_raw_data_hex(pcap_file, output_file):
    packets = rdpcap(pcap_file)  # 使用 scapy 读取 pcap 文件中的数据包
    count = 0

    # 用于存储对应流的原始数据的十六进制表示
    transport_flow_raw_data_hex = []
    raw_data_length = []  # 存储每一行的字节数

    # 判断文件名中包含的协议类型
    if "TCP" in pcap_file.upper():
        protocol = "TCP"
    elif "UDP" in pcap_file.upper():
        protocol = "UDP"
    else:
        print(f"Unsupported protocol in filename: {pcap_file}")
        return transport_flow_raw_data_hex, raw_data_length  # 如果不是 TCP 或 UDP 文件，则返回空列表

    with open(output_file, 'a') as raw_packet:
        # 遍历数据包，查找对应协议类型的流，并提取原始数据的十六进制表示
        for packet in packets:
            # for packet in tqdm(packets, desc="Processing"):
            flag = False
            if protocol in packet:  # 只处理指定协议类型的数据包
                if packet[protocol].payload:
                    # 获取协议数据包的原始数据
                    raw_data = bytes(packet[protocol].payload)

                    # 检查原始数据是否全是 00，如果是则跳过
                    if all(byte == 0 for byte in raw_data):
                        continue
                    # 将原始数据的每个字节转换为十六进制表示，并存储到列表中
                    hex_representation = ' '.join(f'{byte:02x}' for byte in raw_data)
                    # print(len(raw_data))
                    # 每读取一个数据包，就写进去
                    if len(hex_representation) != 0:
                        flag = True
                    raw_packet.write(f"{hex_representation}\t")
                    transport_flow_raw_data_hex.append(hex_representation)
                    # 统计数据包大小
                    raw_data_length.append(len(raw_data))
            count += 1
        # print(raw_data_length)
        # 在这里遍历完一个文件的所有数据包
        # 此时将数据包信息写进去
        # 判断类别
        category_number = None
        for category, number in categories.items():
            if category.lower() in pcap_file.lower():
                category_number = number
                break
        if category_number is not None:
            if len(raw_data_length) != 0:
                # raw_packet.write(f"{output_line}\t{' '.join(map(str, raw_data_length))}\t{category_number}\n")
                raw_packet.write(f"{' '.join(map(str, raw_data_length))}\t")
                raw_packet.write(f"{category_number}\n")
    # print('count', count)

    # return transport_flow_raw_data_hex, raw_data_length


def process_folder(folder_path, output_file):
    # 列出文件夹中的所有文件和子文件夹

    # 列出所有文件夹
    contents = os.listdir(folder_path)
    # print(contents)

    # 遍历文件夹中的每个项目
    # for item in contents:
    for item in tqdm(contents, desc="Processing"):
        # print(item)
        item_path = os.path.join(folder_path, item)
        # print(item_path)

        if os.path.isdir(item_path):  # 如果是子文件夹，则递归处理子文件夹
            print('\n')
            print("当前文件夹：", item_path)
            with open('log_item_path.txt', 'a') as item_log_path:
                item_log_path.write(f"{item_path}\n")
            process_folder(item_path, output_file)
        elif item.endswith(".pcap"):  # 如果是 .pcap 文件，则处理该文件
            # 提取对应协议类型的流的原始数据的十六进制表示和每行的字节数
            extract_transport_protocol_flow_raw_data_hex(item_path, output_file)

            # 合并成一行输出，用制表符分隔
            # output_line = '\t'.join(transport_flow_raw_data_hex)

            # 判断 output_line 是否为空，如果不为空则写入到输出文件中
            # if output_line.strip():  # 使用 strip() 方法去除首尾空白字符，判断是否为空
            # 将 raw_data_length 内部用空格分隔，与 output_line 用制表符分隔，写入到输出文件中

            # category_number = None
            # for category, number in categories.items():
            #     if category.lower() in item.lower():
            #         category_number = number
            #         break
            # if category_number is not None:
            #
            #     output_file.write(f"{output_line}\t{' '.join(map(str, raw_data_length))}\t{category_number}\n")


# 指定要读取的根文件夹路径和输出文件路径
# root_folder_path = "G:\\Users\\22357\\Desktop\\thesis\\DeepTraffic\\2.encrypted_traffic_classification\\2.PreprocessedTools\\2_Session\\AllLayers\\"
root_folder_path = 'D:\\PEAN-Repetition\\PreprocessedTools\\2_Session\AllLayers'
# output_file_path = "output_5_train.txt"
output_file_path = "../TrafficData/Data.txt"

# 打开输出文件，使用 'w' 模式，确保每次写入会覆盖之前的内容
# with open(output_file_path, 'w') as output_file:
# 递归处理根文件夹下的所有文件和子文件夹
process_folder(root_folder_path, output_file_path)
# extract_transport_protocol_flow_raw_data_hex(root_folder_path, output_file_path)

print("Processing completed.")

# folder_path = "G:\\Users\\22357\\Desktop\\thesis\\DeepTraffic\\2.encrypted_traffic_classification\\2.PreprocessedTools\\2_Session\\AllLayers\\aim_chat_3a-ALL"
# output_file_path = 'dataset.txt'
#
# with open(output_file_path, 'w') as output_file:
#     files = os.listdir(folder_path)
#     for index, filename in enumerate(files):
#         # print(filename)
#         pcap_file_path = os.path.join(folder_path, filename)
#         # print(pcap_file_path)
#         transport_flow_raw_data_hex, raw_data_length = extract_transport_protocol_flow_raw_data_hex(pcap_file_path)
#         output_line = '\t'.join(transport_flow_raw_data_hex)
#         print(output_line)
#         if index < len(files) - 1:  # 非最后一行
#
#             if output_line.strip():
#                 # output_file.write(f"{output_line}\n")
#                 output_file.write(output_line + "\t" + " ".join(map(str, raw_data_length)) + "\n")
#
#         else:  # 最后一行
#             if output_line.strip():
#                 output_file.write(f"{output_line}")
# if index < len(files) - 1:
#     output_file.write('\n')

# 指定要读取的 pcap 文件名
# pcap_filename = "G:\\Users\\22357\\Desktop\\thesis\\dataset\\NonVPN-PCAPs-02\\2_Session\\AllLayers\\hangouts_audio2a-ALL\\hangouts_audio2a.pcap.TCP_131-202-240-150_47796_216-58-219-238_443.pcap"

# 提取对应协议类型的流的原始数据的十六进制表示
# transport_flow_raw_data_hex, raw_data_length = extract_transport_protocol_flow_raw_data_hex(pcap_filename)

# 输出原始数据的十六进制表示
# for line, line_len in zip(transport_flow_raw_data_hex, raw_data_length):
#     # print(len(line))
#     print(line, line_len)

# for line in transport_flow_raw_data_hex:
#     print(line)

# 合并成一行输出，用制表符分隔
# output_line = '\t'.join(transport_flow_raw_data_hex)

# 输出合并后的一行数据
# print(output_line)
