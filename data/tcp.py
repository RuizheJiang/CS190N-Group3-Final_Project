import json
import os
from scapy.all import rdpcap
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class FlowDataLoader:
    def __init__(self, path1: str, path2: str):
        """
        初始化 DataLoader 类，指定两个路径。

        :param path1: 第一个路径，作为一个 label
        :param path2: 第二个路径，作为另一个 label
        """
        self.paths = [(path1, os.path.basename(path1)), (path2, os.path.basename(path2))]

    def _read_pcap(self, filepath: str) -> List[int]:
        """
        从单个 pcap 文件中提取所有包的 win 成员。

        :param filepath: pcap 文件的路径
        :return: 包含 win 成员的列表
        """
        packets = rdpcap(filepath)
        win_values = []
        for packet in packets:
            # 检查是否有 TCP 层
            if hasattr(packet, 'haslayer') and packet.haslayer('TCP'):
                win_values.append(packet['TCP'].window)
        return win_values

    def load_data(self) -> List[Tuple[List[int], str]]:
        """
        加载所有路径下的 pcap 文件并提取数据和标签。

        :return: 一个包含 (data, label) 的列表，其中：
                 data 是一个包含 win 成员的序列
                 label 是路径名称
        """
        dataset = []
        for path, label in self.paths:
            if not os.path.exists(path):
                print(f"路径 {path} 不存在，跳过...")
                continue
            for file in os.listdir(path):
                if file.endswith('.pcap'):
                    filepath = os.path.join(path, file)
                    print(f"正在读取文件: {filepath}")
                    try:
                        data = self._read_pcap(filepath)
                        dataset.append((data, label))
                    except Exception as e:
                        print(f"读取文件 {filepath} 时出错: {e}")
        return dataset


class PcapDataset(Dataset):
    def __init__(self):
        """
        初始化 PcapDataset。

        :param data: 包含 (data, label) 的列表
        """

        with open('flows.json', 'r', encoding='utf-8') as f:
            formatted_data = json.load(f)
        # 转换为 List[Tuple[List[int], str]] 的形式
        self.data = [(item["data"], item["label"]) for item in formatted_data]
        self.labels = list(set(label for _, label in self.data))  # 提取唯一的标签集合
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}  # 标签到索引映射

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的数据。

        :param index: 数据索引
        :return: 一个包含张量化的 data 和 label 索引的元组
        """
        data, label = self.data[index]
        # 将 data 转换为 PyTorch 张量
        data_tensor = torch.tensor(data, dtype=torch.float32)
        # 将 label 转换为索引形式
        label_idx = self.label_to_idx[label]
        return data_tensor, label_idx


def split_dataset(dataset, train_ratio=0.8, seed=None):
    """
    将数据集随机分割为训练集和验证集。

    :param dataset: 要分割的 PyTorch Dataset 对象
    :param train_ratio: 训练集占总数据集的比例，默认为 0.8
    :param seed: 随机种子，确保结果可复现
    :return: (train_dataset, val_dataset)
    """
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    if seed is not None:
        torch.manual_seed(seed)

    return random_split(dataset, [train_size, val_size])

# 使用示例
if __name__ == "__main__":
    path1 = "cubic(2)"
    path2 = "reno(2)"
    dataloader = FlowDataLoader(path1, path2)
    dataset = dataloader.load_data()
    formatted_data = [{"data": d, "label": l} for d, l in dataset]
    with open('flows.json', 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    # with open('flows.json', 'r', encoding='utf-8') as f:
    #     formatted_data = json.load(f)
    # # 转换为 List[Tuple[List[int], str]] 的形式
    # dataset = [(item["data"], item["label"]) for item in formatted_data]
    # pcap_dataset = PcapDataset(dataset)
    # dataloader = DataLoader(pcap_dataset, batch_size=4, shuffle=True)
    # for x, y in dataloader:
    #     print(x, y)
    # print(dataset[0])
    # for data, label in dataset:
    #     print(f"Label: {label}, Data Length: {len(data)}")
