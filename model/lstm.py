import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

input_size = 1  # 每个时间步输入的特征数
seq_length = 100  # 序列长度
hidden_size = 64  # LSTM 隐藏层大小
num_layers = 3  # LSTM 层数
num_classes = 2  # 分类数量


# 定义 LSTM 模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes):
        """
        初始化 LSTM 分类模型。

        :param input_size: 输入特征的大小
        :param hidden_size: LSTM 隐藏层的大小
        :param num_layers: LSTM 的层数
        :param num_classes: 输出分类的数量
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，用于分类
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        前向传播。

        :param x: 输入张量，形状为 (batch_size, seq_length, input_size)
        :return: 输出分类的 logits，形状为 (batch_size, num_classes)
        """

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)
        return out

    def reset(self):
        self.h = torch.zeros(self.num_layers, self.hidden_size)
        self.c = torch.zeros(self.num_layers, self.hidden_size)

    def step_forward(self, x):
        # LSTM 前向传播
        out, (self.h, self.c) = self.lstm(x.unsqueeze(-1), (self.h, self.c))
        # 取最后一个时间步的输出
        out = out[-1, :]
        # 全连接层
        out = self.fc(out)
        return out, self.h, self.c
