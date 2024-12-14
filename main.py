import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data.tcp import PcapDataset, split_dataset
from model.lstm import LSTMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 配置超参数
    seq_length = 100
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建数据集和 DataLoader
    dataset = PcapDataset()
    train_dataset, val_dataset = split_dataset(dataset, seed=401234)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = LSTMClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录每轮的损失值
    epoch_losses = []

    # 训练模型
    for epoch in range(num_epochs):
        model.train()  # 进入训练模式
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 平均损失
        epoch_loss /= len(trainloader)
        epoch_losses.append(epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training finished!")

    # 测试模型并计算评价指标
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 计算指标
    acc = (torch.tensor(y_true) == torch.tensor(y_pred)).sum().item() / len(y_true)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # 打印结果
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 绘制损失曲线
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()