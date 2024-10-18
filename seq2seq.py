#!/usr/bin/python3
# -*- encoding: utf-8 -*- 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

DAYS_FOR_TRAIN = 90  # 使用前面90天数据来预测
DAYS_FOR_PREDICT = 90  # 预测后90天数据

# 定义 Encoder 模型
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)  # 返回最后的hidden和cell状态
        return hidden, cell

# 定义 Decoder 模型
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.fc(out)
        return out, hidden, cell

# Seq2Seq 模型，整合 Encoder 和 Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target_len):
        batch_size = source.shape[0]
        target_size = source.shape[2]

        # Encoder forward
        hidden, cell = self.encoder(source)

        # 生成初始输入给 Decoder（假设第一个输入为0）
        decoder_input = torch.zeros((batch_size, 1, target_size)).to(self.device)
        
        outputs = []
        for t in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(output)
            decoder_input = output  # 将当前输出作为下一个时间步的输入

        return torch.cat(outputs, dim=1)

# 数据集生成函数，生成输入数据和标签
def create_dataset(data, days_for_train=90, days_for_predict=90):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train - days_for_predict):
        _x = data[i:(i + days_for_train)]
        _y = data[(i + days_for_train):(i + days_for_train + days_for_predict)]
        dataset_x.append(_x)
        dataset_y.append(_y)
    return np.array(dataset_x), np.array(dataset_y)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数
INPUT_SIZE = 1
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
NUM_LAYERS = 1

# 读取数据
data = pd.read_csv('./data/600000.SH.csv')
data_close = data['close'].values  # 读取收盘价数据

# 数据标准化 (0~1)
max_value = np.max(data_close)
min_value = np.min(data_close)
data_close = (data_close - min_value) / (max_value - min_value)

# 构建数据集
dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN, DAYS_FOR_PREDICT)
X_tensor = torch.Tensor(dataset_x).float().unsqueeze(-1).to(device)  # shape (samples, days_for_train, 1)
Y_tensor = torch.Tensor(dataset_y).float().unsqueeze(-1).to(device)  # shape (samples, days_for_predict, 1)

# # 定义模型
encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
decoder = Decoder(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
NUM_EPOCHS = 200
for epoch in range(NUM_EPOCHS):
    model.train()
    
    # 前向传播
    outputs = model(X_tensor, DAYS_FOR_PREDICT)
    loss = criterion(outputs, Y_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

# 预测和可视化
model.eval()
predicted = model(X_tensor, DAYS_FOR_PREDICT).detach().to('cpu').numpy()

# 还原数据
predicted = predicted * (max_value - min_value) + min_value
dataset_y = dataset_y * (max_value - min_value) + min_value

# 绘制真实值和预测值
plt.figure(figsize=(10, 5))
plt.plot(dataset_y[-1], label="True Data")
plt.plot(predicted[-1], label="Predicted Data")
plt.legend()
plt.show()
