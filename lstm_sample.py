 #!/usr/bin/python3
# -*- encoding: utf-8 -*- 
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import pandas as pd
import torch
from torch import nn
import datetime
import time
 
DAYS_FOR_TRAIN = 90  # 使用前面多少天的数据来预测
 
 
# 3. 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 前向传播 LSTM
        out, _ = self.lstm(x)
        # 通过全连接层输出预测值（取最后一个时间步的输出）
        out = self.fc(out[:, -1, :])
        return out
 
def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集
        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。
        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))
 
 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否使用 GPU
    # 读取数据
    # pro = ts.pro_api("585565065385971c7b140758d4b2c613598334a552c951af88c03d7e")
    # data = pro.daily(ts_code='600000.SH', start_date='20140101', end_date='20240101')
    # data.to_csv('./data/600000.SH.csv')
    data = pd.read_csv('./data/600000.SH.csv')
    data_close = data['close']
    # t0 = time.time()
    # # 将价格标准化到0~1
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close = (data_close - min_value) / (max_value - min_value)
    # dataset_x
    # 是形状为(样本数, 时间窗口大小)
    # 的二维数组，用于训练模型的输入
    # dataset_y
    # 是形状为(样本数, )
    # 的一维数组，用于训练模型的输出。
    dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)  # 分别是（1007,10,1）（1007,1）
    X_tensor = torch.Tensor(dataset_x).float().unsqueeze(-1).to(device)
    Y_tensor = torch.Tensor(dataset_y).float().unsqueeze(-1).to(device)
    print(X_tensor.shape, Y_tensor.shape)
    # 模型参数
    input_size = 1
    hidden_size = 50
    output_size = 1
    num_layers = 1

    model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 5. 训练模型
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        
        # 前向传播
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 6. 预测和可视化结果
    model.eval()
    predicted = model(X_tensor).detach().to('cpu').numpy()
    print(predicted.shape)
    # 绘制真实值和预测值
    plt.plot(dataset_y, label="True Data")
    plt.plot(predicted.squeeze(), label="Predicted Data")
    plt.legend()
    plt.show()