# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nn_func

import calculate.calculate as cal
from toolkit import get_data


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden * 2)
        self.hidden3 = torch.nn.Linear(n_hidden * 2, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)  # 输入层与第一隐层结点数设置，全连接结构
        out = nn_func.softmax(out)  # 第一隐层激活函数采用sigmoid
        out = self.hidden2(out)
        out = nn_func.softmax(out)
        out = self.hidden3(out)
        out = nn_func.softmax(out)
        out = self.predict(out)
        return out


def train_cnn_pose_trainer(x, y):
    # n_data = torch.ones(100, 2)
    # x0 = torch.normal(2 * n_data, 1)
    # y0 = torch.zeros(100)
    #
    # x1 = torch.normal(-2 * n_data, 1)
    # y1 = torch.ones(100)
    #
    # x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    # y = torch.cat((y0, y1)).type(torch.LongTensor)

    net = Net(21, 64, 1)
    device = torch.device('cuda:0')
    x = x.to(device)
    y = y.reshape(-1, 1).type(torch.FloatTensor)
    y = y.to(device)
    net = net.to(device)
    print("神经网络结构:", net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 优化器使用随机梯度下降，传入网络参数和学习率
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数

    for epoch in range(10000):
        prediction = net(x)  # 喂数据并前向传播.type(torch.LongTensor)
        loss = loss_func(prediction, y)  # 计算损失

        optimizer.zero_grad()  # 清除梯度 PyTorch默认会对梯度进行累加，因此为了不使得之前计算的梯度影响到当前计算，需要手动清除梯度。
        loss.backward()  # 计算梯度，误差回传,反向传播
        optimizer.step()  # 根据计算的梯度，更新网络中的参数
        if epoch % 1000 == 0:
            print("loss:", loss)
            print('epoch: {}, loss: {}'.format(epoch, loss.data.item()))
            nn_func.softmax(prediction)
            # 过了一道 softmax 的激励函数后的最大概率才是预测值
            # print(F.softmax(prediction))
            prediction = torch.max(nn_func.softmax(prediction), 1)[1]
            y_pred = prediction.data.cpu().numpy().squeeze()
            y_test = y.data.cpu().numpy()
            cal.calculate_all(y_test, y_pred)  # 评估计算结果
    # torch.max(y_p,dim = 1)[0]是每行最大的值
    # torch.max(y_p,dim = 1)[0]是每行最大的值的下标，可认为标签


if __name__ == "__main__":
    train_dataset, labels = get_data.read_csv_train_label_data(data_id=2)  # 输出为numpy矩阵,shape(num,21),(num,)
    train_dataset_torch = torch.from_numpy(train_dataset).type(torch.FloatTensor)
    labels_torch = torch.from_numpy(labels).type(
        torch.LongTensor)  # tensor和numpy对象共享内存，转换很快，几乎不消耗资源; 但如果其中一个变了，另外一个也随之改变，
    train_cnn_pose_trainer(train_dataset_torch, labels_torch)
