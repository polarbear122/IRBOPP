import torch
import numpy as np

# print("torch version:", torch.__version__)
# print(torch.cuda.is_available())  # cuda是否可用
# print("number of gpu:", torch.cuda.device_count())  # 返回GPU的数量
# print("gpu name", torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(np.amax([0, 0, 0, 0, 0], axis=0))
pose_array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
labels = np.array([[0], [1], [2]])
print(pose_array[:, 1])
a = np.array([labels[0]])
b = np.array([labels[0]])
c = np.concatenate((a, b), axis=0)
print(a, b,"ad", c)
