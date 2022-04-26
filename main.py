import torch

print("torch version:", torch.__version__)
print(torch.cuda.is_available())  # cuda是否可用
print("number of gpu:", torch.cuda.device_count())  # 返回GPU的数量
print("gpu name", torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始
print(torch.version.cuda)
print(torch.backends.cudnn.version())
