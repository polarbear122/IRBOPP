import torch

import cv2

img = cv2.imread('./Pictures/python.png', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ', img.shape)

scale_percent = 60  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

if __name__ == "__main__":
    print("torch version:", torch.__version__)
    print(torch.cuda.is_available())  # cuda是否可用
    print("number of gpu:", torch.cuda.device_count())  # 返回GPU的数量
    print("gpu name", torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    # print(np.amax([0, 0, 0, 0, 0], axis=0))
    # pose_array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    # labels = np.array([[0], [1], [2]])
    # print(pose_array[:, 1])
    # a = np.array([labels[0]])
    # b = np.array([labels[0]])
    # c = np.concatenate((a, b), axis=0)
    # print(a, b,"ad", c)
