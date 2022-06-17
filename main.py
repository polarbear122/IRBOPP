import torch

import cv2
import numpy as np
import train.test_joint_image_video as jo
import pandas as pd


def test_cuda():
    print("torch version:", torch.__version__)
    print(torch.cuda.is_available())  # cuda是否可用
    print("number of gpu:", torch.cuda.device_count())  # 返回GPU的数量
    print("gpu name", torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())


def test_numpy():
    print(np.amax([0, 0, 0, 0, 0], axis=0))
    pose_array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    labels = np.array([[0], [1], [2]])
    print(pose_array[:, 1])
    a = np.array([labels[0]])
    b = np.array([labels[0]])
    c = np.concatenate((a, b), axis=0)
    print(a, b, "ad", c)


def test_img_resize():
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


def test_python_input():
    init_list = [0] * 10
    print(init_list)
    jo.add_one_num(init_list, 1)
    print(init_list)
    re = jo.calculate_result(init_list)
    print(re)
    print(init_list)


def test_file_read():
    data_path = "halpe26_data/data_by_video/single/"

    for str_id in range(2, 10):
        try:
            f = open(data_path + "data" + str(str_id) + ".csv", encoding='utf-8')
            pose_arr = pd.read_csv(f)
            print("shape:", pose_arr.shape)
        except OSError:
            print("data or label ", str_id, "is not exist")
        else:
            print("data has been load ", str_id)


def test_numpy_max():
    y_pre_joint = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])
    y_pre_joint = y_pre_joint.reshape((-1, 2))
    y_pre_joint_max = np.max(y_pre_joint, axis=1)
    print(y_pre_joint)
    print(y_pre_joint_max)


if __name__ == "__main__":
    test_numpy_max()
