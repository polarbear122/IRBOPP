import random

import cv2
import numpy as np
import pandas as pd
import torch
import os
import config
import train.test_joint_image_video as jo


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


# 测试读取csv数据
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


# 求numpy最大值
def test_numpy_max():
    y_pre_joint = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])
    y_pre_joint = y_pre_joint.reshape((-1, 2))
    y_pre_joint_max = np.max(y_pre_joint, axis=1)
    print(y_pre_joint)
    print(y_pre_joint_max)


# 测试numpy的排序功能
def test_np_sort():
    data_path = "train/halpe26_reid/"
    train_pose = pd.read_csv(data_path + "data1.csv", header=None, sep=',', encoding='utf-8').values
    a = train_pose
    a = a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
    print(a)
    return a


# 生成随机数组
def random_int_list():
    all_list = config.all_data_list
    train_list = []
    for i in range(242):
        j = random.randint(0, len(all_list) - 1)
        train_list.append(all_list[j])
        del all_list[j]
    test_list = all_list
    print("train:", train_list)
    print("test:", test_list)
    return train_list, test_list


def get_result():
    path = 'D:/CodeResp/jaad_data/AlphaReidResultNoFast/video_'
    new_path = 'D:/CodeResp/jaad_data/new/AlphaReidResultNoFast/video_'
    for i in range(1, 347):
        i_name = str(i).zfill(4)
        path_name = path + i_name
        new_path_name = new_path + i_name
        file_name = path_name + '/alphapose-results.json'
        new_file_name = new_path_name + '/alphapose-results.json'
        if not os.path.exists(new_path_name):
            os.mkdir(new_path_name)
        print(file_name, new_file_name)
        # shutil.copy(file_name, new_file_name)


# 校验随机的训练和测试数组是合适的
def test_train_test_list():
    train, test, val = config.jaad_all_videos_train, config.jaad_all_videos_test, config.jaad_all_videos_val
    for i in range(1, 347):
        if i in train and i in test or i in train and i in val or i in test and i in val:
            print("error")
            break
        if i not in train and i not in test and i not in val:
            print(i)
            print("error:not in")
            break


# 测试numpy数组大小能否重新初始化
def numpy_arr_reshape():
    num1 = np.zeros(2)
    num2 = np.zeros(2)
    num1 = np.concatenate((num1, num2))
    print(num1)
    num1 = np.zeros((1, 2))
    print(num1)


# 从jaad数据集中解析出train、val、test数据
def get_txt_file_to_num():
    # file_path = "D:/CodeResp/JAAD-JAAD_2.0/split_ids/all_videos/"
    file_path = "D:/CodeResp/jaad_data/JAAD-JAAD_2.0/split_ids/all_videos/"
    name = "test.txt"
    file = file_path + name
    num_list = []
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            if line:
                num = int(line.split("_")[1])
                num_list.append(num)
    print("len test", len(num_list))
    print("test_list = ", num_list)


def angle_row_wise_v2(l1_arr, l2_arr):
    p1 = np.einsum('ij,ij->i', l1_arr, l2_arr)
    p2 = np.einsum('ij,ij->i', l1_arr, l1_arr)
    p3 = np.einsum('ij,ij->i', l2_arr, l2_arr)
    p4 = p1 / np.sqrt(p2 * p3)
    return np.arccos(np.clip(p4, -1.0, 1.0))


# train len 188
train_list = [1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 24, 25, 26, 27, 30, 31, 33, 34, 35, 37, 38, 39, 47, 49,
              50, 51, 52, 54, 56, 57, 60, 61, 62, 64, 66, 69, 74, 77, 78, 79, 80, 81, 83, 85, 86, 88, 91, 94, 95, 98,
              108, 109, 111, 112, 114, 119, 120, 121, 122, 126, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140,
              142, 143, 145, 146, 147, 149, 154, 157, 158, 159, 161, 166, 167, 168, 169, 171, 174, 175, 176, 180, 182,
              184, 185, 186, 188, 189, 190, 191, 192, 194, 195, 196, 198, 200, 202, 204, 205, 207, 208, 209, 210, 214,
              215, 218, 219, 220, 225, 227, 228, 229, 231, 232, 233, 235, 236, 237, 240, 241, 242, 246, 247, 248, 249,
              250, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 266, 268, 269, 272, 275, 276, 281, 282, 283, 284,
              286, 289, 290, 293, 296, 297, 298, 301, 302, 310, 311, 312, 315, 317, 318, 319, 320, 321, 323, 324, 325,
              326, 328, 331, 335, 341, 342, 345, 346]
# test len 126
test_list = [5, 15, 16, 17, 22, 23, 28, 29, 32, 36, 42, 43, 45, 46, 48, 53, 55, 58, 59, 63, 67, 68, 70, 71, 75, 76, 84,
             87, 90, 92, 93, 96, 97, 100, 101, 103, 104, 105, 106, 107, 110, 113, 115, 116, 117, 118, 124, 125, 127,
             128, 135, 141, 144, 148, 150, 151, 152, 153, 155, 162, 163, 164, 165, 173, 177, 178, 179, 183, 187, 197,
             201, 203, 206, 211, 212, 213, 216, 221, 222, 223, 224, 230, 234, 238, 239, 243, 244, 245, 251, 253, 265,
             267, 270, 271, 277, 278, 279, 280, 285, 287, 288, 292, 294, 295, 299, 300, 304, 305, 307, 308, 309, 313,
             314, 316, 322, 327, 329, 330, 332, 333, 334, 336, 337, 338, 339, 344]

# val len 32
val_list = [2, 6, 21, 40, 41, 44, 65, 72, 73, 82, 89, 99, 102, 123, 156, 160, 170, 172, 181, 193, 199, 217, 226, 252,
            263, 273, 274, 291, 303, 306, 340, 343]

if __name__ == "__main__":
    get_txt_file_to_num()
