# 读取data和label的csv文件，并以一定的格式返回
import cv2
import numpy as np
import pandas as pd
import scipy.io as scio

from config import cross_csv, jaad_all_videos_train, jaad_all_videos_val


def read_data_track():
    data_path = cross_csv
    label_path = cross_csv
    # train_pose、test_pose会读取到87列数据
    # train_data_list, test_data_list = random_int_list()
    l1, l2 = [], []
    for i in range(1, 347):
        if i < 266:
            l1.append(i)
        else:
            l2.append(i)
    train_pose, train_label, train_video_length_list = normalize_read(data_path, label_path, l1)
    test_pose, test_label, test_video_length_list = normalize_read(data_path, label_path, l2)
    # 4是特征点开始，11为第一个腿部特征点，82为特征点结束，82:86为box
    train_norm_pose = normalize_all_point(train_pose[:, 4:86])
    test_norm_pose = normalize_all_point(test_pose[:, 4:86])

    return train_norm_pose, train_label, train_video_length_list, test_norm_pose, test_label, test_video_length_list


# 标准化读取数据集
def normalize_read(data_path: str, label_path: str, _data_list: list):
    # 先初始化向量
    _pose = pd.read_csv(data_path + "data" + str(_data_list[0]) + ".csv", header=None, sep=',', encoding='utf-8')
    _label = pd.read_csv(label_path + "label" + str(_data_list[0]) + ".csv", header=None, sep=',',
                         encoding='utf-8')
    _video_length_list = [len(_label)]
    for v_id in _data_list[1:]:
        try:
            _pose_arr = pd.read_csv(data_path + "data" + str(v_id) + ".csv", header=None, sep=',', encoding='utf-8')
            _label_arr = pd.read_csv(label_path + "label" + str(v_id) + ".csv", header=None, sep=',', encoding='utf-8')
            print("shape:", _pose_arr.shape, _label_arr.shape)
            _pose = np.concatenate((_pose, _pose_arr), axis=0)
            _label = np.concatenate((_label, _label_arr), axis=0)
            _video_length_list.append(len(_label_arr))
        except OSError:
            print("data or label ", v_id, "is not exist")
        else:
            print("data has been load ", v_id)
    return _pose, _label, _video_length_list


# 正则化所有特征点，以0位置（鼻子）作为零点，所有脸部特征点减去该点坐标
def normalize_all_point(_keypoints_arr: np.array):
    # # # 脸部特征点1-2，3-4，5-6，额头17-18，腿部11-12，13-14，15-16，两两相减
    # for __j in [1, 3, 5, 11, 13, 15, 17]:
    #     norm_x = __keypoints_arr[:, __j * 3] - __keypoints_arr[:, (__j + 1) * 3]
    #     norm_y = __keypoints_arr[:, __j * 3 + 1] - __keypoints_arr[:, (__j + 1) * 3 + 1]  # 特征点的y轴值
    #     __keypoints_arr = np.concatenate((__keypoints_arr, norm_x.reshape(-1, 1)), axis=1)
    #     __keypoints_arr = np.concatenate((__keypoints_arr, norm_y.reshape(-1, 1)), axis=1)
    # 所有特征点再减去19位置的特征点(x,y)
    zero_position = 19
    for _i in range(27):
        if _i != zero_position:
            _keypoints_arr[:, _i * 3] -= _keypoints_arr[:, zero_position * 3]
            _keypoints_arr[:, _i * 3 + 1] -= _keypoints_arr[:, zero_position * 3 + 1]
    # angle 5-7,7-9;6-8,8-10;12-14,14-16;11-13,13-15;12-16,11-15;
    # distance:21-23,20-22,21-25,20-24,11-12,13-14,15-16;
    print(len(_keypoints_arr))
    out_put = np.zeros((len(_keypoints_arr), 4))
    # 先获取bound box
    for _i in range(4):
        out_put[:, _i] = _keypoints_arr[:, 78 + _i]
    angle_pair_list = [[5, 7, 7, 9], [6, 8, 8, 10], [6, 10, 5, 9], [11, 13, 13, 15], [12, 14, 14, 16],
                       [12, 16, 11, 15]]
    distance_pair_list = [[21, 23], [20, 22], [21, 25], [20, 24], [11, 12], [13, 14], [15, 16]]
    for _i in range(len(angle_pair_list)):
        angle_pair = angle_pair_list[_i]  # 取出一对直线
        line1 = keypoints_line(_keypoints_arr, angle_pair[0], angle_pair[1])
        line2 = keypoints_line(_keypoints_arr, angle_pair[2], angle_pair[3])
        angle = angle_row_wise_v2(line1, line2)
        out_put = np.concatenate((out_put, angle), axis=1)
    for _i in range(len(distance_pair_list)):
        distance_pair = distance_pair_list[_i]
        dx, dy = keypoints_distance(_keypoints_arr, distance_pair[0], distance_pair[1])
        out_put = np.concatenate((out_put, dx, dy), axis=1)
    return out_put


def keypoints_line(keypoints, position1, position2):
    x = (keypoints[:, position1 * 3] - keypoints[:, position2 * 3]).reshape((-1, 1))
    y = (keypoints[:, position1 * 3 + 1] - keypoints[:, position2 * 3 + 1]).reshape((-1, 1))
    return np.concatenate((x, y), axis=1)


def keypoints_distance(keypoints, position1, position2):
    dx = keypoints[:, position1 * 3] - keypoints[:, position2 * 3]
    dy = keypoints[:, position1 * 3 + 1] - keypoints[:, position2 * 3 + 1]
    return dx.reshape((-1, 1)), dy.reshape((-1, 1))


"""
l1_arr = np.array([[1, 0],
                  [0, 1],
                  [0, 5],
                  [-1, 0]])
l2_arr = np.array([[1, 0],
                  [1, 1],
                  [0, 1],
                  [5, 0]])
result = angle_row_wise_v2(l1_arr, l2_arr)
print(result)
"""


# 计算夹角
def angle_row_wise_v2(l1_arr, l2_arr):
    p1 = np.einsum('ij,ij->i', l1_arr, l2_arr)
    p2 = np.einsum('ij,ij->i', l1_arr, l1_arr)
    p3 = np.einsum('ij,ij->i', l2_arr, l2_arr)
    # p4 = p1 / np.sqrt(p2 * p3)
    a = p1
    b = np.sqrt(p2 * p3)
    p4 = np.divide(a, b, out=np.zeros_like(b), where=b != 0)
    return np.arccos(np.clip(p4, -1.0, 1.0)).reshape((-1, 1))
