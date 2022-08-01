# 读取data和label的csv文件，并以一定的格式返回
import cv2
import numpy as np
import pandas as pd
import scipy.io as scio

from config import cross_csv, jaad_all_videos_train,jaad_all_videos_val


def read_data_track():
    data_path = cross_csv
    label_path = cross_csv
    # train_pose、test_pose会读取到87列数据
    # train_data_list, test_data_list = random_int_list()

    train_pose, train_label, train_video_length_list = normalize_read(data_path, label_path, jaad_all_videos_train)
    test_pose, test_label, test_video_length_list = normalize_read(data_path, label_path, jaad_all_videos_val)
    train_norm_pose = normalize_all_point(train_pose[:, 4:86])  # 4是特征点开始，82为特征点结束，82:86为box
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
def normalize_all_point(__keypoints_arr: np.array):
    # # # 脸部特征点1-2，3-4，5-6，额头17-18，腿部11-12，13-14，15-16，两两相减
    # for __j in [1, 3, 5, 11, 13, 15, 17]:
    #     norm_x = __keypoints_arr[:, __j * 3] - __keypoints_arr[:, (__j + 1) * 3]
    #     norm_y = __keypoints_arr[:, __j * 3 + 1] - __keypoints_arr[:, (__j + 1) * 3 + 1]  # 特征点的y轴值
    #     __keypoints_arr = np.concatenate((__keypoints_arr, norm_x.reshape(-1, 1)), axis=1)
    #     __keypoints_arr = np.concatenate((__keypoints_arr, norm_y.reshape(-1, 1)), axis=1)
    # 所有特征点再减去0位置鼻子处的特征点(x,y)
    for __i in range(1, 26):
        __keypoints_arr[:, __i * 3] -= __keypoints_arr[:, 0]
        __keypoints_arr[:, __i * 3 + 1] -= __keypoints_arr[:, 1]
    # __pose_array = __pose_array[:, [0, 1, 2, 3, 4, 5, 6, 17, 18, 78, 79, 80, 81]]

    return __keypoints_arr
