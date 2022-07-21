# 读取data和label的csv文件，并以一定的格式返回
import cv2
import numpy as np
import pandas as pd
import scipy.io as scio

import config
from config import csv_data, train_data_list, test_data_list, all_data_list


# 读取有track的alpha pose的csv数据，带有idx
# 但是没有转换为stream姿势流的处理过程，依然是单帧数据
def read_data_track():
    data_path = csv_data
    label_path = csv_data
    # train_pose、test_pose会读取到87列数据
    train_pose, train_label, train_video_length_list = normalize_read(data_path, label_path, train_data_list)
    test_pose, test_label, test_video_length_list = normalize_read(data_path, label_path, test_data_list)
    train_norm_pose = normalize_all_point(train_pose[:, 4:86])  # 4是特征点开始，82为特征点结束，82:86为box
    test_norm_pose = normalize_all_point(test_pose[:, 4:86])
    return train_norm_pose, train_label, train_video_length_list, test_norm_pose, test_label, test_video_length_list


# 测试时读取所有数据，不区分训练和测试
def read_data_track_test():
    data_path = csv_data
    label_path = csv_data
    # train_pose会读取到87列数据
    all_pose, all_label, all_video_length_list = normalize_read(data_path, label_path, all_data_list)
    all_norm_pose = normalize_all_point(all_pose[:, 4:86])  # 4是特征点开始，82为特征点结束，82:86为box
    return all_norm_pose, all_label, all_video_length_list


# 数据转换成stream后导出
def read_data_stream():
    data_path = csv_data
    label_path = csv_data
    # train_pose、test_pose会读取到87列数据
    train_pose, train_label, train_video_length_list = normalize_read(data_path, label_path, train_data_list)
    test_pose, test_label, test_video_length_list = normalize_read(data_path, label_path, test_data_list)
    train_norm_pose = norm_points_to_stream(train_pose)
    test_norm_pose = norm_points_to_stream(test_pose)
    return train_norm_pose, train_label, train_video_length_list, test_norm_pose, test_label, test_video_length_list


# 测试时读取所有数据，不区分训练和测试
def read_data_stream_test():
    data_path = csv_data
    label_path = csv_data
    all_pose, all_label, all_video_length_list = normalize_read(data_path, label_path, all_data_list)
    all_norm_pose = norm_points_to_stream(all_pose)  # 4是特征点开始，82为特征点结束，82:86为box
    return all_norm_pose, all_label, all_video_length_list


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


# 将读取到的单帧csv数据转化为流数据
def norm_points_to_stream(_pose_data: np.array):
    # 一次叠加5帧姿势
    uuid_arr, v_id_arr, idx_arr, img_id_arr = _pose_data[:, 0], _pose_data[:, 1], _pose_data[:, 2], _pose_data[:, 3]
    half_top_position_list = []
    for i in config.half_top_position:
        half_top_position_list.append(3 * i + 4)
        half_top_position_list.append(3 * i + 1 + 4)
        half_top_position_list.append(3 * i + 2 + 4)
    poses_arr = _pose_data[:, half_top_position_list + [0, 1, 2, 3, 82, 83, 84, 85]]  # 广义范围内的特征点
    poses_arr = normalize_all_point(_pose_data[:, 4:86])
    pose_point_arr = _pose_data[:, 4:82]  # 狭义范围内的特征点
    box_arr = _pose_data[:, 82:86]
    label_arr = _pose_data[:, 86]
    pose_norm_stream = []
    for u in range(len(_pose_data)):
        # 往前追溯5帧
        pose_concat = poses_arr[u]
        for i in range(4):
            # 如果视频id不正确，或第u帧之前无图像，或者前面i帧的idx和第u帧的idx不一致，都只添加0矩阵
            if v_id_arr[u - i - 1] != v_id_arr[u] or u - i - 1 <= 0 or idx_arr[u - i - 1] != idx_arr[u]:
                pose_temp = np.zeros((len(poses_arr[u]),))
            else:
                pose_temp = poses_arr[u - i - 1]
            pose_concat = np.concatenate((pose_concat, pose_temp), axis=0)
        pose_norm_stream.append(pose_concat)
    pose_norm_stream_mat = np.asarray(pose_norm_stream)
    return pose_norm_stream_mat


# 读取无track的alpha pose的csv数据，并根据video id随机分成训练数据和测试数据
def read_data_no_track():
    data_path = "halpe26_data/data_by_video/single/"
    label_path = "halpe26_data/data_by_video/single/"
    # 先读取训练数据集
    train_pose = pd.read_csv(data_path + "data" + str(train_data_list[0]) + ".csv", header=None, sep=',',
                             encoding='utf-8')
    train_label = pd.read_csv(label_path + "label" + str(train_data_list[0]) + ".csv", header=None, sep=',',
                              encoding='utf-8')
    train_video_length_list = [len(train_label)]
    for str_id in train_data_list[1:]:
        try:
            train_pose_arr = pd.read_csv(data_path + "data" + str(str_id) + ".csv", header=None, sep=',',
                                         encoding='utf-8')
            train_label_arr = pd.read_csv(label_path + "label" + str(str_id) + ".csv", header=None, sep=',',
                                          encoding='utf-8')
            print("shape:", train_pose_arr.shape, train_label_arr.shape)
            train_video_length_list.append(len(train_label_arr))
            train_pose = np.concatenate((train_pose, train_pose_arr), axis=0)
            train_label = np.concatenate((train_label, train_label_arr), axis=0)
        except FileNotFoundError:
            print("data or label ", str_id, "is not exist")
        else:
            print("data has been load ", str_id)
    train_norm_pose = normalize_all_point(train_pose)

    # 再读取测试数据集
    test_pose = pd.read_csv(data_path + "data" + str(test_data_list[0]) + ".csv", header=None, sep=',',
                            encoding='utf-8')
    test_label = pd.read_csv(label_path + "label" + str(test_data_list[0]) + ".csv", header=None, sep=',',
                             encoding='utf-8')
    test_video_length_list = [len(test_label)]
    for str_id in test_data_list[1:]:
        try:
            test_pose_arr = pd.read_csv(data_path + "data" + str(str_id) + ".csv", header=None, sep=',',
                                        encoding='utf-8')
            test_label_arr = pd.read_csv(label_path + "label" + str(str_id) + ".csv", header=None, sep=',',
                                         encoding='utf-8')
            print("shape:", test_pose_arr.shape, test_label_arr.shape)
            test_video_length_list.append(len(test_label_arr))
            test_pose = np.concatenate((test_pose, test_pose_arr), axis=0)
            test_label = np.concatenate((test_label, test_label_arr), axis=0)
        except OSError:
            print("data or label ", str_id, "is not exist")
        else:
            print("data has been load ", str_id)
    test_norm_pose = normalize_all_point(test_pose)
    return train_norm_pose, train_label, train_video_length_list, test_norm_pose, test_label, test_video_length_list


# 正则化脸部特征点
# 使用0位置（鼻子）作为零点，所有特征点减去该点坐标，分别除以人的box宽和17，18特征点（额头、下巴）之间高度
# [0, 1, 2, 3, 4, 17, 18] 脸部的特征点范围，共7个特征点
def normalize_face_point(__pose_arr: np.array):
    face_range = [0, 1, 2, 3, 4, 17, 18]
    normalize_array = np.zeros((len(__pose_arr), 1))

    box_width = np.max(__pose_arr, axis=1) - np.min(__pose_arr, axis=1)  # 行人的宽度，shape=(number,1)
    face_height = __pose_arr[:, 18 * 3 + 1] - __pose_arr[:, 17 * 3 + 1]  # 脸部的高度
    face_center_x, face_center_y = __pose_arr[:, 1], __pose_arr[:, 2]  # 脸部中心点（鼻子）的坐标，列向量
    for position in face_range:
        sub_x, sub_y = __pose_arr[:, position] - face_center_x, __pose_arr[:, position + 1] - face_center_y
        # 如果被除数为0，则将结果置为1
        norm_x = np.divide(sub_x, box_width, out=np.ones_like(sub_x), where=box_width != 0).reshape(-1, 1)
        norm_y = np.divide(sub_y, face_height, out=np.ones_like(sub_y), where=face_height != 0).reshape(-1, 1)

        normalize_array = np.concatenate((normalize_array, norm_x), axis=1)  # 特征点的x轴值
        normalize_array = np.concatenate((normalize_array, norm_y), axis=1)  # 特征点的y轴值
        normalize_array = np.concatenate((normalize_array, __pose_arr[:, position + 2].reshape(-1, 1)), axis=1)  # 可见性
    return normalize_array[:, 1:]


# 正则化所有特征点，以0位置（鼻子）作为零点，所有脸部特征点减去该点坐标
def normalize_all_point(__keypoints_arr: np.array):
    # 脸部特征点1-2，3-4，5-6，额头17-18，腿部11-12，13-14，15-16，两两相减
    for __j in [1, 3, 5, 11, 13, 15, 17]:
        norm_x = __keypoints_arr[:, __j * 3] - __keypoints_arr[:, (__j + 1) * 3]
        norm_y = __keypoints_arr[:, __j * 3 + 1] - __keypoints_arr[:, (__j + 1) * 3 + 1]  # 特征点的y轴值
        __keypoints_arr = np.concatenate((__keypoints_arr, norm_x.reshape(-1, 1)), axis=1)
        __keypoints_arr = np.concatenate((__keypoints_arr, norm_y.reshape(-1, 1)), axis=1)
    # 所有特征点再减去0位置鼻子处的特征点(x,y)
    for __i in range(1, 26):
        __keypoints_arr[:, __i * 3] -= __keypoints_arr[:, 0]
        __keypoints_arr[:, __i * 3 + 1] -= __keypoints_arr[:, 1]
    # __pose_array = __pose_array[:, [0, 1, 2, 3, 4, 5, 6, 17, 18, 78, 79, 80, 81]]

    return __keypoints_arr


# 一秒三十帧，每次输出f_p_stream帧为一个视频流，在每行数据后面直接append，标签采用“或”方式相加
def normalize_face_point_stream(pose_array: np.array, labels: np.array):
    f_p_stream, features_len = 6, 21  # f_p_stream每个流中的帧数，features_len特征长度
    face_range = [0, 1, 2, 3, 4, 17, 18]
    norm_array = np.zeros((len(pose_array), 1))  # norm_array 正则化数组
    box_width = np.max(pose_array, axis=1)  # 行人的宽度，shape=(number,1)
    face_height = pose_array[:, 18 * 3 + 1] - pose_array[:, 17 * 3 + 1]  # 脸部的高度
    face_center_x, face_center_y = pose_array[:, 0], pose_array[:, 1]  # 脸部中心点（鼻子）的坐标，列向量
    for position in face_range:
        sub_x, sub_y = pose_array[:, position * 3] - face_center_x, pose_array[:, position * 3 + 1] - face_center_y
        # 如果被除数为0，则将结果置为1
        norm_x = np.divide(sub_x, box_width, out=np.ones_like(sub_x), where=box_width != 0).reshape(-1, 1)
        norm_y = np.divide(sub_y, face_height, out=np.ones_like(sub_y), where=face_height != 0).reshape(-1, 1)

        norm_array = np.concatenate((norm_array, norm_x), axis=1)  # 特征点的x轴值
        norm_array = np.concatenate((norm_array, norm_y), axis=1)  # 特征点的y轴值
        norm_array = np.concatenate((norm_array, pose_array[:, position * 3 + 2].reshape(-1, 1)), axis=1)  # 可见性

    norm_array = norm_array[:, 1:]  # 1:代表裁剪之前的初始0值
    stream_array, stream_labels = np.zeros((1, f_p_stream * features_len)), np.zeros((1, 1))
    sample_method = 1
    if sample_method == 0:
        # 1、采用reshape的方式采样，数据量缩减为原来的(1/f_p_stream)
        norm_array = norm_array[:len(norm_array) // f_p_stream * f_p_stream, :]  # 先除后乘,避免无法reshape
        stream_array = norm_array.reshape(-1, f_p_stream * features_len)
        labels = labels[:len(labels) // f_p_stream * f_p_stream]
        labels = labels.reshape(-1, f_p_stream)
        stream_labels = np.amax(labels, axis=1)
    elif sample_method == 1:
        #  2、采用叠加的方式,数据叠加到同一行,不会减少数据量
        for i in range(len(norm_array) - f_p_stream):
            array_to_1 = norm_array[i:i + f_p_stream].reshape(1, -1)  # 将f_p_stream帧数据变成一行
            stream_array = np.concatenate((stream_array, array_to_1), axis=0)
            label_to_1 = np.array([[labels[i + f_p_stream - 1]]])  # 找出该f_p_stream帧数据中最大值/或提取最近的值
            stream_labels = np.concatenate((stream_labels, label_to_1), axis=0)
            if i % 1000 == 1:
                print(i)
        # stream_labels = stream_labels.reshape(1, -1).ravel()
    else:
        print("error 未选择正则化输出中，图像转视频流的方法")
    return stream_array, stream_labels


def mat_img_read():
    mat_img = 'E:/CodeResp/pycode/DataSet/Supplementary Materials/data/PedestrianImageRecognitionData_Standing_P1.mat'
    data = scio.loadmat(mat_img)
    print(data)
    store_video = data["STOREVIDEO"]
    img_file = "from_mat"
    i = 0
    for img in store_video:  # numpy_images.shape[0]==152
        cv2.imwrite(img_file + "/" + str(i) + ".jpg", img)
        i += 1


if __name__ == "__main__":
    read_data_track()
