# 读取data和label的csv文件，并以一定的格式返回
import os.path

import numpy as np
import pandas as pd

from config import cross_csv


def read_data_track():
    data_path = cross_csv
    label_path = cross_csv
    # train_pose、test_pose会读取到87列数据

    l1, l2 = [], []
    for i in range(1, 347):
        if i < 266:
            l1.append(i)
        else:
            l2.append(i)
    # l1, l2 = jaad_all_videos_train, jaad_all_videos_val
    train_pose, train_label, train_video_length_list = normalize_read(data_path, label_path, l1)
    test_pose, test_label, test_video_length_list = normalize_read(data_path, label_path, l2)
    # 4是特征点开始，11为第一个腿部特征点，82为特征点结束，82:86为box,87和88是look以及车速
    train_norm_pose = normalize_all_point(train_pose[:, 4:88])
    test_norm_pose = normalize_all_point(test_pose[:, 4:88])

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
    out_put = np.zeros((len(_keypoints_arr), 6))
    # 先获取bound box
    for _i in range(6):
        if _i % 2 == 0:
            out_put[:, _i] = _keypoints_arr[:, 78 + _i] / 1080
        else:
            out_put[:, _i] = _keypoints_arr[:, 78 + _i] / 1920
    # for _i in range(78):
    #     out_put = np.concatenate((out_put, _keypoints_arr[:, _i].reshape((-1, 1))), axis=1)
    # look
    look_label = _keypoints_arr[:, -2].reshape((-1, 1))
    out_put = np.concatenate((out_put, look_label), axis=1)
    # vehicle behaviour
    v_behaviour = _keypoints_arr[:, -1].reshape((-1, 1))
    out_put = np.concatenate((out_put, v_behaviour), axis=1)
    # 点乘
    dot = look_label * v_behaviour.reshape((-1, 1))
    out_put = np.concatenate((out_put, dot), axis=1)
    angle_pair_list = [[5, 7, 7, 9], [6, 8, 8, 10], [6, 10, 5, 9], [11, 13, 13, 15], [12, 14, 14, 16],
                       [12, 16, 11, 15]]
    distance_pair_list = [[21, 23], [20, 22], [21, 25], [20, 24], [11, 12], [13, 14], [15, 16]]
    norm_line = np.zeros((len(_keypoints_arr), 2))
    norm_line[:, 0] = 1
    for _i in range(len(angle_pair_list)):
        angle_pair = angle_pair_list[_i]  # 取出一对直线
        line1 = keypoints_line(_keypoints_arr, angle_pair[0], angle_pair[1])
        angle1 = angle_row_wise_v2(line1, norm_line)
        line2 = keypoints_line(_keypoints_arr, angle_pair[2], angle_pair[3])
        angle2 = angle_row_wise_v2(line2, norm_line)
        angle = angle_row_wise_v2(line1, line2)
        out_put = np.concatenate((out_put, angle1, angle2, angle), axis=1)
    for _i in range(len(distance_pair_list)):
        distance_pair = distance_pair_list[_i]
        dx, dy = keypoints_distance(_keypoints_arr, distance_pair[0], distance_pair[1])
        out_put = np.concatenate((out_put, dx, dy), axis=1)

    return out_put


# 数据转换成stream后导出
def read_data_stream():
    data_path = cross_csv
    label_path = cross_csv
    l1, l2 = [], []
    for i in range(1, 347):
        if i < 266:
            l1.append(i)
        else:
            l2.append(i)
    # l1, l2 = jaad_all_videos_train, jaad_all_videos_val
    # train_pose、test_pose会读取到列数据
    train_pose, train_label, train_video_length_list = normalize_read(data_path, label_path, l1)
    test_pose, test_label, test_video_length_list = normalize_read(data_path, label_path, l2)
    print("pre_train_pose:shape", train_pose.shape)
    train_norm_pose = normalize_all_point(train_pose[:, 4:88])
    test_norm_pose = normalize_all_point(test_pose[:, 4:88])
    train_norm_pose = norm_points_to_stream(train_norm_pose)
    test_norm_pose = norm_points_to_stream(test_norm_pose)
    return train_norm_pose, train_label, train_video_length_list, test_norm_pose, test_label, test_video_length_list


# 将读取到的单帧csv数据转化为流数据
def norm_points_to_stream(_pose_data: np.array):
    # 一次叠加5帧姿势
    uuid_arr, v_id_arr, idx_arr, img_id_arr = _pose_data[:, 0], _pose_data[:, 1], _pose_data[:, 2], _pose_data[:, 3]
    poses_arr = normalize_all_point(_pose_data[:, 4:86])  # 截取特征点和box
    # pose_point_arr = _pose_data[:, 4:82]  # 狭义范围内的特征点
    # box_arr = _pose_data[:, 82:86]
    # label_arr = _pose_data[:, 86]
    pose_norm_stream = []
    for u in range(len(_pose_data)):
        # 往前追溯30帧
        pose_concat = poses_arr[u]
        for i in range(10):
            pre = u - i * 1 - 1  # 之前的帧，选择抽取一秒内的5帧，即30帧抽取5帧
            # 如果视频id不正确，或第u帧之前无图像，或者前面i帧的idx和第u帧的idx不一致，都只添加0矩阵
            # if v_id_arr[pre] != v_id_arr[u] or pre <= 0 or idx_arr[pre] != idx_arr[u]:
            if v_id_arr[pre] != v_id_arr[u] or pre <= 0:
                pose_temp = np.zeros((len(poses_arr[u]),))
            else:
                pose_temp = poses_arr[pre]
            pose_concat = np.concatenate((pose_concat, pose_temp), axis=0)
        pose_norm_stream.append(pose_concat)
    pose_norm_stream_mat = np.asarray(pose_norm_stream)
    return pose_norm_stream_mat


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


def convert_float_to_int():
    root = 'cross/data/'
    for i in range(1, 347):
        str_i = str(i)
        if not os.path.exists(root + 'data' + str_i + '.csv'):
            continue
        _pose = pd.read_csv(root + 'data' + str_i + '.csv', header=None, sep=',', encoding='utf-8')
        _label = pd.read_csv(root + 'label' + str_i + '.csv', header=None, sep=',', encoding='utf-8')
        np.savetxt('cross/new_data/data' + str(i) + ".csv", _pose, delimiter=',', fmt='%.3f')
        np.savetxt('cross/new_data/label' + str(i) + ".csv", _label, delimiter=',', fmt='%d')


if __name__ == '__main__':
    convert_float_to_int()
