# 读取data和label的csv文件，并以一定的格式返回
import numpy as np
import pandas as pd
from log_config import log
import time
import scipy.io as scio
import cv2

train_data_list = [80, 9, 203, 198, 101, 237, 244, 17, 261, 62, 242, 115, 220, 31, 65, 270, 185, 12, 172, 168, 180, 110,
                   150, 336,
                   294, 206, 116, 339, 119, 240, 184, 19, 98, 277, 137, 221, 128, 87, 170, 1, 78, 192, 288, 5, 189, 194,
                   287, 112,
                   122, 103, 274, 2, 120, 205, 15, 307, 164, 284, 36, 282, 304, 276, 81, 278, 285, 281, 318, 211, 230,
                   266, 217, 16,
                   68, 311, 233, 188, 182, 169, 236, 66, 154, 344, 85, 262, 24, 256, 72, 340, 271, 18, 293, 149, 152,
                   249, 207, 298,
                   191, 273, 268, 279, 329, 209, 303, 238, 323, 222, 156, 32, 136, 60, 187, 253, 25, 176, 113, 297, 37,
                   3, 196, 82,
                   159, 229, 13, 147, 105, 342, 286, 343, 138, 96, 160, 56, 324, 126, 50, 171, 38, 202, 314, 106, 94,
                   67, 58, 239,
                   210, 260, 208, 316, 46, 302, 111, 337, 43, 49, 83, 90, 177, 131, 86, 310, 280, 218, 228, 0, 320, 133,
                   291, 61,
                   132, 100, 47, 91, 199, 158, 22, 10, 144, 225, 251, 77, 296, 44, 195, 51, 88, 21, 272, 334, 26, 255,
                   123, 27, 263,
                   55, 163, 118, 226, 175, 254, 312, 257, 20, 248, 332, 79, 162, 148]

test_data_list = [140, 121, 259, 174, 167, 333, 41, 299, 42, 73, 63, 223, 246, 212, 6, 151, 345, 104, 40, 109, 327, 200,
                  28, 258, 135, 232, 267, 326, 141, 45, 57, 305, 75, 338, 231, 30, 153, 264, 215, 309, 54, 317, 295,
                  325, 283, 64, 53, 52, 183, 289, 193, 319, 335, 89, 99, 224, 76, 214, 197, 4, 179, 155, 322, 243, 7,
                  92, 14, 29, 157, 84, 130, 213, 321, 204, 108, 69, 290, 301, 331, 250, 39, 129, 190, 146, 134, 300,
                  216, 241, 93, 95, 275, 306, 227, 313, 166, 127, 292, 219, 107, 142, 315, 330, 145, 186, 71, 102, 114,
                  201, 143, 48, 33, 341, 59, 235, 124, 161, 139, 308, 247, 125, 74, 97, 35, 181, 328, 117, 269, 178,
                  265, 234, 23, 165, 11, 34, 70, 252, 8, 245, 173]


# 读取csv数据，并根据video id随机分成训练数据和测试数据
def read_csv_data_random(data_id: int):
    if data_id == 2:
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
            except OSError:
                print("data or label ", str_id, "is not exist")
            else:
                print("data has been load ", str_id)
        train_norm_pose = normalize_all_point(train_pose)

        # 再读取测试数据集
        test_pose = pd.read_csv(data_path + "data" + str(train_data_list[0]) + ".csv", header=None, sep=',',
                                encoding='utf-8')
        test_label = pd.read_csv(label_path + "label" + str(train_data_list[0]) + ".csv", header=None, sep=',',
                                 encoding='utf-8')
        test_video_length_list = [len(test_label)]
        for str_id in train_data_list[1:]:
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


# test=0:测试用小文件，1:iou all数据，2:中心点检测得到的数据
def read_csv_train_label_data(data_id: int):
    number_oftest = 347
    # 从csv文件中读取
    if data_id == 2:
        data_path = "../halpe26_data/data_by_video/single/"
        label_path = "../halpe26_data/data_by_video/single/"
        single_pose = pd.read_csv(data_path + "data1.csv", header=None, sep=',', encoding='utf-8')
        single_label = pd.read_csv(label_path + "label1.csv", header=None, sep=',', encoding='utf-8')
        video_length_list = [len(single_label)]
        for str_id in range(2, number_oftest):
            try:
                pose_arr = pd.read_csv(data_path + "data" + str(str_id) + ".csv", header=None, sep=',',
                                       encoding='utf-8')
                label_arr = pd.read_csv(label_path + "label" + str(str_id) + ".csv", header=None, sep=',',
                                        encoding='utf-8')
                print("shape:", pose_arr.shape, label_arr.shape)
                video_length_list.append(len(label_arr))
                single_pose = np.concatenate((single_pose, pose_arr), axis=0)
                single_label = np.concatenate((single_label, label_arr), axis=0)
            except OSError:
                print("data or label ", str_id, "is not exist")
            else:
                print("data has been load ", str_id)
        norm_single_pose = normalize_all_point(single_pose)
        return norm_single_pose, single_label, video_length_list
    elif data_id == 3:
        data_path = "halpe26_data/data_by_video/all_single/"
        label_path = "halpe26_data/data_by_video/all_single/"
        stream_pose = pd.read_csv(data_path + "data1.csv", header=None, sep=',', encoding='utf-8').values
        stream_label = pd.read_csv(data_path + "label1.csv", header=None, sep=',', encoding='utf-8').values
        for str_id in range(2, number_oftest):
            try:
                pose_arr = pd.read_csv(data_path + "data" + str(str_id) + ".csv", header=None, sep=',',
                                       encoding='utf-8').values
                label_arr = pd.read_csv(label_path + "label" + str(str_id) + ".csv", header=None, sep=',',
                                        encoding='utf-8').values
                print(str_id, "shape:", pose_arr.shape, label_arr.shape)
                stream_pose = np.concatenate((stream_pose, pose_arr), axis=0)
                stream_label = np.concatenate((stream_label, label_arr), axis=0)
            except OSError:
                print("data or label ", str_id, "is not exist")
            else:
                print("data has been load ", str_id)
        stream_pose = stream_pose[:, 2:stream_pose.shape[1] - 1]  # 原始数据包含视频id，图片id，标签，需要去除
        return stream_pose, stream_label
    elif data_id == 4:
        data_path = "halpe26_data/data_by_video/stream/"
        label_path = "halpe26_data/data_by_video/stream/"
        stream_pose = pd.read_csv(data_path + "data1.csv", header=None, sep=',', encoding='utf-8')
        stream_label = pd.read_csv(label_path + "label1.csv", header=None, sep=',', encoding='utf-8')
        for str_id in range(2, number_oftest):
            try:
                pose_arr = pd.read_csv(data_path + "data" + str(str_id) + ".csv", header=None, sep=',',
                                       encoding='utf-8')
                label_arr = pd.read_csv(label_path + "label" + str(str_id) + ".csv", header=None, sep=',',
                                        encoding='utf-8')
                print(str_id, "shape:", pose_arr.shape, label_arr.shape)
                stream_pose = np.concatenate((stream_pose, pose_arr), axis=0)
                stream_label = np.concatenate((stream_label, label_arr), axis=0)
            except OSError:
                print("data or label ", str_id, "is not exist")
            else:
                print("data has been load ", str_id)
        return stream_pose, stream_label
    else:
        print("读取数据的参数错误，test=0:测试用小文件，1:iou 匹配数据，2:中心点匹配数据")
        return


# 正则化脸部特征点
# 使用0位置（鼻子）作为零点，所有特征点减去该点坐标，分别除以人的box宽和17，18特征点（额头、下巴）之间高度
# [0, 1, 2, 3, 4, 17, 18] 脸部的特征点范围，共7个特征点
def normalize_face_point(__pose_arr: np.array):
    face_range = [0, 1, 2, 3, 4, 17, 18]
    normalize_array = np.zeros((len(__pose_arr), 1))

    box_width = np.max(__pose_arr, axis=1)  # 行人的宽度，shape=(number,1)
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
def normalize_all_point(__pose_arr: np.array):
    for __j in [1, 3, 5, 17]:
        norm_x = __pose_arr[:, __j * 3] - __pose_arr[:, (__j + 1) * 3]
        norm_y = __pose_arr[:, __j * 3 + 1] - __pose_arr[:, (__j + 1) * 3 + 1]
        __pose_arr = np.concatenate((__pose_arr, norm_x.reshape(-1, 1)), axis=1)  # 特征点的y轴值
        __pose_arr = np.concatenate((__pose_arr, norm_y.reshape(-1, 1)), axis=1)
    for __i in range(1, 26):
        __pose_arr[:, __i * 3] -= __pose_arr[:, 0]
        __pose_arr[:, __i * 3 + 1] -= __pose_arr[:, 1]
    # __pose_array = __pose_array[:, [0, 1, 2, 3, 4, 5, 6, 17, 18, 78, 79, 80, 81]]
    return __pose_arr


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
    st = time.time()
    read_csv_train_label_data(data_id=4, output_type=1)
    t1 = time.time()
    print("------------------------------------")
    print("pandas time: ", t1 - st)
    read_csv_train_label_data(data_id=5, output_type=1)
    t2 = time.time()
    print("np time: ", t2 - t1)
