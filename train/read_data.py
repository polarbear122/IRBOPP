import numpy as np

from log_config import log


# test=0:测试用小文件，1:iou all数据，2:中心点检测得到的数据
def read_csv_train_label_data(data_id: int, output_type: int):
    number_oftest = 374
    # 从csv文件中读取
    if data_id == 0:
        pose_arr = np.loadtxt("train_data/test_small_train_data.csv", dtype=np.float_, delimiter=',')
        label_arr = np.loadtxt("train_data/test_small_label.csv", dtype=np.float_, delimiter=',')
    elif data_id == 1:
        pose_arr = np.loadtxt("train_data/iou/all_data.csv", dtype=np.float_, delimiter=',')
        label_arr = np.loadtxt("train_data/iou/all_label.csv", dtype=np.float_, delimiter=',')
    elif data_id == 2:
        stream_pose = np.loadtxt("train_data/iou/stream_sub/data0.csv", dtype=np.float_, delimiter=',')
        stream_label = np.loadtxt("train_data/iou/stream_sub/label0.csv", dtype=np.float_, delimiter=',')
        for str_id in range(1, 10, 1):
            pose_arr = np.loadtxt(
                "train_data/iou/stream_sub/data" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
            label_arr = np.loadtxt(
                "train_data/iou/stream_sub/label_nearest" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
            stream_pose = np.concatenate((stream_pose, pose_arr), axis=0)
            stream_label = np.concatenate((stream_label, label_arr), axis=0)
        return stream_pose, stream_label
    elif data_id == 3:
        data_path = "train_data/iou/data_by_video/single/"
        label_path = "train_data/iou/data_by_video/single/"

        single_pose = np.loadtxt(data_path + "data1.csv", dtype=np.float_, delimiter=',')
        single_label = np.loadtxt(label_path + "label1.csv", dtype=np.float_, delimiter=',')
        video_length_list = [len(single_label)]
        for str_id in range(2, number_oftest):
            try:
                pose_arr = np.loadtxt(data_path + "data" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
                label_arr = np.loadtxt(label_path + "label" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
                print("shape:", pose_arr.shape, label_arr.shape)
                video_length_list.append(len(label_arr))
                single_pose = np.concatenate((single_pose, pose_arr), axis=0)
                single_label = np.concatenate((single_label, label_arr), axis=0)
            except OSError:
                print("data or label ", str_id, "is not exist")
            else:
                print("data has been load ", str_id)
        return single_pose, single_label, video_length_list
    elif data_id == 4:
        data_path = "train_data/iou/data_by_video/stream/"
        label_path = "train_data/iou/data_by_video/stream/"
        stream_pose = np.loadtxt(data_path + "data1.csv", dtype=np.float_, delimiter=',')
        stream_label = np.loadtxt(label_path + "label1.csv", dtype=np.float_, delimiter=',')
        for str_id in range(2, number_oftest):
            try:
                pose_arr = np.loadtxt(data_path + "data" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
                label_arr = np.loadtxt(label_path + "label" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
                print(str_id, "shape:", pose_arr.shape, label_arr.shape)
                stream_pose = np.concatenate((stream_pose, pose_arr), axis=0)
                stream_label = np.concatenate((stream_label, label_arr), axis=0)
            except OSError:
                print("data or label ", str_id, "is not exist")
            else:
                print("data has been load ", str_id)
        return stream_pose, stream_label
    elif data_id == 5:
        pose_arr = np.loadtxt("train_data/center/all_data.csv", dtype=np.float_, delimiter=',')
        label_arr = np.loadtxt("train_data/center/all_label.csv", dtype=np.float_, delimiter=',')
    else:
        print("读取数据的参数错误，test=0:测试用小文件，1:iou 匹配数据，2:中心点匹配数据")
        return
    log.logger.info("csv data has been load")
    num_start, num_stop = 80000, 100000  # 99623  # 总共有84141条数据,[start,stop),前开后闭区间
    # return pose_arr[num_start:num_stop, output_range], label_arr[num_start:num_stop]  # 使用逗号进行裁剪维度分割

    if output_type == 0:
        # 单帧姿势
        train_data, label = normalize_face_point(pose_arr), label_arr
        return train_data[num_start:num_stop], label[num_start:num_stop]
    elif output_type == 1:
        # 视频流姿势
        train_data, label = [], []
        for i in range(0, 10, 1):
            st, end = 10000 * i, 10000 * i + 10000
            train_data, label = normalize_face_point_stream(pose_arr[st:end], label_arr[st:end])
            np.savetxt("train_data/iou/data_by_video/stream/data" + str(i) + ".csv", train_data, delimiter=',')
            np.savetxt("train_data/iou/data_by_video/stream/label" + str(i) + ".csv", label, delimiter=',')
        return train_data, label
    else:
        return "未选定输出为视频流姿势或单帧姿势"


# 正则化脸部特征点
# 使用0位置（鼻子）作为零点，所有特征点减去该点坐标，分别除以人的box宽和17，18特征点（额头、下巴）之间高度
# [0, 1, 2, 3, 4, 17, 18] 脸部的特征点范围，共7个特征点
def normalize_face_point(pose_array: np.array):
    face_range = [0, 1, 2, 3, 4, 17, 18]
    normalize_array = np.zeros((len(pose_array), 1))

    box_width = np.max(pose_array, axis=1)  # 行人的宽度，shape=(number,1)
    face_height = pose_array[:, 18 * 3 + 1] - pose_array[:, 17 * 3 + 1]  # 脸部的高度
    face_center_x, face_center_y = pose_array[:, 1], pose_array[:, 2]  # 脸部中心点（鼻子）的坐标，列向量
    for position in face_range:
        sub_x, sub_y = pose_array[:, position] - face_center_x, pose_array[:, position + 1] - face_center_y
        # 如果被除数为0，则将结果置为1
        norm_x = np.divide(sub_x, box_width, out=np.ones_like(sub_x), where=box_width != 0).reshape(-1, 1)
        norm_y = np.divide(sub_y, face_height, out=np.ones_like(sub_y), where=face_height != 0).reshape(-1, 1)

        normalize_array = np.concatenate((normalize_array, norm_x), axis=1)  # 特征点的x轴值
        normalize_array = np.concatenate((normalize_array, norm_y), axis=1)  # 特征点的y轴值
        normalize_array = np.concatenate((normalize_array, pose_array[:, position + 2].reshape(-1, 1)), axis=1)  # 可见性
    return normalize_array[:, 1:]


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


if __name__ == "__main__":
    pass
