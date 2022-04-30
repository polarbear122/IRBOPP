# 提供读取数据的方法
import pickle

import cv2
import numpy as np

from log_config import log
from toolkit.read_pose_data import read_json
from toolkit.xml_read import xml_read, str_to_int


# test=0:测试用小文件，1:iou all数据，2:中心点检测得到的数据
def read_csv_train_label_data(data_id: int, output_type: int):
    # 从csv文件中读取
    if data_id == 0:
        pose_array = np.loadtxt("train_data/test_small_train_data.csv", dtype=np.float_, delimiter=',')
        label_array = np.loadtxt("train_data/test_small_label.csv", dtype=np.float_, delimiter=',')
    elif data_id == 1:
        pose_array = np.loadtxt("train_data/iou_all_train_data.csv", dtype=np.float_, delimiter=',')
        label_array = np.loadtxt("train_data/iou_all_label.csv", dtype=np.float_, delimiter=',')
    elif data_id == 2:
        pose_array = np.loadtxt("train_data/center_point_all_train_data.csv", dtype=np.float_, delimiter=',')
        label_array = np.loadtxt("train_data/center_point_all_label.csv", dtype=np.float_, delimiter=',')
    else:
        print("读取数据的参数错误，test=0:测试用小文件，1:iou 匹配数据，2:中心点匹配数据")
        return
    log.logger.info("csv data has been load")
    num_start, num_stop = 0, 99623  # 总共有84141条数据,[start,stop),前开后闭区间
    # return pose_array[num_start:num_stop, output_range], label_array[num_start:num_stop]  # 使用逗号进行裁剪维度分割

    if output_type == 0:
        # 单帧姿势
        return normalize_face_point(pose_array)[num_start:num_stop], label_array[num_start:num_stop]
    elif output_type == 1:
        # 视频流姿势
        train_data, label = normalize_face_point_stream(pose_array, label_array)
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


# 一秒三十帧，每次输出三十帧为一个视频流，在每行数据后面直接append，标签采用“或”方式相加
def normalize_face_point_stream(pose_array: np.array, labels: np.array):
    f_p_stream, features_len = 6, 21  # f_p_stream每个流中的帧数，features_len特征长度
    face_range = [0, 1, 2, 3, 4, 17, 18]
    norm_array = np.zeros((len(pose_array), 1))  # norm_array 正则化数组
    box_width = np.max(pose_array, axis=1)  # 行人的宽度，shape=(number,1)
    face_height = pose_array[:, 18 * 3 + 1] - pose_array[:, 17 * 3 + 1]  # 脸部的高度
    face_center_x, face_center_y = pose_array[:, 1], pose_array[:, 2]  # 脸部中心点（鼻子）的坐标，列向量
    for position in face_range:
        sub_x, sub_y = pose_array[:, position] - face_center_x, pose_array[:, position + 1] - face_center_y
        # 如果被除数为0，则将结果置为1
        norm_x = np.divide(sub_x, box_width, out=np.ones_like(sub_x), where=box_width != 0).reshape(-1, 1)
        norm_y = np.divide(sub_y, face_height, out=np.ones_like(sub_y), where=face_height != 0).reshape(-1, 1)

        norm_array = np.concatenate((norm_array, norm_x), axis=1)  # 特征点的x轴值
        norm_array = np.concatenate((norm_array, norm_y), axis=1)  # 特征点的y轴值
        norm_array = np.concatenate((norm_array, pose_array[:, position + 2].reshape(-1, 1)), axis=1)  # 可见性

    norm_array = norm_array[:, 1:]  # 1:代表裁剪之前的初始0值
    sample_method = 1
    if sample_method == 0:
        # 1、采用reshape的方式采样，数据量缩减为原来的(1/f_p_stream)
        norm_array = norm_array[:len(norm_array) // f_p_stream * f_p_stream, :]  # 先除后乘,避免无法reshape
        stream_array = norm_array.reshape(-1, f_p_stream * features_len)
        labels = labels[:len(labels) // f_p_stream * f_p_stream]
        labels = labels.reshape(-1, f_p_stream)
        stream_labels = np.amax(labels, axis=1)
        return stream_array, stream_labels
    elif sample_method == 1:
        #  2、采用叠加的方式，不会减少数据量
        for i in range(len(norm_array) - f_p_stream):
            array_30_to_1 = norm_array[i:i + f_p_stream].reshape(1, -1)  # 将f_p_stream帧数据变成一行
            label_30_to_1 = np.amax(labels[i:i + f_p_stream], axis=1)
            stream_array = norm_array.reshape(-1, f_p_stream * features_len)
            stream_labels = np.amax(labels, axis=1)

    else:
        print("error 未选择正则化输出中，图像转视频流的方法")


def get_key_points(keypoints: list):
    # key points [0, 1, 2, 3, 4, 17, 18]
    if len(keypoints) != 78:
        return []
    key_points = []
    # for i in [0, 1, 2, 3, 4, 17, 18]:
    #     key_points.append(keypoints[i * 3 + 2])
    for i in range(26):
        key_points.append(keypoints[i * 3 + 0])
        key_points.append(keypoints[i * 3 + 1])
        key_points.append(keypoints[i * 3 + 2])
    return key_points


# 画出（左上角，宽，高）格式的box
def plot_pose_box_look_b(pose_box, annotation, is_look, video_id):
    output_file = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
    img = cv2.imread(output_file + annotation["@frame"] + ".jpg")
    a, b, c, d = int(pose_box[0]), int(pose_box[1]), int(pose_box[2]), int(pose_box[3])
    cv2.line(img, (a, b), (a + c, b), (0, 0, 255), thickness=2)
    cv2.line(img, (a, b), (a, b + d), (0, 0, 255), thickness=2)
    cv2.line(img, (a + c, b), (a + c, b + d), (0, 0, 255), thickness=2)
    cv2.line(img, (a, b + d), (a + c, b + d), (0, 0, 255), thickness=2)
    cv2.putText(img, is_look, (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
    img = cv2.resize(img, (1920 // 2, 1080 // 2))
    cv2.imshow("./pose box looking", img)
    cv2.waitKey(1)


# 画出（左上角，右下角）格式的box
def plot_pose_box_look(pose_box, annotation, is_look, video_id):
    output_file = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
    img = cv2.imread(output_file + annotation["@frame"] + ".jpg")
    xtl, ytl, xbr, ybr = int(pose_box[0]), int(pose_box[1]), int(pose_box[2]), int(pose_box[3])
    cv2.line(img, (xbr, ytl), (xtl, ytl), (0, 0, 255), thickness=2)
    cv2.line(img, (xbr, ytl), (xbr, ybr), (0, 0, 255), thickness=2)
    cv2.line(img, (xbr, ybr), (xtl, ybr), (0, 0, 255), thickness=2)
    cv2.line(img, (xtl, ybr), (xtl, ytl), (0, 0, 255), thickness=2)
    x_mid, y_mid = (xbr + xtl) // 2, (ybr + ytl) // 2
    # cv2.putText(img, is_look, (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
    cv2.putText(img, str(pose_box[0]), (x_mid, y_mid + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
    img = cv2.resize(img, (1920 // 2, 1080 // 2))
    cv2.imshow("./pose box looking", img)
    cv2.waitKey(1)


def box_iou(box1, box2):
    # 计算box a和box b的IOU值,输入左上角坐标，右下角坐标, box:[x1, y1, x2, y2]
    # 例如box1 = [0,0,10,10], box2 = [5,5,15,15]
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inner = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inner
    if union == 0.0:
        return 0.0
    iou = inner / union
    return iou


def get_train_data(jaad_anno_path, alpha_pose_path, video_id):
    x = []  # key points 0 1 2 3 4 17 18
    y = []  # label looking not-looking
    alpha_pose = read_json(alpha_pose_path)
    jaad_anno = xml_read(jaad_anno_path)
    annotations = jaad_anno["annotations"]
    if "track" not in annotations:
        return np.mat(x), np.mat(y).T
    if "box" in annotations["track"]:
        track_box = [annotations["track"]]
    else:
        track_box = annotations["track"]

    is_repeat = 0
    for i in track_box:
        if i["@label"] != "pedestrian":
            continue
        is_repeat += 1
        if is_repeat >= 2:
            print(jaad_anno_path, "there are two box in one video")
        for annotation in i["box"]:
            # jaad注释文件，左上角，右下角 (top-left, bottom-right), ybr>xbr,ytl>xtl
            xtl, ytl = str_to_int(annotation["@xtl"]), str_to_int(annotation["@ytl"])
            xbr, ybr = str_to_int(annotation["@xbr"]), str_to_int(annotation["@ybr"])
            # x_mid, y_mid = (xtl + xbr) // 2, (ytl + ybr) // 2

            max_iou = 0.6
            pose_box = []  # alpha pose的box位置,json文件中为([0],[1])左上角,([2],[3])宽和高,修改成(左上角,右下角)格式
            x_keypoints_proposal = []  # 存储key points
            for pose in alpha_pose:
                if pose["score"] < 1:
                    continue
                if pose["image_id"] == annotation["@frame"] + ".jpg":
                    pose_box = [pose["box"][0], pose["box"][1], pose["box"][0] + pose["box"][2],
                                pose["box"][1] + pose["box"][3]]
                    true_box = [xtl, ytl, xbr, ybr]
                    iou_val = box_iou(pose_box, true_box)
                    if iou_val > max_iou:
                        x_keypoints_proposal = get_key_points(pose["keypoints"])
                        max_iou = iou_val
                elif pose["image_id"] == str(int(annotation["@frame"]) + 1) + ".jpg":
                    break
            is_look = annotation["attribute"][2]["#text"]
            if x_keypoints_proposal:
                x.append(x_keypoints_proposal)
                if is_look == "looking":
                    y.append(1)
                else:
                    y.append(0)
                need_plot = False
                if need_plot and pose_box and max_iou > 0.6:
                    plot_pose_box_look(pose_box, annotation, is_look, video_id)

    print(video_id, "shape:", np.mat(x).shape, np.mat(y).T.shape)
    return np.mat(x), np.mat(y).T


def get_init_data():
    train_data_shape = 78  # 训练的数据的列的大小为7，总训练的数据格式为(number_of_data,train_data_shape)
    train_dataset, labels = np.zeros((1, train_data_shape), float), np.zeros((1, 1), float)
    for i in range(1, 347):
        video_id = "video_" + str(i).zfill(4)
        xml_anno_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
        alpha_pose_path = "E:/CodeResp/pycode/DataSet/pose_result/" + video_id + "/alphapose-results.json"
        x, y = get_train_data(xml_anno_path, alpha_pose_path, video_id)
        if x.shape[1] == train_data_shape:
            train_dataset = np.concatenate((train_dataset, x))
            labels = np.concatenate((labels, y))
    print("all data saved, shape:", train_dataset.shape, labels.shape, "true shape:", train_data_shape)
    train_dataset = np.asarray(train_dataset)
    labels = np.asarray(labels)
    np.savetxt("train_data/test_all_train_data.csv", train_dataset, delimiter=',')
    np.savetxt("train_data/test_all_label.csv", labels, delimiter=',')
    return train_dataset, labels


def save_model(file_path, file_name, model):
    with open(file=file_path + file_name, mode="wb+") as f:
        f.write(model)


def load_model(file_path):
    with open(file=file_path, mode="wb+") as trained_model:
        s2 = trained_model.read()
        model = pickle.loads(s2)
    # expected = test_y
    # predicted = model1.predict(test_X)
    return model


if __name__ == "__main__":
    get_init_data()
    box1 = [1755.9654541015625, 515.1502075195312, 1914.33056640625, 948.6547241210938]
    box2 = [444, 702, 505, 869]
    print(box_iou(box1, box2))
