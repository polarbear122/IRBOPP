"""
读取alpha pose的结果，在训练阶段，将json数据转换成csv数据，存储一包含video_id，idx，img_id，keypoints，box，label的向量。
以video_id，idx，img_id的顺序排序
"""

import csv

import numpy as np
# 提供读取数据的方法
# 从alpha pose的检测结果和jaad的注释文件中读取keypoints和对应img id，保存结果为csv文件
import pandas as pd

import config
from toolkit.plot_data import plot_pose_box_look
from toolkit.read_data import simple_normalize_read
from toolkit.read_pose_data import read_json
from toolkit.tool import xml_read, str_to_int


def get_key_points(keypoints: list):
    # key points [0, 1, 2, 3, 4, 17, 18]
    if len(keypoints) != 26 * 3 and len(keypoints) != 17 * 3:
        print("the len of keypoints is not 26*3 or 17*3")
        return []
    key_points = []
    # for i in [0, 1, 2, 3, 4, 17, 18]:
    #     key_points.append(keypoints[i * 3 + 2])
    for i in range(len(keypoints) // 3):
        key_points.append(keypoints[i * 3 + 0])
        key_points.append(keypoints[i * 3 + 1])
        key_points.append(keypoints[i * 3 + 2])
    return key_points


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


def get_train_data(jaad_anno_path, alpha_pose_path, video_id, int_video_id, uuid):
    x = []
    alpha_pose = read_json(alpha_pose_path)
    jaad_anno = xml_read(jaad_anno_path)
    annotations = jaad_anno["annotations"]
    if "track" not in annotations:
        return np.mat(x), []
    if "box" in annotations["track"]:
        track_box = [annotations["track"]]
    else:
        track_box = annotations["track"]
    is_repeat = 0
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式 cap_fps, size = 30, (1920,1080)
    # size（width，height）
    # video_pose_box = cv2.VideoWriter("train_data/iou/data_by_video-pose-box" + str(
    # video_id) + ".mp4", fourcc, cap_fps, size)
    id_uuid_list = []

    for i in track_box:
        if i["@label"] != "pedestrian":
            continue
        is_repeat += 1
        for annotation in i["box"]:
            # jaad 注释文件，左上角，右下角 (top-left, bottom-right), ybr>xbr,ytl>xtl
            xtl, ytl = str_to_int(annotation["@xtl"]), str_to_int(annotation["@ytl"])
            xbr, ybr = str_to_int(annotation["@xbr"]), str_to_int(annotation["@ybr"])
            # x_mid, y_mid = (xtl + xbr) // 2, (ytl + ybr) // 2
            if xtl <= 0 or ytl <= 0 or (xbr - xtl) <= 0 or (ybr - ytl) <= 0:
                # print(annotation)
                continue
            max_iou = max_iou_threshold = 0.6
            pose_box = []  # alpha pose的box位置,格式为([0],[1])左上角,([2],[3])宽和高,修改成(左上角,右下角)格式
            x_keypoints_proposal, max_pose_box = [], []  # 存储key points,max_pose_box为iou最大时的box（左上角，宽高）格式
            img_frame_id, idx = 0, 0
            plot_max_box = []
            for pose in alpha_pose:
                if pose["score"] < 1:
                    continue
                if pose["image_id"] == annotation["@frame"] + ".jpg":
                    idx = pose["idx"]
                    pose_box = [pose["box"][0], pose["box"][1], pose["box"][0] + pose["box"][2],
                                pose["box"][1] + pose["box"][3]]
                    tl_width_height_box = pose["box"]  # 获取alpha pose中的box，(左上角点，宽高)格式
                    true_box = [xtl, ytl, xbr, ybr]
                    iou_val = box_iou(pose_box, true_box)
                    if iou_val > max_iou:
                        x_keypoints_proposal = get_key_points(pose["keypoints"])
                        max_pose_box = tl_width_height_box
                        plot_max_box = pose_box
                        img_frame_id = int(annotation["@frame"])
                        max_iou = iou_val
                elif pose["image_id"] == str(int(annotation["@frame"]) + 1) + ".jpg":
                    break
            frame_id = annotation['@frame']
            human_id = annotation["attribute"][0]['#text']
            is_look = annotation["attribute"][2]["#text"]
            pose_in_img = max_pose_box and 0 < max_pose_box[0] < 1920 and 0 < max_pose_box[1] < 1080 and 0 < \
                          max_pose_box[2] < 1920 and 0 < max_pose_box[3] < 1080
            if x_keypoints_proposal and max_iou > max_iou_threshold and pose_in_img:
                label = 0
                if is_look == "looking":
                    label = 1
                x.append(
                    [uuid] + [int_video_id] + [idx] + [img_frame_id] + x_keypoints_proposal + max_pose_box + [label])
                id_uuid_list.append([uuid, int_video_id, human_id, frame_id])
                uuid += 1
                need_plot = False
                if need_plot and pose_box and max_iou > max_iou_threshold:
                    plot_img = plot_pose_box_look(plot_max_box, annotation, is_look, video_id, x_keypoints_proposal)
                    # video_pose_box.write(plot_img)

    # video_pose_box.release()
    print(video_id, "shape:", np.mat(x).shape)
    return np.mat(x), id_uuid_list


# 传入一个numpy数组，按第一列、第二列、第三列的顺序对numpy数组进行排序
def np_sort(n_arr: np.array):
    _a = n_arr
    _a = _a[np.lexsort((_a[:, 2], _a[:, 1], _a[:, 0]))]
    return _a


def get_init_data():
    video_count = 0  # 计算有多少个视频是有效的
    xml_anno = config.jaad_anno
    alpha_pose = config.alpha_pose
    uuid = 0
    id_uuid_list_all = []
    for i in range(1, 347):
        video_id_name = "video_" + str(i).zfill(4)
        xml_anno_path = xml_anno + video_id_name + ".xml"
        alpha_pose_path = alpha_pose + video_id_name + "/alphapose-results.json"
        x, id_uuid_list = get_train_data(xml_anno_path, alpha_pose_path, video_id_name, i, uuid)
        id_uuid_list_all.append(id_uuid_list)
        print("x.shape", x.shape[0], x.shape[1])
        if x.shape[1] > 1:
            uuid += x.shape[0]
            video_count += 1
            x_array = np_sort(np.asarray(x))

            y_array = x_array[:, -1]
            # np.savetxt("../cross/data/data" + str(i) + ".csv", x_array, delimiter=',')
            # np.savetxt("../cross/data/label" + str(i) + ".csv", y_array, delimiter=',')
    with open('文件名.csv', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(id_uuid_list_all)):
            for j in range(len(id_uuid_list_all[i])):
                row = id_uuid_list_all[i][j]
                csv_writer.writerow(row)
    return video_count


def read_all_to_one():
    all_pose = simple_normalize_read('train/halpe26_reid/', range(1, 347))
    sort_pose = np_sort(all_pose)
    np.savetxt('train/pose_all.csv', sort_pose, delimiter=',')
    pass


def rename_uuid():
    pose_arr = pd.read_csv('train/pose_all.csv', header=None, sep=',', encoding='utf-8').values
    for i in range(len(pose_arr)):
        pose_arr[i, 0] = i
    np.savetxt('train/pose_all_re.csv', pose_arr, delimiter=',')


if __name__ == "__main__":
    get_init_data()
