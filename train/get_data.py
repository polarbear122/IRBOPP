import cv2
from toolkit.xml_read import xml_read, str_to_int
from toolkit.read_pose_data import read_json
import numpy as np


def get_key_points(keypoints):
    # key points [0, 1, 2, 3, 4, 17, 18]
    if len(keypoints) != 78:
        return []
    key_points = []
    for i in [0, 1, 2, 3, 4, 17, 18]:
        key_points.append(keypoints[i * 3 + 2])

    return key_points


def plot_pose_box_look(pose_box, annotation, is_look, video_id):
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

            xbr = str_to_int(annotation["@xbr"])  # 左上角，右下角 (top-left, bottom-right)
            xtl = str_to_int(annotation["@xtl"])
            ybr = str_to_int(annotation["@ybr"])  # ybr>xbr,ytl>xtl
            ytl = str_to_int(annotation["@ytl"])
            x_mid, y_mid = (xtl + xbr) // 2, (ytl + ybr) // 2

            mid_difference = 100  # 中心点误差
            pose_box = []  # alpha pose的box位置，用于检测,([0],[1])为左上角，（[2],[3])为宽和高
            x_keypoints_proposal = []  # 存储key points
            for pose in alpha_pose:
                if pose["score"] < 1.0:
                    continue
                if pose["image_id"] == annotation["@frame"] + ".jpg":
                    diff = (pose["box"][0] + pose["box"][2] / 2 - x_mid) ** 2 + (
                            pose["box"][1] + pose["box"][3] / 2 - y_mid) ** 2
                    if diff < mid_difference:
                        mid_difference = diff
                        x_keypoints_proposal = get_key_points(pose["keypoints"])
                        pose_box = pose["box"]
            is_look = annotation["attribute"][2]["#text"]
            if x_keypoints_proposal:
                x.append(x_keypoints_proposal)
                if is_look == "looking":
                    y.append(1)
                else:
                    y.append(0)
            need_plot = False
            if pose_box:
                if need_plot:
                    plot_pose_box_look(pose_box, annotation, is_look)

    print(video_id, "shape:", np.mat(x).shape, np.mat(y).T.shape)
    return np.mat(x), np.mat(y).T
