# 提供读取数据的方法
import pickle
import cv2
import numpy as np
from toolkit.read_pose_data import read_json
from toolkit.xml_read import xml_read, str_to_int
from toolkit.read_data import normalize_face_point_stream


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
def plot_pose_box_look(pose_box, annotation, is_look, video_id, points):
    output_file = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
    img = cv2.imread(output_file + annotation["@frame"] + ".jpg")
    xtl, ytl, xbr, ybr = int(pose_box[0]), int(pose_box[1]), int(pose_box[2]), int(pose_box[3])
    cv2.line(img, (xbr, ytl), (xtl, ytl), (0, 0, 255), thickness=2)
    cv2.line(img, (xbr, ytl), (xbr, ybr), (0, 0, 255), thickness=2)
    cv2.line(img, (xbr, ybr), (xtl, ybr), (0, 0, 255), thickness=2)
    cv2.line(img, (xtl, ybr), (xtl, ytl), (0, 0, 255), thickness=2)
    x_mid, y_mid = (xbr + xtl) // 2, (ybr + ytl) // 2
    cv2.putText(img, is_look, (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
    # cv2.putText(img, str(pose_box[0]), (x_mid, y_mid + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    for i in range(26):
        point_colour = (0, 0, 255)
        if points[i + 2] > 0.7:
            point_colour = (255, 255, 0)
        cv2.circle(img, (round(points[i * 3]), round(points[i * 3 + 1])), 2, point_colour, 2)
    # img = cv2.resize(img, (1920 // 2, 1080 // 2))
    cv2.imshow("./pose box looking", img)
    cv2.waitKey(1)
    return img


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


def get_train_data(jaad_anno_path, alpha_pose_path, video_id, int_video_id):
    x, y = [], []  # # key points 0 1 2 3 4 17 18 label looking not-looking
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
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式 cap_fps, size = 30, (1920,1080)
    # size（width，height）
    # video_pose_box = cv2.VideoWriter("train_data/iou/data_by_video-pose-box" + str(
    # video_id) + ".mp4", fourcc, cap_fps, size)
    for i in track_box:
        if i["@label"] != "pedestrian":
            continue
        is_repeat += 1
        if is_repeat >= 2:
            print(jaad_anno_path, "there are two box in one data_by_video")
        for annotation in i["box"]:
            # jaad 注释文件，左上角，右下角 (top-left, bottom-right), ybr>xbr,ytl>xtl
            xtl, ytl = str_to_int(annotation["@xtl"]), str_to_int(annotation["@ytl"])
            xbr, ybr = str_to_int(annotation["@xbr"]), str_to_int(annotation["@ybr"])
            # x_mid, y_mid = (xtl + xbr) // 2, (ytl + ybr) // 2

            max_iou = 0.6
            pose_box = []  # alpha pose的box位置,json文件中为([0],[1])左上角,([2],[3])宽和高,修改成(左上角,右下角)格式
            x_keypoints_proposal, max_pose_box = [], []  # 存储key points,max_pose_box为iou最大时的box（中心点，宽高）格式
            img_frame_id = 0
            plot_max_box = []
            for pose in alpha_pose:
                if pose["score"] < 1:
                    continue
                if pose["image_id"] == annotation["@frame"] + ".jpg":
                    pose_box = [pose["box"][0], pose["box"][1], pose["box"][0] + pose["box"][2],
                                pose["box"][1] + pose["box"][3]]
                    mid_width_height_box = pose["box"]
                    true_box = [xtl, ytl, xbr, ybr]
                    iou_val = box_iou(pose_box, true_box)
                    if iou_val > max_iou:
                        x_keypoints_proposal = get_key_points(pose["keypoints"])
                        max_pose_box = mid_width_height_box
                        plot_max_box = pose_box
                        img_frame_id = int(annotation["@frame"])
                        max_iou = iou_val
                elif pose["image_id"] == str(int(annotation["@frame"]) + 1) + ".jpg":
                    break
            is_look = annotation["attribute"][2]["#text"]
            if x_keypoints_proposal and max_iou > 0.6:
                label = 0
                if is_look == "looking":
                    label = 1
                y.append(label)
                x.append([int_video_id] + [img_frame_id] + x_keypoints_proposal + max_pose_box + [label])
                need_plot = False
                if need_plot and pose_box and max_iou > 0.6:
                    plot_img = plot_pose_box_look(plot_max_box, annotation, is_look, video_id, x_keypoints_proposal)
                    # video_pose_box.write(plot_img)

    # video_pose_box.release()
    print(video_id, "shape:", np.mat(x).shape, np.mat(y).T.shape)
    return np.mat(x), np.mat(y).T


def get_stream_data():
    data_path = "../train/train_data/iou/data_by_video/single/"
    label_path = "../train/train_data/iou/data_by_video/single/"
    for str_id in range(1, 347):
        try:
            pose_arr = np.loadtxt(data_path + "data" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
            label_arr = np.loadtxt(label_path + "label" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
            pose, label = normalize_face_point_stream(pose_arr, label_arr)
            np.savetxt("train_data/iou/data_by_video/stream/data" + str(str_id) + ".csv", pose, delimiter=',')
            np.savetxt("train_data/iou/data_by_video/stream/label" + str(str_id) + ".csv", label, delimiter=',')
        except OSError:
            print("data or label ", str_id, "is not exist")
        else:
            print("data has been load ", str_id)


def get_init_data():
    train_data_shape = 85  # 训练的数据的列的大小为7，总训练的数据格式为(number_of_data,train_data_shape)
    train_dataset, labels = np.zeros((1, train_data_shape), float), np.zeros((1, 1), float)
    for i in range(1, 347):
        video_id = "video_" + str(i).zfill(4)
        xml_anno_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        # output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
        alpha_pose_path = "E:/CodeResp/pycode/DataSet/pose_result/" + video_id + "/alphapose-results.json"
        x, y = get_train_data(xml_anno_path, alpha_pose_path, video_id, i)
        if x.shape[1] == train_data_shape:
            x_array, y_array = np.asarray(x), np.asarray(y)
            np.savetxt("train_data/iou/data_by_video/all_single/data" + str(i) + ".csv", x_array, delimiter=',')
            np.savetxt("train_data/iou/data_by_video/all_single/label" + str(i) + ".csv", y_array, delimiter=',')
            train_dataset = np.concatenate((train_dataset, x))
            labels = np.concatenate((labels, y))
    # print("all data saved, shape:", train_dataset.shape, labels.shape, "true shape:", train_data_shape)
    # train_dataset = np.asarray(train_dataset)
    # labels = np.asarray(labels)
    # np.savetxt("train_data/test_all_train_data.csv", train_dataset, delimiter=',')
    # np.savetxt("train_data/test_all_label.csv", labels, delimiter=',')
    return train_dataset, labels


def save_model(file_path, file_name, model):
    with open(file=file_path + file_name, mode="wb") as f:
        f.write(model)


def load_model(file_path):
    with open(file=file_path, mode="rb") as trained_model:
        s2 = trained_model.read()
        model = pickle.loads(s2)
    # expected = test_y
    # predicted = model1.predict(test_X)
    return model


if __name__ == "__main__":
    get_init_data()
    exit(0)
