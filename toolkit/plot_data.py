import cv2
import numpy as np
import pandas as pd
# from train.test_joint_image_video import calculate_result
# python 需要import自己的目录，如何加这个目录
from config import jaad_img

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


# 画出（左上角，宽，高）格式的box
def plot_pose_box_look_w_h(pose_box, annotation, is_look, video_id, keypoints):
    a, b, c, __i = round(pose_box[0]), round(pose_box[1]), round(pose_box[2]), round(pose_box[3])
    box = [a, b, a + c, b + __i]
    return plot_pose_box_look(box, annotation, is_look, video_id, keypoints)


# 画出（左上角，右下角）格式的box
def plot_pose_box_look(pose_box, annotation, is_look, video_id, keypoints):
    img_file_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
    img = cv2.imread(img_file_path + annotation["@frame"] + ".jpg")
    img = draw_pose(img, pose_box, keypoints)
    xtl, ytl, xbr, ybr = round(pose_box[0]), round(pose_box[1]), round(pose_box[2]), round(pose_box[3])
    # x_mid, y_mid = (xbr + xtl) // 2, (ybr + ytl) // 2
    cv2.putText(img, is_look, (xtl, ytl), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    img = cv2.resize(img, (1920 // 2, 1080 // 2))
    cv2.imshow("pose box looking", img)
    cv2.waitKey(1)
    return img


def draw_pose(frame, bbox, human_keypoints):
    # kp_num == 26
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
        (17, 18), (18, 19), (19, 11), (19, 12),
        (11, 13), (12, 14), (13, 15), (14, 16),
        (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25), ]  # Foot

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
               # Nose, LEye, REye, LEar, REar
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
               # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
               (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
               (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)]  # foot

    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                  (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                  (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                  (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]

    if len(human_keypoints) == 17 * 3:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    img = frame.copy()
    part_line = {}
    kp_preds = np.array(human_keypoints).reshape(-1, 3)
    color = RED
    # Draw bboxes # xmin,xmax,ymin,ymax
    cv2.rectangle(img, (round(bbox[0]), round(bbox[1])), (round(bbox[2]), round(bbox[3])), color, 2)

    # Draw keypoints
    for n in range(kp_preds.shape[0]):
        cor_x, cor_y = round(kp_preds[n, 0]), round(kp_preds[n, 1])
        part_line[n] = (cor_x, cor_y)
        if n < len(p_color):
            cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
        else:
            cv2.circle(img, (cor_x, cor_y), 1, (255, 255, 255), 2)

    # Draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(img, start_xy, end_xy, line_color[i])
    return img


# 画出检测结果look/not-look,并与真实jaad数据集进行对比
def plot_test_result():
    video_st, video_end = 1, 347
    img_path = jaad_img + "/video_"
    data_path = "../train/halpe26_data/data_by_video/all_single/"
    # 经过训练后的模型的检测结果
    trained_label = pd.read_csv("../train/y_pred_joint.csv", header=None, sep=',',
                                encoding='utf-8').values
    __st_id = 0
    for str_id in range(video_st, video_end):
        try:
            pose_arr = pd.read_csv(data_path + "data" + str(str_id) + ".csv", header=None, sep=',',
                                   encoding='utf-8').values
            print(str_id, "shape:", pose_arr.shape)
            for __i in range(len(pose_arr)):
                pose = pose_arr[__i]
                video_id, img_frame_id = str(int(pose[0])), str(int(pose[1]))
                img = cv2.imread(img_path + video_id.zfill(4) + "/" + img_frame_id + ".jpg")

                keypoints = []
                for j in range(26):
                    keypoints += pose[j * 3 + 2], pose[j * 3 + 2 + 1], pose[j * 3 + 2 + 2]

                xtl, ytl, width, height = round(pose[80]), round(pose[81]), round(pose[82]), round(pose[83])
                label_true = int(pose[84])
                pose_box = [xtl, ytl, xtl + width, ytl + height]

                img = draw_pose(img, pose_box, keypoints)

                is_look_t = is_look_p = "not-looking"  # is_look_t: label真值, is_look_p:label预测值
                if label_true == 1:
                    is_look_t = "looking"
                if trained_label[__st_id] == 1:
                    is_look_p = "looking"
                __st_id += 1
                cv2.putText(img, is_look_t, (xtl, ytl), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
                wait_time = 10
                # 当预测值与真值不同时，画出异常值
                if is_look_t != is_look_p:
                    cv2.putText(img, is_look_p, (xtl, ytl - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
                    wait_time = 100
                img = cv2.resize(img, (1920 // 2, 1080 // 2))
                cv2.imshow("pose box looking", img)
                cv2.waitKey(wait_time)
        except OSError:
            print("data or label ", str_id, "is not exist")
        else:
            print("data has been load ", str_id)


if __name__ == "__main__":
    plot_test_result()
