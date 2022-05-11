# 生成行人bounding box的图像块，原始数据是大小不一的
# 出于训练神经网络的考虑，结果需要resize到大小一致
# 数据集评价宽长比为0.4/1，将所有图像resize为80-200，不足处补0
import pandas as pd
import numpy as np
import cv2


def read_pose_annotation():
    data_path = "../train/train_data/iou/data_by_video/all_single/"
    label_path = "../train/train_data/iou/data_by_video/all_single/"
    stream_pose = pd.read_csv(data_path + "data1.csv", header=None, sep=',', encoding='utf-8').values
    stream_label = pd.read_csv(data_path + "label1.csv", header=None, sep=',', encoding='utf-8').values
    for str_id in range(2, number_of_test):
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
    # stream_pose = stream_pose[:, 2:stream_pose.shape[1] - 1]  # 原始数据包含视频id，图片id，标签，需要去除
    return stream_pose, stream_label


# 将中心点(middle_x, middle_y)，宽高(box_width, box_height)格式转左上右下点格式
def mid_width_to_tl_br(middle_x, middle_y, box_width, box_height):
    top_left_x, top_left_y = middle_x - box_width / 2, middle_y - box_height / 2
    bottom_right_x, bottom_right_y = middle_x + box_width / 2, middle_y + box_height / 2
    return round(top_left_x), round(top_left_y), round(bottom_right_x), round(bottom_right_y)


# 将左上(top_left_x, top_left_y)右下(bottom_right_x, bottom_right_y)点格式转中心点，宽高格式
def tl_br_to_mid_width(top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    middle_x, middle_y = (top_left_x + bottom_right_x) / 2, (top_left_y + bottom_right_y) / 2
    box_width, box_height = (bottom_right_x - top_left_x), (bottom_right_y - top_left_y)
    return middle_x, middle_y, box_width, box_height


if __name__ == "__main__":
    image_path = "E:/CodeResp/pycode/DataSet/JAAD_image/video_"
    number_of_test = 3  # 测试的视频量
    image_result_width, image_result_height = 80, 200
    all_pose, all_label = read_pose_annotation()
    for pose in all_pose:
        v_id, img_id, label = int(pose[0]), int(pose[1]), int(pose[84])
        img_file_path = image_path + str(v_id).zfill(4) + "/" + str(img_id) + ".jpg"
        raw_image = cv2.imread(img_file_path)
        x_mid, y_mid, width, height = pose[80], pose[81], pose[82], pose[83]
        xtl, ytl, xbr, ybr = mid_width_to_tl_br(x_mid, y_mid, width, height)
        # print(v_id, img_id, x_mid, y_mid, width, height, label)
        img_patch = raw_image[xtl:xbr, ytl:ybr]
        img_width, img_height, img_shape = img_patch.shape
        print(img_width, img_height, img_shape)
        if img_width / img_height < image_result_width/image_result_height:
            pass
#         todo
