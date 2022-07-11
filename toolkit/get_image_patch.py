# 生成行人bounding box的图像块，原始数据是大小不一的
# 出于训练神经网络的考虑，结果需要resize到大小一致
# 数据集评价宽长比为0.4/1，将所有图像resize为80-200，不足处补0
import pandas as pd
import numpy as np
import cv2
import os

from config import config_jaad_img, config_img_save_patch, config_dataset_root
from toolkit.read_data import train_data_list, test_data_list

config_csv_data = "../train/halpe26_reid/"


def read_pose_annotation(__video_id: int):
    data_path = config_csv_data
    pose_arr = pd.read_csv(data_path + "data" + str(__video_id) + ".csv", header=None, sep=',',
                           encoding='utf-8').values
    print(__video_id, "shape:", pose_arr.shape)
    return pose_arr,


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


# 整数转偶数
def int_to_even(number: int):
    return int(number // 2 * 2)


# 整个的人体图像patch,未经过resize，保留原始大小
def total_body_img_patch_init(each_video_all_pose):
    image_path = config_jaad_img + "video_"
    each_video_pose = each_video_all_pose[0]
    img_id_start = 0
    print(config_dataset_root)
    train_txt = open(config_dataset_root + 'train.txt', 'a')  # 以追加写方式打开文件
    test_txt = open(config_dataset_root + 'test.txt', 'a')
    for pose in each_video_pose:
        uuid, v_id, idx, img_id, label = int(pose[0]), int(pose[1]), int(pose[2]), int(pose[3]), int(pose[86])
        img_file_path = image_path + str(v_id).zfill(4) + "/" + str(img_id) + ".jpg"
        raw_image = cv2.imread(img_file_path, 1)

        xtl, ytl, width, height = round(pose[82]), round(pose[83]), round(pose[84]), round(pose[85])
        xbr, ybr = xtl + width, ytl + height
        print(ytl, ybr, xtl, xbr)
        # print("xtl, ytl, xbr, ybr", xtl, ytl, xbr, ybr)
        img_patch = raw_image[ytl:ybr, xtl:xbr, :]
        print("img patch shape:", img_patch.shape)
        os_dir = config_img_save_patch + str(v_id).zfill(4)
        if not os.path.exists(os_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os_dir)
        img_patch_path = os_dir + "/" + str(img_id_start) + ".jpg"
        img_id_start += 1
        print(img_patch_path)
        cv2.imwrite(img_patch_path, img_patch)
        # 由于训练的图像需要得到uuid，所以路径中新增uuid和idx
        img_patch_path_to_train = img_patch_path + "*" + str(uuid) + "/" + str(idx)
        if v_id in train_data_list:
            train_txt.write(img_patch_path_to_train + ' ' + str(label) + '\n')
        elif v_id in test_data_list:
            test_txt.write(img_patch_path_to_train + ' ' + str(label) + '\n')
        else:
            print("error, video id is not in train or test list")
    train_txt.close()
    test_txt.close()


if __name__ == "__main__":
    number_of_test = 347  # 测试的视频量
    for video_read_id in range(1, number_of_test):
        try:
            all_pose = np.array(read_pose_annotation(video_read_id))
            total_body_img_patch_init(all_pose)
        except OSError:
            print("data ", video_read_id, "is not exist")
        else:
            print("data has been load ", video_read_id)
