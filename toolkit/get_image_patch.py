# 生成行人bounding box的图像块，原始数据是大小不一的
# 出于训练神经网络的考虑，结果需要resize到大小一致
# 数据集评价宽长比为0.4/1，将所有图像resize为80-200，不足处补0
import pandas as pd
import numpy as np
import cv2
import os

from config import config_jaad_img


def read_pose_annotation(__video_id: int):
    data_path = "../train/halpe26_data/data_by_video/all_single/"
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


# 输出脸部图像的大小，宽和高
def face_image_patch(each_video_all_pose):
    image_path = config_jaad_img + "/video_"
    img_r_width, img_r_height = 80, 200  # 输出脸部图像的大小，宽和高
    each_video_pose = each_video_all_pose[0]
    img_id_start = 0
    for pose in each_video_pose:
        v_id, img_id, label = int(pose[0]), int(pose[1]), int(pose[84])
        img_file_path = image_path + str(v_id).zfill(4) + "/" + str(img_id) + ".jpg"
        raw_image = cv2.imread(img_file_path, 1)
        print("raw image shape:", raw_image.shape)
        xtl, ytl, width, height = round(pose[80]), round(pose[81]), round(pose[82]), round(pose[83])
        xbr, ybr = xtl + width, ytl + height
        print("xtl, ytl, xbr, ybr", xtl, ytl, xbr, ybr)

        img_patch = raw_image[ytl:ybr, xtl:xbr, :]
        print("img cropped patch shape:", img_patch.shape)
        img_height, img_width, img_shape = img_patch.shape
        os_dir = "../train/halpe26_data/data_by_video/image_face_patch/video_" + str(v_id).zfill(4)
        if not os.path.exists(os_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os_dir)
        else:
            print("保存为图像patch的文件夹已存在，注意不要覆盖图像")
            break
        save_path = os_dir + "/" + str(img_id_start) + ".jpg"
        img_id_start += 1
        print(save_path)
        if img_width / img_height < img_r_width / img_r_height:
            # 高度与结果一致
            fx = img_r_height / img_height
            img_patch_resize = cv2.resize(img_patch, dsize=(int_to_even(img_width * fx), img_r_height))
            patch_resize_height, patch_resize_width = img_patch_resize.shape[0], img_patch_resize.shape[1]
            print("img_patch_resize.shape:", img_patch_resize.shape)
            img_padding = np.zeros((img_r_height, (img_r_width - patch_resize_width) // 2, 3), np.uint8)
            print("img_padding.shape:", img_padding.shape)
            img_patch_concat = np.concatenate((img_padding, img_patch_resize, img_padding), axis=1)
        else:
            # 宽度与结果一致
            fy = img_r_width / img_width
            img_patch_resize = cv2.resize(img_patch, dsize=(img_r_width, int_to_even(img_height * fy)))
            patch_resize_height, patch_resize_width = img_patch_resize.shape[0], img_patch_resize.shape[1]
            print("img_patch_resize.shape:", img_patch_resize.shape)
            img_padding = np.zeros(((img_r_height - patch_resize_height) // 2, img_r_width, 3), np.uint8)
            print("img_padding shape: ", img_padding.shape)
            img_patch_concat = np.concatenate((img_padding, img_patch_resize, img_padding), axis=0)
        print("img_patch_concat shape", img_patch_concat.shape)
        cv2.imwrite(save_path, img_patch_concat)


# 整个的人体图像patch,未经过resize，保留原始大小
def total_body_img_patch_init(each_video_all_pose):
    image_path = config_jaad_img + "/video_"
    each_video_pose = each_video_all_pose[0]
    img_id_start = 0
    for pose in each_video_pose:
        v_id, img_id, label = int(pose[0]), int(pose[1]), int(pose[84])
        img_file_path = image_path + str(v_id).zfill(4) + "/" + str(img_id) + ".jpg"
        raw_image = cv2.imread(img_file_path, 1)
        print("raw image shape:", raw_image.shape)
        xtl, ytl, width, height = round(pose[80]), round(pose[81]), round(pose[82]), round(pose[83])
        xbr, ybr = xtl + width, ytl + height
        # print("xtl, ytl, xbr, ybr", xtl, ytl, xbr, ybr)
        img_patch = raw_image[ytl:ybr, xtl:xbr, :]
        print("img patch shape:", img_patch.shape)
        os_dir = "../train/halpe26_data/data_by_video/image_patch/video_" + str(v_id).zfill(4)
        if not os.path.exists(os_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os_dir)
        save_path = os_dir + "/" + str(img_id_start) + ".jpg"
        img_id_start += 1
        print(save_path)
        cv2.imwrite(save_path, img_patch)


# 整个的人体图像patch
def total_body_img_patch(each_video_all_pose):
    image_path = config_jaad_img + "/video_"
    img_r_width, img_r_height = 80, 200  # 输出结果图像的大小，宽和高
    each_video_pose = each_video_all_pose[0]
    img_id_start = 0
    for pose in each_video_pose:
        v_id, img_id, label = int(pose[0]), int(pose[1]), int(pose[84])
        img_file_path = image_path + str(v_id).zfill(4) + "/" + str(img_id) + ".jpg"
        raw_image = cv2.imread(img_file_path, 1)
        print("raw image shape:", raw_image.shape)
        xtl, ytl, width, height = round(pose[80]), round(pose[81]), round(pose[82]), round(pose[83])
        xbr, ybr = xtl + width, ytl + height
        print("xtl, ytl, xbr, ybr", xtl, ytl, xbr, ybr)

        img_patch = raw_image[ytl:ybr, xtl:xbr, :]
        print("img cropped patch shape:", img_patch.shape)
        img_height, img_width, img_shape = img_patch.shape
        os_dir = "../train/halpe26_data/data_by_video/image_patch/video_" + str(v_id).zfill(4)
        if not os.path.exists(os_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os_dir)
        else:
            print("保存为图像patch的文件夹已存在，注意不要覆盖图像")
            break
        save_path = os_dir + "/" + str(img_id_start) + ".jpg"
        img_id_start += 1
        print(save_path)
        if img_width / img_height < img_r_width / img_r_height:
            # 高度与结果一致
            fx = img_r_height / img_height
            img_patch_resize = cv2.resize(img_patch, dsize=(int_to_even(img_width * fx), img_r_height))
            patch_resize_height, patch_resize_width = img_patch_resize.shape[0], img_patch_resize.shape[1]
            print("img_patch_resize.shape:", img_patch_resize.shape)
            img_padding = np.zeros((img_r_height, (img_r_width - patch_resize_width) // 2, 3), np.uint8)
            print("img_padding.shape:", img_padding.shape)
            img_patch_concat = np.concatenate((img_padding, img_patch_resize, img_padding), axis=1)
        else:
            # 宽度与结果一致
            fy = img_r_width / img_width
            img_patch_resize = cv2.resize(img_patch, dsize=(img_r_width, int_to_even(img_height * fy)))
            patch_resize_height, patch_resize_width = img_patch_resize.shape[0], img_patch_resize.shape[1]
            print("img_patch_resize.shape:", img_patch_resize.shape)
            img_padding = np.zeros(((img_r_height - patch_resize_height) // 2, img_r_width, 3), np.uint8)
            print("img_padding shape: ", img_padding.shape)
            img_patch_concat = np.concatenate((img_padding, img_patch_resize, img_padding), axis=0)
        print("img_patch_concat shape", img_patch_concat.shape)
        cv2.imwrite(save_path, img_patch_concat)


if __name__ == "__main__":
    number_of_test = 347  # 测试的视频量
    for video_read_id in range(0, number_of_test):
        try:
            all_pose = np.array(read_pose_annotation(video_read_id))
            total_body_img_patch_init(all_pose)
        except OSError:
            print("data ", video_read_id, "is not exist")
        else:
            print("data has been load ", video_read_id)
