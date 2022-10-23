# 生成行人bounding box的图像块，原始数据是大小不一的
# 出于训练神经网络的考虑，结果需要resize到大小一致
# 数据集评价宽长比为0.4/1，将所有图像resize为80-200，不足处补0
import pandas as pd
import numpy as np
from PIL import Image

from config import jaad_total_img, img_all_patch, img_face_patch

train_list = [1, 3, 4, 5, 6, 9, 10, 11, 12, 17, 21, 25, 27, 28, 31, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 53, 54, 55,
              56, 57, 59, 60, 61, 62, 63, 65, 66, 68, 70, 71, 79, 80, 82, 84, 86, 87, 89, 90, 91, 92, 95, 97, 98, 100,
              101, 104, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 119, 122, 124, 125, 126, 128, 129, 130, 131,
              132, 135, 136, 137, 138, 139, 140, 141, 143, 146, 147, 149, 150, 151, 152, 154, 155, 156, 157, 158, 161,
              163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 178, 183, 184, 188, 189, 190, 191, 193, 194, 195,
              196, 197, 198, 199, 200, 201, 204, 206, 207, 208, 210, 213, 214, 215, 217, 218, 219, 221, 222, 223, 224,
              225, 226, 227, 228, 231, 233, 234, 235, 236, 238, 239, 240, 241, 242, 244, 246, 247, 248, 249, 250, 253,
              254, 255, 256, 257, 260, 261, 263, 264, 266, 267, 268, 269, 270, 271, 272, 274, 275, 276, 277, 279, 280,
              281, 283, 284, 286, 287, 288, 290, 293, 294, 295, 297, 298, 300, 302, 303, 304, 305, 306, 307, 308, 309,
              310, 311, 312, 313, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
              334, 336, 337, 339, 340, 341, 342, 344, 345]

test_list = [20, 30, 64, 69, 72, 73, 76, 78, 81, 83, 88, 93, 94, 96, 99, 102, 107, 120, 123, 134, 144, 148, 159, 160,
             175, 176, 177, 180, 185, 186, 187, 192, 209, 216, 230, 245, 252, 258, 262, 265, 278, 285, 291, 299, 314,
             315]
val_list = [7, 8, 14, 16, 35, 42, 67, 77, 85, 103, 105, 117, 118, 133, 145, 162, 171, 179, 181, 203, 205, 211, 212, 229,
            232, 237, 243, 251, 259, 289, 292, 301, 335, 338]


def read_pose_annotation(__video_id: int):
    pose_arr = pd.read_csv('train/pose_all.csv', header=None, sep=',', encoding='utf-8').values
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


# 输入所有特征点26个特征点(x,y,c)加上4维box，输出box范围，脸部范围：0，1，2，3，4，17，18
def get_box_from_keypoints(pose, is_body):
    if is_body:
        xtl, ytl, width, height = pose[-4], pose[-3], pose[-2], pose[-1]
        xbr, ybr = xtl + width, ytl + height
        xtl_factor, xbr_factor, ytl_factor, ybr_factor = 1.2, 1.2, 0.7, 0.5
    else:
        face_list = [0, 1, 2, 3, 4, 17, 18]
        xtl, ytl = 1920, 1080
        xbr, ybr = 0, 0
        xtl_factor, xbr_factor, ytl_factor, ybr_factor = 1.5, 1.5, 1.5, 1.5
        for i in face_list:
            xtl = min(xtl, pose[i * 3])
            ytl = min(ytl, pose[i * 3 + 1])
            xbr = max(xbr, pose[i * 3])
            ybr = max(ybr, pose[i * 3 + 1])
    x_mid, x_sub = (xtl + xbr) / 2, xbr - xtl
    xtl = x_mid - x_sub * xtl_factor
    xbr = x_mid + x_sub * xbr_factor
    y_mid, y_sub = (ytl + ybr) / 2, ybr - ytl
    ytl = y_mid - y_sub * ytl_factor
    ybr = y_mid + y_sub * ybr_factor
    if xbr < 0:
        xbr = 0
    elif xbr >= 1920:
        xbr = 1920 - 1
    if xtl < 0:
        xtl = 0
    elif xtl >= 1920:
        xtl = 1920 - 1
    if ytl < 0:
        ytl = 0
    elif ytl >= 1080:
        ytl = 1080 - 1
    if ybr < 0:
        ybr = 0
    elif ybr >= 1080:
        ybr = 1080 - 1
    need_continue = False
    xtl, ytl, xbr, ybr = round(xtl), round(ytl), round(xbr), round(ybr)
    if ybr <= ytl or xbr <= xtl:
        need_continue = True
    # print(ytl, ybr, xtl, xbr)
    return xtl, ytl, xbr, ybr, need_continue


# 整个的人体图像patch,未经过resize，保留原始大小
def generate_img_patch_init(all_pose, txt_path, _is_body_not_face):
    image_path = jaad_total_img + "video_"
    print(txt_path)
    train_txt = open(txt_path + 'train.txt', 'a')  # 以追加写方式打开文件
    test_txt = open(txt_path + 'test.txt', 'a')
    val_txt = open(txt_path + 'val.txt', 'a')
    all_txt = open(txt_path + 'all.txt', 'a')
    for i in range(len(all_pose)):
        pose = all_pose[i]
        uuid, v_id, idx, img_id, label = int(pose[0]), int(pose[1]), int(pose[2]), int(pose[3]), int(pose[86])
        xtl, ytl, xbr, ybr, need_continue = get_box_from_keypoints(pose[4:86], True)
        xtl_f, ytl_f, xbr_f, ybr_f, need_continue_f = get_box_from_keypoints(pose[4:86], False)
        if need_continue or need_continue_f:
            continue
        img_file_path = image_path + str(v_id).zfill(4) + "/" + str(img_id) + ".jpg"
        if i == 0 or v_id != int(all_pose[i - 1][1]) or img_id != int(all_pose[i - 1][3]):
            process_raw_img = Image.open(img_file_path)
        box = (xtl, ytl, xbr, ybr)
        img_patch = process_raw_img.crop(box)
        img_patch.save('train/halpe26_data/body_img/' + str(uuid) + ".bmp")
        face_box = (xtl_f, ytl_f, xbr_f, ybr_f)
        print(box, face_box)
        face_img_patch = process_raw_img.crop(face_box)
        face_img_patch.save('train/halpe26_data/face_img/' + str(uuid) + ".bmp")
    train_txt.close()
    test_txt.close()
    val_txt.close()
    all_txt.close()


if __name__ == "__main__":
    # train_list_random, test_list_random, val_list_random = random_sort_high_vis_list()
    is_body_not_face = True  # 是否生成全身图像数据
    if is_body_not_face:
        save_txt_path = ''
        img_save_path = 'train/halpe26_data/body_img/'
    else:
        save_txt_path = ''
        img_save_path = 'train/halpe26_data/face_img/'
    train_txt_file = open(save_txt_path + 'train.txt', 'w')  # 以写方式打开文件
    test_txt_file = open(save_txt_path + 'test.txt', 'w')
    val_txt_file = open(save_txt_path + 'val.txt', 'w')
    all_txt_file = open(save_txt_path + 'all.txt', 'w')
    train_txt_file.close()
    test_txt_file.close()
    val_txt_file.close()
    all_txt_file.close()
    all_pose = pd.read_csv('train/pose_all.csv', header=None, sep=',', encoding='utf-8').values
    generate_img_patch_init(all_pose, save_txt_path, is_body_not_face)
