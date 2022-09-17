# 生成行人bounding box的图像块，原始数据是大小不一的
# 出于训练神经网络的考虑，结果需要resize到大小一致
# 数据集评价宽长比为0.4/1，将所有图像resize为80-200，不足处补0
import pandas as pd
import numpy as np
import cv2
import os

from config import jaad_total_img, img_all_patch, generate_dataset_txt_root, img_face_patch, \
    train_list, val_list, test_list
from toolkit.read_data import train_data_list, test_data_list

config_csv_data = "../train/halpe26_reid/"

# 高可见性的视频序列test 121 train 182 val  30.sum = 333
jaad_all_videos_test = [5, 15, 16, 17, 28, 36, 42, 43, 45, 46, 48, 53, 55, 58, 59, 63, 67, 68, 70, 71, 75, 76, 84, 87,
                        90, 92, 93, 96, 97, 100, 101, 103, 104, 105, 106, 107, 110, 113, 115, 116, 117, 118, 124, 125,
                        128, 135, 141, 144, 148, 150, 151, 152, 153, 155, 162, 163, 164, 165, 173, 177, 178, 179, 183,
                        187, 197, 201, 203, 206, 211, 212, 213, 216, 221, 222, 223, 224, 230, 234, 238, 239, 243, 244,
                        245, 251, 253, 265, 267, 270, 271, 277, 278, 279, 280, 285, 287, 288, 292, 294, 295, 299, 300,
                        304, 305, 307, 308, 309, 313, 314, 316, 322, 327, 329, 330, 332, 333, 334, 336, 337, 338, 339,
                        344]
jaad_all_videos_train = [1, 3, 4, 7, 8, 9, 10, 11, 12, 14, 18, 19, 20, 25, 27, 30, 31, 35, 37, 38, 39, 47, 49, 50, 51,
                         52, 54, 56, 57, 60, 61, 62, 64, 66, 69, 74, 77, 78, 79, 80, 81, 83, 85, 86, 88, 91, 94, 95, 98,
                         108, 109, 111, 112, 114, 119, 120, 121, 122, 126, 129, 130, 131, 132, 133, 134, 136, 137, 138,
                         139, 140, 142, 143, 145, 146, 147, 149, 154, 157, 158, 159, 161, 166, 167, 168, 169, 171, 174,
                         175, 176, 180, 182, 184, 185, 186, 188, 189, 190, 191, 192, 194, 195, 196, 198, 200, 202, 204,
                         205, 207, 208, 209, 210, 214, 215, 218, 219, 220, 225, 227, 228, 229, 231, 232, 233, 235, 236,
                         237, 240, 241, 242, 246, 247, 248, 249, 250, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264,
                         266, 268, 269, 272, 275, 276, 281, 282, 283, 284, 286, 289, 290, 293, 296, 297, 298, 301, 302,
                         310, 311, 312, 315, 317, 318, 319, 320, 321, 323, 324, 325, 326, 328, 331, 335, 341, 342, 345]
jaad_all_videos_val = [6, 21, 40, 44, 65, 72, 73, 82, 89, 99, 102, 123, 156, 160, 170, 172, 181, 193, 199, 217, 226,
                       252, 263, 273, 274, 291, 303, 306, 340, 343]

high_visibility_all_list = jaad_all_videos_test + jaad_all_videos_train + jaad_all_videos_val


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


# 输入所有特征点，输出box范围，脸部范围：0，1，2，3，4，17，18
def get_box_from_keypoints(pose, is_face=False):
    xtl, ytl, width, height = pose[82], pose[83], pose[84], pose[85]
    if is_face:
        face_list = [0, 1, 2, 3, 4, 17, 18]
        xtl, ytl = 1920, 1080
        xbr, ybr = 0, 0
        for i in face_list:
            xtl = min(xtl, face_list[i * 3])
            ytl = min(ytl, face_list[i * 3 + 1])
            xbr = max(xbr, face_list[i * 3])
            ybr = max(ybr, face_list[i * 3 + 1])
    xbr, ybr = xtl + width, ytl + height
    x_mid, x_sub = (xtl + xbr) / 2, xbr - xtl
    xtl = x_mid - x_sub * 1.2
    xbr = x_mid + x_sub * 1.2
    y_mid, y_sub = (ytl + ybr) / 2, ybr - ytl
    ytl = y_mid - y_sub * 0.7
    ybr = y_mid + y_sub * 0.3

    if xbr < 0:
        xbr = 0
    elif xbr > 1920:
        xbr = 1920
    if xtl < 0:
        xtl = 0
    elif xtl > 1920:
        xtl = 1920
    if ytl < 0:
        ytl = 0
    elif ytl > 1080:
        ytl = 1080
    if ybr < 0:
        ybr = 0
    elif ybr > 1080:
        ybr = 1080
    need_continue = False
    if ybr == ytl or xbr == xtl:
        need_continue = True
    # print(ytl, ybr, xtl, xbr)
    xtl, ytl, xbr, ybr = round(xtl), round(ytl), round(xbr), round(ybr)
    return xtl, ytl, xbr, ybr, need_continue


# 整个的人体图像patch,未经过resize，保留原始大小
def total_body_img_patch_init(each_video_all_pose, txt_path):
    global st_id, train_id, test_id
    image_path = jaad_total_img + "video_"
    each_video_pose = each_video_all_pose[0]
    img_id_start = 0
    print(txt_path)
    train_txt = open(txt_path + 'train.txt', 'a')  # 以追加写方式打开文件
    test_txt = open(txt_path + 'test.txt', 'a')
    val_txt = open(txt_path + 'val.txt', 'a')
    for pose in each_video_pose:
        uuid, v_id, idx, img_id, label = int(pose[0]), int(pose[1]), int(pose[2]), int(pose[3]), int(pose[86])
        xtl, ytl, xbr, ybr, need_continue = get_box_from_keypoints(pose)
        if need_continue:
            continue
        h_all[st_id], w_all[st_id] = ybr - ytl, xbr - xtl
        st_id += 1
        os_dir = img_all_patch + str(v_id).zfill(4)
        if not os.path.exists(os_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os_dir)
        img_id_start += 1
        img_patch_path = os_dir + "/" + str(img_id_start) + ".jpg"
        # img_file_path = image_path + str(v_id).zfill(4) + "/" + str(img_id) + ".jpg"
        # print("img_file_path", img_file_path)
        # raw_image = cv2.imread(img_file_path, 1)
        # img_patch = raw_image[ytl:ybr, xtl:xbr, :]
        # # img_patch = cv2.resize(img_patch, (128, 128))
        # print("img patch shape:", img_patch.shape)
        # cv2.putText(img_patch, str(v_id) + "v" + str(img_id), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        # cv2.putText(img_patch, str(label), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # print("video", v_id, img_id, label)
        # cv2.imshow("img", img_patch)
        # cv2.waitKey(1)
        # # print(img_patch_path)
        # cv2.imwrite(img_patch_path, img_patch)
        # 由于训练的图像需要得到uuid，所以路径中新增uuid和idx
        img_patch_path_to_train = img_patch_path + "*" + str(uuid) + "/" + str(img_id_start)
        if img_id_start % 3 == 0:
            if v_id in train_list:
                train_txt.write(img_patch_path_to_train + ' ' + str(label) + '\n')
                label_train[train_id] = label
                train_id += 1
            elif v_id in test_list:
                test_txt.write(img_patch_path_to_train + ' ' + str(label) + '\n')
                label_test[test_id] = label
                test_id += 1
            elif v_id in val_list:
                val_txt.write(img_patch_path_to_train + ' ' + str(label) + '\n')
            else:
                print(v_id, "error, video id is not in train or test list")
    print("height width average:%.2f,%.2f" % (np.average(h_all[:st_id]), np.average(w_all[:st_id])))
    train_txt.close()
    test_txt.close()


# 整个的脸部图像patch,未经过resize，保留原始大小
def face_img_patch_init(each_video_all_pose, _save_txt_path):
    image_path = jaad_total_img + "video_"
    each_video_pose = each_video_all_pose[0]
    img_id_start = 0
    print(_save_txt_path)
    train_txt = open(_save_txt_path + 'train.txt', 'a')  # 以写方式打开文件
    test_txt = open(_save_txt_path + 'test.txt', 'a')
    train_sum, test_sum, val_sum = 0, 0, 0
    for pose in each_video_pose:
        uuid, v_id, idx, img_id, label = int(pose[0]), int(pose[1]), int(pose[2]), int(pose[3]), int(pose[86])
        xtl, ytl, xbr, ybr, need_continue = get_box_from_keypoints(pose, is_face=True)
        if need_continue:
            continue
        os_dir = img_face_patch + str(v_id).zfill(4)
        if not os.path.exists(os_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os_dir)
        img_patch_path = os_dir + "/" + str(img_id_start) + ".jpg"
        img_id_start += 1
        # img_file_path = image_path + str(v_id).zfill(4) + "/" + str(img_id) + ".jpg"
        # # print(img_patch_path)
        # raw_image = cv2.imread(img_file_path, 1)
        # img_patch = raw_image[ytl:ybr, xtl:xbr, :]
        # print("img patch shape:", img_patch.shape)
        # cv2.imwrite(img_patch_path, img_patch)
        # 由于训练的图像需要得到uuid，所以路径中新增uuid和id_in_video
        img_patch_path_to_train = img_patch_path + "*" + str(uuid) + "/" + str(img_id_start)
        if img_id_start % 4 == 0:
            if v_id in train_list:
                train_txt.write(img_patch_path_to_train + ' ' + str(label) + '\n')
                train_sum += 1
            elif v_id in test_list:
                test_txt.write(img_patch_path_to_train + ' ' + str(label) + '\n')
                test_sum += 1
            else:
                print(v_id, "error, video id is not in train or test list")
    print("train_sum,test_sum:", train_sum, test_sum)
    train_txt.close()
    test_txt.close()
    return train_sum, test_sum, val_sum


if __name__ == "__main__":
    st_id = 0
    h_all, w_all = np.zeros((100000,), float), np.zeros((100000,), float)
    train_id, test_id = 0, 0
    label_train, label_test = np.zeros((100000,), float), np.zeros((100000,), float)
    save_txt_path = generate_dataset_txt_root
    train_txt_file = open(save_txt_path + 'train.txt', 'w')  # 以写方式打开文件
    test_txt_file = open(save_txt_path + 'test.txt', 'w')
    val_txt_file = open(save_txt_path + 'val.txt', 'w')

    train_txt_file.close()
    test_txt_file.close()
    val_txt_file.close()
    for video_read_id in range(1, 347):
        try:
            all_pose = np.array(read_pose_annotation(video_read_id))
            total_body_img_patch_init(all_pose, save_txt_path)
        except OSError:
            print("data ", video_read_id, "is not exist")
        else:
            print("data has been load ", video_read_id)

    l_train = label_train[:train_id]
    l_test = label_test[:test_id]
    print("false-ratio: train:%.4f,test:%.4f" % (1 - np.average(l_train), 1 - np.average(l_test)))
    print("data-sum: train:%d,test%d" % (train_id, test_id))
    print("false-sum: train:%.0f,test:%.0f" % (train_id - np.sum(l_train), test_id - np.sum(l_test)))
    np.savetxt("./test_result/train_label.csv", l_train, delimiter=',')
    np.savetxt("./test_result/test_label.csv", label_test[:test_id], delimiter=',')
