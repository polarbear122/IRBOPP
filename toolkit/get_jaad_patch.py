# 生成行人bounding box的图像块，原始数据是大小不一的
# 出于训练神经网络的考虑，结果需要resize到大小一致
# 数据集评价宽长比为0.4/1，将所有图像resize为80-200，不足处补0
import pandas as pd
from PIL import Image

from config import jaad_total_img


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


def limit_axis(xbr, x_0, x_1):
    if xbr < x_0:
        xbr = x_0
    elif xbr >= x_1:
        xbr = x_1 - 1
    return xbr


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
    xtl, xbr, ytl, ybr = limit_axis(
        xtl, 0, 1920), limit_axis(xbr, 0, 1920), limit_axis(ytl, 0, 1080), limit_axis(ybr, 0, 1080)
    need_continue = False
    xtl, ytl, xbr, ybr = round(xtl), round(ytl), round(xbr), round(ybr)
    if ybr <= ytl or xbr <= xtl:
        need_continue = True
        xtl, ytl, xbr, ybr = xtl - 10, ytl - 10, xbr + 10, ybr + 10
    return xtl, ytl, xbr, ybr, need_continue


# 整个的人体图像patch,未经过resize，保留原始大小
def generate_img_patch_init(all_pose):
    image_path = jaad_total_img + "video_"
    process_raw_img = 0
    for i in range(len(all_pose)):
        pose = all_pose[i]
        uuid, v_id, idx, img_id, label = int(pose[0]), int(pose[1]), int(pose[2]), int(pose[3]), int(pose[86])
        xtl, ytl, xbr, ybr, need_continue = get_box_from_keypoints(pose[4:86], True)
        xtl_f, ytl_f, xbr_f, ybr_f, need_continue_f = get_box_from_keypoints(pose[4:86], False)
        if need_continue:
            print(uuid, need_continue)
        need_gen_img = False
        if need_gen_img:
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


if __name__ == "__main__":
    all_pose = pd.read_csv('../save_data/new_pose.csv',
                           header=None, sep=',', encoding='utf-8').values
    generate_img_patch_init(all_pose)
