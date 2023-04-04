import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
import train.test_joint_image_video as jo
from JAAD_2.jaad_data import JAAD


def print_cuda():
    print("torch version:", torch.__version__)
    print(torch.cuda.is_available())  # cuda是否可用
    print("number of gpu:", torch.cuda.device_count())  # 返回GPU的数量
    print("gpu name", torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())


def print_numpy():
    print(np.amax([0, 0, 0, 0, 0], axis=0))
    pose_array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    labels = np.array([[0], [1], [2]])
    print(pose_array[:, 1])
    a = np.array([labels[0]])
    b = np.array([labels[0]])
    c = np.concatenate((a, b), axis=0)
    print(a, b, "ad", c)


def test_img_resize():
    img = cv2.imread('./Pictures/python.png', cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ', img.shape)

    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)
    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_python_input():
    init_list = [0] * 10
    print(init_list)
    jo.add_one_num(init_list, 1)
    print(init_list)
    re = jo.calculate_result(init_list)
    print(re)
    print(init_list)


# 测试读取csv数据
def test_file_read():
    data_path = "halpe26_data/data_by_video/single/"

    for str_id in range(2, 10):
        try:
            f = open(data_path + "data" + str(str_id) + ".csv", encoding='utf-8')
            pose_arr = pd.read_csv(f)
            print("shape:", pose_arr.shape)
        except OSError:
            print("data or label ", str_id, "is not exist")
        else:
            print("data has been load ", str_id)


# 求numpy最大值
def test_numpy_max():
    y_pre_joint = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])
    y_pre_joint = y_pre_joint.reshape((-1, 2))
    y_pre_joint_max = np.max(y_pre_joint, axis=1)
    print(y_pre_joint)
    print(y_pre_joint_max)


# 测试numpy的排序功能
def test_np_sort():
    data_path = "train/halpe26_reid/"
    train_pose = pd.read_csv(data_path + "data1.csv", header=None, sep=',', encoding='utf-8').values
    a = train_pose
    a = a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
    print(a)
    return a


# 生成随机数组
def random_int_list():
    all_list = config.all_data_list
    train_list = []
    for i in range(242):
        j = random.randint(0, len(all_list) - 1)
        train_list.append(all_list[j])
        del all_list[j]
    test_list = all_list
    print("train:", train_list)
    print("test:", test_list)
    return train_list, test_list


def get_result():
    path = 'D:/CodeResp/jaad_data/AlphaReidResultNoFast/video_'
    new_path = 'D:/CodeResp/jaad_data/new/AlphaReidResultNoFast/video_'
    for i in range(1, 347):
        i_name = str(i).zfill(4)
        path_name = path + i_name
        new_path_name = new_path + i_name
        file_name = path_name + '/alphapose-results.json'
        new_file_name = new_path_name + '/alphapose-results.json'
        if not os.path.exists(new_path_name):
            os.mkdir(new_path_name)
        print(file_name, new_file_name)
        # shutil.copy(file_name, new_file_name)


# 校验随机的训练和测试数组是合适的
def test_train_test_list():
    train, test, val = config.jaad_all_videos_train, config.jaad_all_videos_test, config.jaad_all_videos_val
    for i in range(1, 347):
        if i in train and i in test or i in train and i in val or i in test and i in val:
            print("error")
            break
        if i not in train and i not in test and i not in val:
            print(i)
            print("error:not in")
            break


# 测试numpy数组大小能否重新初始化
def numpy_arr_reshape():
    num1 = np.zeros(2)
    num2 = np.zeros(2)
    num1 = np.concatenate((num1, num2))
    print(num1)
    num1 = np.zeros((1, 2))
    print(num1)


# 从jaad数据集中解析出train、val、test数据
def get_txt_file_to_num():
    # file_path = "D:/CodeResp/JAAD-JAAD_2.0/split_ids/all_videos/"
    file_path = "D:/CodeResp/jaad_data/JAAD-JAAD_2.0/split_ids/all_videos/"
    name = "test.txt"
    file = file_path + name
    num_list = []
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            if line:
                num = int(line.split("_")[1])
                num_list.append(num)
    print("len test", len(num_list))
    print("test_list = ", num_list)


def angle_row_wise_v2(l1_arr, l2_arr):
    p1 = np.einsum('ij,ij->i', l1_arr, l2_arr)
    p2 = np.einsum('ij,ij->i', l1_arr, l1_arr)
    p3 = np.einsum('ij,ij->i', l2_arr, l2_arr)
    p4 = p1 / np.sqrt(p2 * p3)
    return np.arccos(np.clip(p4, -1.0, 1.0))


def random_sort_list():
    arr = np.array(range(1, 347, 1))
    print(arr)
    r = np.random.permutation(arr)
    print(r)  # 打乱的索引序列，如[2, 6, 4, 8, 12, 16, 0, 18, 10, 14, 20]


def delete_img():
    # 指定路径
    path = 'D:/CodeResp/jaad_data/patch_face/'
    # for root, dirs, files in os.walk(path):
    #     for name in files:
    #         if name.endswith(".png"):  # 填写规则
    #             # os.remove(os.path.join(root, name))
    #             print("Delete File: " + os.path.join(root, name))
    # D:/CodeResp/jaad_data/patch_img/video_0006/708.jpg D:/CodeResp/jaad_data/patch_img/video_0006/707.jpg
    for v_id in range(1, 347):
        for img_st_id in range(1, 10000):
            img_path = path + 'video_' + str(v_id).zfill(4) + '/' + str(img_st_id) + '.jpg'
            if os.path.exists(img_path):
                img_new_path = path + 'video_' + str(v_id).zfill(4) + '/' + str(img_st_id - 1) + '.jpg'
                print(img_path, img_new_path)
                os.rename(img_path, img_new_path)
        # try:
        #     os.remove(os.path.join(img_path))
        # except FileNotFoundError:
        #     print(v_id, 'no video')


def test_pil_read_img():
    st = time.time()
    for i in range(200):
        raw_img = Image.open('E:/CodeResp/pycode/DataSet/PIE_image/set03/video_0010/' + str(i) + '.jpg')
        save_path = 'D:/CodeResp/pie_data/patch_img/testpil/' + str(i) + '.jpg'
        box = (100, 100, 550, 350)
        region = raw_img.crop(box)
        region.save(save_path)
        # raw_img = cv2.imread('E:/CodeResp/pycode/DataSet/PIE_image/set03/video_0010/' + str(i) + '.jpg', 1)
        # crop = raw_img[100:350, 100:550, :]
        # cv2.imwrite('D:/CodeResp/pie_data/patch_img/testpil/' + str(i) + '.jpg', crop)
    end = time.time()
    print('end-st %f s' % ((end - st) / 200))


# 生成行人的唯一id，由原来的str转换为int
# 由于发现jaad提供了唯一id，所以不再自己生成
def gen_id():
    x = JAAD(data_path='JAAD_2/')
    jaad_data = x.generate_database()
    ped_id = 0
    ped_id_dict = {}
    for video_name in jaad_data:
        anno_dict = jaad_data[video_name]
        ped_name_dict = anno_dict['ped_annotations']
        for ped_name in ped_name_dict:
            if ped_name in ped_id_dict:
                continue
            else:
                ped_id_dict[ped_name] = ped_id
                ped_id += 1
    print(ped_id_dict)


if __name__ == "__main__":
    gen_id()
