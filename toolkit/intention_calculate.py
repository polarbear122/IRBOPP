# 课程：人机协作与交互作业
import json
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_line(x_axis_data, y_axis_data):
    # plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='angle-time')  # 'bo-'表示蓝色实线，数据点实心原点标注
    x = [0, 9]
    y = [5, 5]
    plt.plot(x, y, 'r*--', alpha=0.5, linewidth=1, label='angle=5°')
    x = [0, 9]
    y = [50, 50]
    plt.plot(x, y, 'r*--', alpha=0.5, linewidth=1, label='angle=50°')
    plt.legend()  # 显示上面的label
    plt.xlabel('time/s')  # x_label
    plt.ylabel('angle/°')  # y_label
    plt.show()


def readJson(json_path):
    json_data = open(json_path)
    json_string = json_data.read()
    j = json.loads(json_string)
    return j


def pose_compute(json_path):
    pose_result = readJson(json_path)
    x_axis_data, y_axis_data = [], []
    pre_one, nearest_id_one, pre_two, nearest_id_two = False, 0, False, 0
    img_id_list = []
    st = time.time() * 1000
    count = 0
    for data in pose_result:
        count += 1
        key_points = data["keypoints"]
        p13_x = key_points[39]
        p13_y = key_points[40]
        p15_x = key_points[45]
        p15_y = key_points[46]

        p14_x = key_points[42]
        p14_y = key_points[43]
        p16_x = key_points[48]
        p16_y = key_points[49]

        angle = get_cross_angle(p13_x, p13_y, p15_x, p15_y, p14_x, p14_y, p16_x, p16_y)
        img_id = get_image_id(data["image_id"])
        if img_id not in img_id_list:
            img_id_list.append(img_id)
            x_axis_data.append(img_id / 30)
            y_axis_data.append(angle)
            if angle > 50 and pre_two and pre_one:
                if nearest_id_two < nearest_id_one < img_id:
                    print("img_id:", nearest_id_two, nearest_id_one, img_id)

            if angle > 50:
                pre_two = True
                nearest_id_two = img_id
            if angle < 5:
                pre_one = True
                nearest_id_one = img_id
            # box = data["box"]
            # img = cv2.imread("E:/CodeResp/pycode/DataSet/JAAD_image/video_0014/" + data["image_id"])
            # cv2.circle(img, (int(box[0]), int(box[1])), 3, (0, 0, 255), 4)
            # cv2.imshow("./inten.jpg", img)
            # cv2.waitKey(1)
    # print(x_axis_data)
    plot_line(x_axis_data, y_axis_data)
    end = time.time() * 1000
    sum_time = end - st
    average_time = sum_time / count
    print("sum time", sum_time, "ave time", average_time)


def get_image_id(image_id):
    result = ""
    for i in image_id:
        if i == ".":
            break
        result += i
    return int(result)


def get_cross_angle(x1, y1, x2, y2, x3, y3, x4, y4):
    arr_0 = np.array([(x2 - x1), (y2 - y1)])
    arr_1 = np.array([(x4 - x3), (y4 - y3)])

    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))
    return np.arccos(cos_value) * (180 / np.pi)


def main():
    json_path = "E:\CodeResp\pycode\DataSet\pose_result/alphapose-results-0014-track.json"
    pose_compute(json_path)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
