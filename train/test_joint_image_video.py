import time

import numpy as np

import calculate.calculate as cal
import toolkit.read_data as read_data
from log_config import log
from toolkit.get_data import load_model


def read_data_to_test():
    pass


if __name__ == "__main__":
    start_at = time.time()
    log.logger.info("联合测试开始-------------------------------------------")
    raw_image_data, raw_image_label, video_len_list = read_data.read_csv_train_label_data(data_id=2, output_type=1)
    raw_video_data, raw_video_label = read_data.read_csv_train_label_data(data_id=4, output_type=1)
    get_data_at = time.time()
    log.logger.info("测试SGD分类器, data大小(%d,%d),%d" %
                    (raw_image_data.shape[0], raw_image_data.shape[1], raw_image_label.shape[0]))
    # 修改原始数据格式，图像级别检测的数据不需要改变，视频级别数据需要修改
    model_image = load_model("trained_model/SGD_image_unsampled_ml.model")
    model_video = load_model("trained_model/SGD_video_unsampled_ml.model")
    log.logger.info("video len list : %s" % len(video_len_list))
    print("raw_image_data shape: ", raw_image_data.shape)
    print("raw_video_data shape: ", raw_video_data.shape)
    y_pre_image = model_image.predict(raw_image_data)
    y_pre_video = model_video.predict(raw_video_data)
    #  初始化迭代条件
    start_position, end_position = 0, 0
    y_pre_joint = np.zeros(len(y_pre_image))
    for video_num in range(len(video_len_list)):
        end_position += video_len_list[video_num]
        for iter_x in range(start_position, end_position):
            if iter_x - start_position <= 5:
                y_pre_joint[iter_x] = y_pre_image[iter_x]
            else:
                y_pre_joint[iter_x] = min(y_pre_image[iter_x], y_pre_video[iter_x - 5 * (video_num + 1)])
    print("y_pre_joint shape,raw_image_label shape:", y_pre_joint.shape, raw_image_label)
    print("____________________________")
    print("predict shape", y_pre_image.shape, y_pre_video.shape)
    cal.calculate_all(raw_image_label, y_pre_joint)  # 评估计算结果
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("%s:总运行时间%f秒,数据读取时间%f秒,测试时间%f秒" % ("SGD", total_con, read_con, train_con))
