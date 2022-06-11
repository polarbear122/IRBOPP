import time

import numpy as np

import calculate.calculate as cal
import toolkit.read_data as read_data
from log_config import log
from toolkit.get_data import load_model


def read_data_to_test():
    pass


# 最近的数字加入头部，队列头部权重最大
def add_one_num(joint_list: list, num: int):
    length = len(joint_list)
    for __i in range(1, length):
        joint_list[__i] = joint_list[__i - 1]
    joint_list[0] = num


def calculate_result(joint_list: list):
    weight = 0
    for __i in range(len(joint_list)):
        weight += joint_list[__i] / (2 + __i)
    if weight < 0.5:
        return 0
    else:
        # print(joint_list)
        return 1


if __name__ == "__main__":
    start_at = time.time()
    log.logger.info("联合测试开始-------------------------------------------")
    train_norm_pose, train_label, train_video_length_list, test_norm_pose, test_label, test_video_length_list \
        = read_data.read_csv_data_random(data_id=2)
    get_data_at = time.time()
    log.logger.info("测试联合分类器, 测试data大小(%d,%d),%d" %
                    (test_norm_pose.shape[0], test_norm_pose.shape[1], test_label.shape[0]))
    # 修改原始数据格式，图像级别检测的数据不需要改变，视频级别数据需要修改
    forest_model = load_model("../train/trained_model/Forest_image_ml.model")
    sgd_model = load_model("../train/trained_model/SGD_image_ml.model")
    log.logger.info("video len list : %s" % len(test_video_length_list))
    # raw_video_data, raw_video_label = read_data.read_csv_train_label_data(data_id=4, output_type=1)
    # model_video = load_model("trained_model/SGD_video_unsampled_ml.model")
    # print("raw_video_data shape: ", raw_video_data.shape)
    # y_pre_video = model_video.predict(raw_video_data)
    y_forest_pred = forest_model.predict(test_norm_pose)
    for __i in range(len(y_forest_pred)):
        if y_forest_pred[__i] < 0.5:
            y_forest_pred[__i] = 0
        else:
            y_forest_pred[__i] = 1
    # print("y_forest_pred", y_forest_pred)
    y_sgd_pred = sgd_model.predict(test_norm_pose)
    #  初始化迭代条件
    start_position, end_position = 0, 0
    y_pre_joint = np.zeros(len(y_forest_pred))
    # for video_num in range(len(video_len_list)):
    #     end_position += video_len_list[video_num]
    #     for iter_x in range(start_position, end_position):
    #         if iter_x - start_position <= 5:
    #             y_pre_joint[iter_x] = y_pre_image[iter_x]
    #         else:
    #             y_pre_joint[iter_x] = max(y_pre_image[iter_x], y_pre_video[iter_x - 5 * (video_num + 1)])
    init_joint_list = [0] * 10
    for __i in range(len(y_forest_pred)):
        if y_forest_pred[__i] == y_sgd_pred[__i]:
            y_pre_joint[__i] = y_forest_pred[__i]
            add_one_num(init_joint_list, y_forest_pred[__i])
        else:
            result = calculate_result(init_joint_list)
            y_pre_joint[__i] = result
            add_one_num(init_joint_list, result)
    np.savetxt("y_pre_joint.csv", y_pre_joint, delimiter=',')
    print("y_pre_joint shape,raw_image_label shape:", y_pre_joint.shape)
    print("____________________________")
    # print("predict shape", y_pre_image.shape, y_pre_video.shape)
    # cal.calculate_all(test_label, y_pre_joint)  # 基于单张图像评估计算结果

    # 还需要基于一段时间的联合结果来判断，这里设置为1s 30帧判断
    useful_length = len(y_pre_joint) // 30 * 30
    y_pre_joint = y_pre_joint[:useful_length].reshape((-1, 30))
    y_pre_joint_max = np.sum(y_pre_joint, axis=1)  # 30帧图像取一个最大值
    for __i in range(len(y_pre_joint_max)):
        if y_pre_joint_max[__i] >= 2:
            y_pre_joint_max[__i] = 1
        else:
            y_pre_joint_max[__i] = 0
    test_label_max = np.max(test_label[:useful_length].reshape((-1, 30)), axis=1)  # 30帧图像取一个最大值
    cal.calculate_all(test_label_max, y_pre_joint_max)  # 基于30张图像评估计算结果

    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("%s:总运行时间%f秒,数据读取时间%f秒,测试时间%f秒" % ("联合训练", total_con, read_con, train_con))
