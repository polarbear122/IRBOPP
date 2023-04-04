import time

import numpy as np

import calculate.calculate as cal
import toolkit.read_data as read_data
from toolkit.tool import load_model


def read_data_to_test():
    pass


# 最近的数字加入头部，队列头部权重最大
def add_one_num(joint_list: list, num: int) -> None:
    length = len(joint_list)
    for __i in range(1, length):
        joint_list[__i] = joint_list[__i - 1]
    joint_list[0] = num


def calculate_result(joint_list: list) -> int:
    weight = 0
    for __i in range(len(joint_list)):
        weight += joint_list[__i] / (2 + __i)
    if weight < 0.5:
        return 0
    else:
        # print(joint_list)
        return 1


# 以resize的形式进行判断，将cal__f_n帧图像reshape为1行数据，求和之后，如果该行的和大于2，则设置为label=1
# jaad的真实数据则reshape为30列之后，算最大值
def calculate_joint_img_reshape(__y_pred_joint: np.array, __test_label: list) -> None:
    cal__f_n = 15  # 设置计算的帧数，1s对应30帧
    reshape_len = len(__y_pred_joint) // cal__f_n * cal__f_n  # 避免长度不规整导致不能reshape
    __y_pred_joint = __y_pred_joint[:reshape_len].reshape((-1, cal__f_n))
    y_pred_joint_sum = np.sum(__y_pred_joint, axis=1)  # 多帧图像得到的预测标签求和
    for __i in range(len(y_pred_joint_sum)):
        if y_pred_joint_sum[__i] >= 2:
            y_pred_joint_sum[__i] = 1
        else:
            y_pred_joint_sum[__i] = 0

    test_label_max = np.max(test_label[:reshape_len].reshape((-1, cal__f_n)), axis=1)  # 多帧图像取一个最大值
    cal.calculate_all(test_label_max, y_pred_joint_sum)  # 基于30张图像评估计算结果


# 考虑一个人look车辆之后的0.2s内，即某一帧label=1的之后6帧，其依然可以对车辆有一个监测
# 设置算法如下：如果某帧label=1，则遍历之后六帧，全部设置为label=1
def calculate_joint_img_delay(__y_pred_joint: np.array, __test_label: list):
    delay_frame_num = 6
    label_length = len(__y_pred_joint)
    y_pred_joint_delay = np.zeros(label_length)
    for __i in range(label_length):
        if __y_pred_joint[__i] == 1:
            for __j in range(delay_frame_num):
                if __i + __j < label_length - 1:
                    y_pred_joint_delay[__i + __j] = 1
    cal.calculate_all(__test_label, y_pred_joint_delay)


if __name__ == "__main__":
    start_at = time.time()
    print("联合测试开始-------------------------------------------")
    train_norm_pose, train_label, train_video_length_list, test_norm_pose, test_label, test_video_length_list \
        = read_data.read_data_no_track()
    get_data_at = time.time()
    print("测试联合分类器, 测试data大小(%d,%d),%d" %
          (test_norm_pose.shape[0], test_norm_pose.shape[1], test_label.shape[0]))
    # 修改原始数据格式，图像级别检测的数据不需要改变，视频级别数据需要修改
    forest_model = load_model("../train/trained_model/Forest_image.model")
    sgd_model = load_model("../train/trained_model/SGD_image_ml.model")
    print("video len list : %s" % len(test_video_length_list))

    y_forest_pred = forest_model.predict(test_norm_pose)
    for i in range(len(y_forest_pred)):
        if y_forest_pred[i] < 0.5:
            y_forest_pred[i] = 0
        else:
            y_forest_pred[i] = 1
    # print("y_forest_pred", y_forest_pred)
    y_sgd_pred = sgd_model.predict(test_norm_pose)
    #  初始化迭代条件
    start_position, end_position = 0, 0
    y_pred_joint = np.zeros(len(y_forest_pred))

    init_joint_list = [0] * 10
    for i in range(len(y_forest_pred)):
        if y_forest_pred[i] == y_sgd_pred[i]:
            y_pred_joint[i] = y_forest_pred[i]
            add_one_num(init_joint_list, y_forest_pred[i])
        else:
            result = calculate_result(init_joint_list)
            y_pred_joint[i] = result
            add_one_num(init_joint_list, result)
    np.savetxt("y_pred_joint.csv", y_pred_joint, delimiter=',')
    print("y_pre_joint shape,raw_image_label shape:", y_pred_joint.shape)
    print("____________________________")
    # print("predict shape", y_pre_image.shape, y_pre_video.shape)
    # cal.calculate_all(test_label, y_pre_joint)  # 基于单张图像评估计算结果

    # 还需要基于一段时间的联合结果来判断，calculate_frame参数控制计算的帧数
    cal.calculate_all(y_forest_pred, test_label)
    calculate_joint_img_reshape(y_pred_joint, test_label)
    calculate_joint_img_delay(y_forest_pred, test_label)
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    print("%s:总运行时间%f秒,数据读取时间%f秒,测试时间%f秒" % ("联合训练", total_con, read_con, train_con))
