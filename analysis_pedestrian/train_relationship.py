""""
猜测：
对于道路上的行人：
如果行人一直没有穿越马路的意图，行人与驾驶员的眼神交流会促使驾驶员更加激进。
如果行人有穿越马路意图，则会促使驾驶员减速。
"""
import numpy as np
import pandas as pd

import config
from calculate import calculate


def normalize_read(data_path: str, _data_list: list):
    # 先初始化向量
    _data = pd.read_csv(data_path + "data" + str(_data_list[0]) + ".csv", header=None, sep=',', encoding='utf-8')
    for v_id in _data_list[1:]:
        try:
            _pose_arr = pd.read_csv(data_path + "data" + str(v_id) + ".csv", header=None, sep=',', encoding='utf-8')
            _data = np.concatenate((_data, _pose_arr), axis=0)
        except OSError:
            print("data or label ", v_id, "is not exist")
        else:
            print("data has been load ", v_id)
    return _data


def generate_predict_data():
    data = normalize_read("./data/", config.cross_list)
    vehicle_behavior = data[:, 2]
    look_label = data[:, -2]
    cross_label = data[:, -1]
    count_arr = [0, 0, 0, 0, 0]
    for _i in range(len(vehicle_behavior)):
        for _j in range(5):
            if vehicle_behavior[_i] == _j:
                count_arr[_j] += 1
    print(count_arr)
    for _i in range(len(vehicle_behavior)):
        if vehicle_behavior[_i] == 1 or vehicle_behavior[_i] == 3 or vehicle_behavior[_i] == 4:
            vehicle_behavior[_i] = 0
        elif vehicle_behavior[_i] == 2:
            vehicle_behavior[_i] = 1
    print(vehicle_behavior.sum(), len(vehicle_behavior))
    no_look_predict = cross_label

    look_predict = np.zeros((len(cross_label),))
    for _i in range(15, len(cross_label)):
        if look_label[_i - 15] == 0 and cross_label[_i - 15] == 1:
            look_predict[_i] = 0
        elif look_label[_i - 10] == 1 and cross_label[_i - 10] == 0:
            look_predict[_i] = 1
        else:
            look_predict[_i] = cross_label[_i]
    # calculate.calculate_all(vehicle_behavior, no_look_predict)

    calculate.calculate_all(vehicle_behavior, look_predict)


if __name__ == "__main__":
    generate_predict_data()
