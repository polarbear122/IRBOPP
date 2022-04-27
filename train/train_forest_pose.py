import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from train.get_data import get_train_data


def train_forest_pose(train_dataset, labels):
    x_train, x_test, y_train, y_test = train_test_split(train_dataset, labels, test_size=0.2, random_state=101)
    # 评估回归性能
    # criterion ：
    # 回归树衡量分枝质量的指标，支持的标准有三种：
    # 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，
    # 这种方法通过使用叶子节点的均值来最小化L2损失
    # 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
    # 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失

    # 此处使用mse
    forest = RandomForestRegressor(n_estimators=1000,
                                   criterion='squared_error',
                                   random_state=1,
                                   n_jobs=-1)
    forest.fit(x_train, y_train)
    train_data_score = forest.score(x_train, y_train) * 100
    test_data_score = forest.score(x_test, y_test) * 100
    print("训练集正确率:%0.3f%%\n" % train_data_score)
    print("测试集正确率:%0.3f%%\n" % test_data_score)
    # y_train_pred = forest.predict(x_train)
    # y_test_pred = forest.predict(x_test)
    # print('MSE train: %.3f, test: %.3f' % (
    #     mean_squared_error(y_train, y_train_pred),
    #     mean_squared_error(y_test, y_test_pred)))
    # print('R^2 train: %.3f, test: %.3f' % (
    #     r2_score(y_train, y_train_pred),
    #     r2_score(y_test, y_test_pred)))
    s = pickle.dumps(forest)
    with open(file="trained_model/forest.model", mode="wb+") as f:
        f.write(s)


if __name__ == "__main__":
    train_data_shape = 7  # 训练的数据的列的大小为7，总训练的数据格式为(number_of_data,7)
    train_dataset, labels = np.zeros((1, train_data_shape), float), np.zeros((1, 1), float)
    for i in range(1, 347):
        video_id = "video_" + str(i).zfill(4)
        xml_anno_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
        alpha_pose_path = "E:/CodeResp/pycode/DataSet/pose_result/" + video_id + "/alphapose-results.json"
        x, y = get_train_data(xml_anno_path, alpha_pose_path, video_id)
        if x.shape[1] == train_data_shape:
            train_dataset = np.concatenate((train_dataset, x))
            labels = np.concatenate((labels, y))

    np.savetxt("trained_model/forest_data.csv", train_dataset, delimiter=',')
    np.savetxt("trained_model/forest_label.csv", labels, delimiter=',')
    train_dataset = np.asarray(train_dataset)
    labels = np.asarray(labels)
    train_forest_pose(train_dataset, labels.ravel())
