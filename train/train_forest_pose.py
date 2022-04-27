import pickle
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from train import get_data
from log_config import log


def train_forest_pose(train_dataset, labels):
    x_train, x_test, y_train, y_test = train_test_split(train_dataset, labels, test_size=0.4, random_state=1)
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
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
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
    start_at = time.time()
    train_dataset, labels = get_data.read_csv_train_label_data()
    get_data_at = time.time()
    log.logger.info(
        "forest data to be trained:(%d,%d),%d" % (train_dataset.shape[0], train_dataset.shape[1], labels.shape[0]))
    train_forest_pose(train_dataset, labels)
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("forest:总运行时间%f秒,数据读取%f秒,训练%f秒" % (total_con, read_con, train_con))
