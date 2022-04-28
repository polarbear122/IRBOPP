import pickle
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import calculate.calculate as cal
from log_config import log
from train import get_data


def train_forest_pose(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data, all_labels, random_state=1, train_size=0.8, test_size=0.2)
    # 此处使用squared_error
    # clf = RandomForestRegressor(n_estimators=1000,criterion='squared_error',random_state=1,n_jobs=-1)
    clf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=0, verbose=True,n_jobs=-1)
    clf.fit(x_train, y_train)
    print("数据分割完成，开始训练forest分类器")
    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    s = pickle.dumps(clf)
    with open(file="trained_model/svm.model", mode="wb+") as f:
        f.write(s)


if __name__ == "__main__":
    start_at = time.time()
    train_dataset, labels = get_data.read_csv_train_label_data(test=False)
    get_data_at = time.time()
    log.logger.info(
        "forest data to be trained:(%d,%d),%d" % (train_dataset.shape[0], train_dataset.shape[1], labels.shape[0]))
    train_forest_pose(train_dataset, labels)
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    log.logger.info("forest:总运行时间%f秒,数据读取%f秒,训练%f秒" % (total_con, read_con, train_con))
