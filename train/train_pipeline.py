# 输入视频流，输出label （0，1）
import time
from train import get_data
from log_config import log

if __name__ == "__main__":
    start_at = time.time()
    train_dataset, labels = get_data.read_csv_train_label_data(test=1)

    get_data_at = time.time()
    trainer = ""
    log.logger.info(
        "开始训练%s分类器:数据规模(%d,%d),%d" % (trainer, train_dataset.shape[0], train_dataset.shape[1], labels.shape[0]))

    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("%s:总运行时间%f秒,数据读取时间%f秒,训练时间%f秒" % (trainer, total_con, read_con, train_con))
