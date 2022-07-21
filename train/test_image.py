import time
import numpy as np
import calculate.calculate as cal
import toolkit.read_data as read_data
from log_config import log
from toolkit.tool import load_model

if __name__ == "__main__":
    start_at = time.time()
    log.logger.info("单帧测试开始-------------------------------------------")
    all_norm_pose, all_label, all_video_length_list = read_data.read_data_track_test()
    get_data_at = time.time()
    log.logger.info("测试联合分类器, 测试data大小(%d,%d),%d" %
                    (all_norm_pose.shape[0], all_norm_pose.shape[1], all_label.shape[0]))

    forest_model = load_model("../train/trained_model/image/Forest_image.model")
    # sgd_model = load_model("../train/trained_model/SGD_image_ml.model")
    log.logger.info("video len list : %s" % len(all_video_length_list))
    y_forest_pred = forest_model.predict(all_norm_pose)
    y_pred = np.zeros(len(y_forest_pred))
    for i in range(len(y_forest_pred)):
        if y_forest_pred[i] < 0.5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    np.savetxt("../train/trained_model/image/Forest_image_y_pred.csv", y_pred, delimiter=',')
    print("y_pre_joint shape,raw_image_label shape:", y_pred.shape)
    cal.calculate_all(y_pred, all_label.ravel())
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("%s:总运行时间%f秒,数据读取时间%f秒,测试时间%f秒" % ("联合训练", total_con, read_con, train_con))
