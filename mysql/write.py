# 导入需要使用到的数据模块
import pandas as pd
import pymysql
import numpy as np


def read_all_single_data():
    data_path = "D:/CodeResp/IRBOPP/train/train_data/iou/data_by_video/all_single/"
    label_path = "D:/CodeResp/IRBOPP/train/train_data/iou/data_by_video/all_single/"
    single_pose = np.loadtxt(data_path + "data1.csv", dtype=np.float_, delimiter=',')
    single_label = np.loadtxt(label_path + "label1.csv", dtype=np.float_, delimiter=',')
    video_length_list = [len(single_label)]
    for str_id in range(2, 347):
        try:
            pose_arr = np.loadtxt(data_path + "data" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
            label_arr = np.loadtxt(label_path + "label" + str(str_id) + ".csv", dtype=np.float_, delimiter=',')
            print("shape:", pose_arr.shape, label_arr.shape)
            video_length_list.append(len(label_arr))
            single_pose = np.concatenate((single_pose, pose_arr), axis=0)
            single_label = np.concatenate((single_label, label_arr), axis=0)
        except OSError:
            print("data or label ", str_id, "is not exist")
        else:
            print("data has been load ", str_id)
    return single_pose, single_label, video_length_list


if __name__ == "__main__":
    # 建立数据库连接
    db = pymysql.connect(host="127.0.0.1",
                         database="irbopp",
                         user="root",
                         password="polarbear",
                         port=3306,
                         charset='utf8')
    # 获取游标对象
    cursor = db.cursor()
    # 插入数据语句
    insert_sql = """insert into alpha_pose (uniq_pose_id, video_id, img_frame_id,x0,y0,c0, x1,y1,c1, x2,y2,c2, x3,y3,c3,
     x4,y4,c4, x5,y5,c5, x6,y6,c6, x7,y7,c7, x8,y8,c8, x9,y9,c9, x10,y10,c10, x11,y11,c11, x12,y12,c12, x13,y13,c13, 
     x14,y14,c14, x15,y15,c15, x16,y16,c16, x17,y17,c17, x18,y18,c18, x19,y19,c19, x20,y20,c20, x21,y21,c21,
      x22,y22,c22, x23,y23,c23, x24,y24,c24, x25,y25,c25,x_mid,y_mid,width,height,label) 
    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    # 读入数据86列数据
    data, label, video_length_list = read_all_single_data()
    x_array, y_array = np.asarray(data), np.asarray(label)
    np.savetxt("test_data" + str(1) + ".csv", x_array, delimiter=',')
    np.savetxt("test_label" + str(1) + ".csv", y_array, delimiter=',')
    # 迭代读取每行数据# values中元素有个类型的强制转换，否则会出错的
    # 应该会有其他更合适的方式，可以进一步了解
    for i in range(0, len(data)):
        d = data[i]
        # values = (int(num), str(date), float(sale))
        values = (int(i), int(d[0]), int(d[1]))
        for j in range(26):
            values = values + (float(d[j * 3 + 2]), float(d[j * 3 + 2 + 1]), float(d[j * 3 + 2 + 2]))
        values = values + (float(d[80]), float(d[81]), float(d[82]), float(d[83]), int(d[84]))
        result = cursor.execute(insert_sql, values)
        print(result)
    # 关闭游标，提交，关闭数据库连接
    # 如果没有这些关闭操作，执行后在数据库中查看不到数据
    cursor.close()
    db.commit()
    db.close()
