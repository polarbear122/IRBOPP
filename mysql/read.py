# 导入需要使用到的数据模块
import pymysql
import time

if __name__ == "__main__":
    st = time.time()
    # 重新建立数据库连接
    db = pymysql.connect(host="127.0.0.1",
                         database="irbopp",
                         user="root",
                         password="polarbear",
                         port=3306,
                         charset='utf8')
    cursor = db.cursor()
    result = []
    # 查询数据库并打印内容
    for i in range(20):
        cursor.execute("select * from alpha_pose where id between " + str(i * 10000) + " and " + str((i+1) * 10000))
        result.append(cursor.fetchall())
        print("len result: ", len(result))
    end = time.time()
    print("time consume : ", end - st)
    # print(results)
    # for row in results:
    #     print(row)
    # 关闭
    cursor.close()
    db.commit()
    db.close()
