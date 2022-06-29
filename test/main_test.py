import random
import numpy as np
import cv2


def generate_random_video_list():
    # randint Return random integer in range [a, b], including both end points.
    # randint函数会返回包含两边终点的一个int值
    __i = random.randint(1, 347)
    print(__i)
    rand_schedule = np.random.permutation(range(346)).tolist()  # 返回一个[0-9]中各个数字的数组

    print(rand_schedule[0:207])
    print(rand_schedule[207:347])


def video_read():
    video_path = "D:/CodeResp/jaad_data/AlphaReidResultNoFast/video_"
    for v_id in range(347):
        v_name = video_path + str(v_id).zfill(4) + "/AlphaPose_video_" + str(v_id).zfill(4) + ".mp4"
        print(v_name)
        # 1.初始化读取视频对象
        cap = cv2.VideoCapture(v_name)

        # 2.循环读取图片
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow("frame", frame)
            else:
                print("视频播放完成！")
                break

            # 退出播放
            key = cv2.waitKey(25)
            if key == 27:  # 按键esc
                break
        # 3.释放资源
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_read()
