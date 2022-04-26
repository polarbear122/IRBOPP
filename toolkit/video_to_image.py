import os

import cv2


def video_to_image(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    is_open = cap.isOpened
    print(is_open)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(fps, width, height)

    i = 0
    while is_open:
        (flag, frame) = cap.read()
        file_name = str(i) + '.jpg'
        if flag:
            cv2.imwrite(save_path + file_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            i = i + 1
        else:
            break
    print("end")


# 如果目录不存在则创建目录，存在则打印存在
def make_dir_file(path):
    url = path
    if os.path.exists(url):
        print("exist")
        exit(0)
    else:
        os.mkdir(url)


def main():
    video_id_start = 1
    end = 347  # 共有1--346号视频
    for i in range(video_id_start, end):
        video_path = "E:/CodeResp/pycode/DataSet/JAAD_clips/video_" + str(i).zfill(4) + ".mp4"
        save_path = "E:/CodeResp/pycode/DataSet/JAAD_image/video_" + str(i).zfill(4) + "/"
        make_dir_file(save_path)
        video_to_image(video_path, save_path)


if __name__ == "__main__":
    main()
