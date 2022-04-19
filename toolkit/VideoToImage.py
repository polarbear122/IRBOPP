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


def main():
    video_path = "E:/CodeResp/pycode/DataSet/JAAD_clips/video_0014.mp4"
    save_path = "E:/CodeResp/pycode/DataSet/JAAD_image/video_0014/"
    video_to_image(video_path, save_path)


if __name__ == "__main__":
    main()
