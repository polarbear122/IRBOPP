# 读取mat文件为图片

import cv2
import scipy.io as scio

dataFile = 'E:/CodeResp/pycode/DataSet/Supplementary Materials/data/PedestrianImageRecognitionData_Standing_P1.mat'
data = scio.loadmat(dataFile)
print(data)
store_video = data["STOREVIDEO"]
img_file = "from_mat"
i = 0
for img in store_video:  # numpy_images.shape[0]==152
    cv2.imwrite(img_file + "/" + str(i) + ".jpg", img)
    i += 1
