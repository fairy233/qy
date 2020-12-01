import cv2
import numpy as np
import random
import os


# pic_name      切割后文件名
# pic           原hdr图片
# num           切割数目
# width         切割宽度
# height        切割高度
def split_pic(pic_name, pic, num, width, height):
    width_t = pic.shape[1]
    height_t = pic.shape[0]
    cropPath = './cropHdr'
    isExists = os.path.exists(cropPath)
    if not isExists:
        print(cropPath + ' success')
        os.makedirs(cropPath)
    else:
        print(cropPath + ' is existed')
    for i in range(0, num, 1):
        width_b = random.randint(-1, width_t - width)
        height_b = random.randint(-1, height_t - height)
        pic_t = pic[height_b:height_b + height, width_b:width_b + width, :]
        # cv2.imshow(str(i), pic_t)
        cropName = pic_name + '_' + str(i) + '.hdr'
        # print(cropPath + '/' + cropName)
        cv2.imwrite(cropPath + '/' + cropName, pic_t)


image_path = './'
files = os.listdir(image_path)
data_extensions = ['.hdr', '.exr', '.jpeg']
for file_name in files:
    if any(
        file_name.lower().endswith(extension)
        for extension in data_extensions
    ):
        # pic_name = os.path.splitext(file_name)[0]
        hdr = cv2.imread(file_name, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        # cv2.imshow('hdr', hdr)
        split_pic(file_name, hdr, 5, 1000, 1020)
