import cv2
from imgaug import augmenters as iaa
import os
import shutil

# 图片文件相关路径
img_path = r""
saved_img_path = r''
filelist = [s for s in os.listdir(img_path)]
aug = iaa.Rain(drop_size=0.01, speed=0.03)
# aug2 = iaa.imgaug.augmenters.contrast.LinearContrast(0.7)

cnt = 0
for item in filelist:
    img = cv2.imread(os.path.join(img_path, item))
    img_aug = aug(images=[img])
    # img_aug = aug2(images=img_aug)
    cv2.imwrite(os.path.join(saved_img_path, item), img_aug[0])
    print(cnt)
    cnt += 1