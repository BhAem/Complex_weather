import os
import random
import numpy as np
import cv2

src_img_path = r""
dst_img_path = r""

imgs = [s.split(".")[0] for s in os.listdir(src_img_path)]

cnt = 0
for img_name in imgs:
    img_path = os.path.join(src_img_path, img_name+".jpg")
    img = cv2.imread(img_path)
    img = img / 255
    # lowlight_param = random.uniform(2, 5)
    lowlight_param = 3
    img = np.power(img, lowlight_param)
    img = (img*255).astype(np.uint8)
    cv2.imwrite(os.path.join(dst_img_path, img_name+"_night.jpg"), img)
    print(cnt, "  ", lowlight_param, "  " ,img_name)
    cnt += 1
    # cv2.imshow("demo", img)
    # cv2.waitKey(0)