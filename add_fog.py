import numpy as np
import os
import cv2
import math
from numba import jit
import random

img_path = r""
saved_img_path = r''
imgs = [s.split(".")[0] for s in os.listdir(img_path)]

cnt = 0
for img in imgs:
    image_path = os.path.join(img_path, img+".jpg")
    image = cv2.imread(image_path)
    # seed = random.randint(1, 3)
    seed = 5
    # for i in range(10):
    @jit()
    def AddHaz_loop(img_f, center, size, beta, A):
        (row, col, chs) = img_f.shape

        for j in range(row):
            for l in range(col):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-beta * d)
                img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
        return img_f

    img_f = image / 255
    (row, col, chs) = image.shape
    A = 0.5
    # beta = 0.08
    beta = 0.01 * seed + 0.05
    size = math.sqrt(max(row, col))
    center = (row // 2, col // 2)
    foggy_image = AddHaz_loop(img_f, center, size, beta, A)
    img_f = np.clip(foggy_image*255, 0, 255)
    img_f = img_f.astype(np.uint8)
    img_name = os.path.join(saved_img_path, img+".jpg")
    cv2.imwrite(img_name, img_f)
    print(seed, "  ", cnt)
    cnt += 1
