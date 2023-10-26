import numpy as np
import cv2
import os
import sys

src_file = sys.argv[1]
dest_file = sys.argv[2]

srm_filter = np.array([[-1, 2, -2, 2, -1],
           [2, -6, 8, -6, 2],
           [-2, 8, -12, 8, -2],
           [2, -6, 8, -6, 2],
           [-1, 2, -2, 2, -1]],dtype=float)

srm_filter /= 12.0

if dest_file not in os.listdir():
    os.mkdir(dest_file)

for i in os.listdir(src_file):

    img = cv2.imread(f"{src_file}/{i}")
    p_img = cv2.filter2D(img,-1,srm_filter)
    cv2.imwrite(f"{dest_file}/{i}",p_img)