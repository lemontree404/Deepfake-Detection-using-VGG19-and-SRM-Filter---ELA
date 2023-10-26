import cv2
import os
from fake_frame_extractor import frame_extractor

num = 0

if "fake_images" not in os.listdir():
    os.mkdir("fake_images")

for file in os.listdir(".//Data//higher_quality"):
    if file == ".dircksum":
        continue
    else:
        for i in os.listdir(f".//Data//higher_quality//{file}"):
            if i.split(".")[-1] == "avi":
                print(num,file,i)
                frame_extractor(i,file,num)
                num+=10

    