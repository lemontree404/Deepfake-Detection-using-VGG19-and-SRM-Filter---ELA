import os
from ela import ELA
import cv2
import sys

src_file = sys.argv[1]
dest_file = sys.argv[2]

if dest_file not in os.listdir():
    os.mkdir(dest_file)

for i in os.listdir(src_file):
    ELA(f"{src_file}/{i}").save(f"{dest_file}/{i}")