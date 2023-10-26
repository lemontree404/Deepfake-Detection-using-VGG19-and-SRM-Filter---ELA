import os
import cv2
import sys
import numpy as np

src_file = sys.argv[1]
dest_file = sys.argv[2]
num_vids = int(sys.argv[3])
num_frames = int(sys.argv[4])

def frame_extractor(filename,dest_file,num_frames,num):

    vid = cv2.VideoCapture(filename)
    if vid.isOpened():
        pass
    else:
        raise("File not found.")

    frame_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    req_frames = list(range(0,frame_len,frame_len//num_frames + 1))
    
    for i in range(len(req_frames)):
        vid.set(cv2.CAP_PROP_POS_FRAMES,req_frames[i])
        ret,frame = vid.read()
        cv2.imwrite(f"{dest_file}/{num+i}.png",frame)

if dest_file not in os.listdir():
    os.mkdir(dest_file)

req_files = np.random.choice(os.listdir(src_file),num_vids,replace=False)

num = 0

for file in req_files:
    frame_extractor(f"{src_file}/{file}",dest_file,num_frames,num)
    num += num_frames