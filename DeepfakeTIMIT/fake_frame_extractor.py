import cv2
import os

def frame_extractor(video,file,num):

    vid = cv2.VideoCapture(f".//Data//higher_quality//{file}//{video}")
    if vid.isOpened():
        pass
    else:
        print("File not found.")
        quit()

    frame_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    req_frames = list(range(0,frame_len,frame_len//10 + 1))

    img_files = os.listdir(".//fake_images")

    for i in range(len(req_frames)):
        vid.set(cv2.CAP_PROP_POS_FRAMES,req_frames[i])
        ret,frame = vid.read()
        cv2.imwrite(f".//fake_images//{num+i}.png",frame)