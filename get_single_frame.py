import cv2
import numpy as np
import os

def get_single_frame(video, outdir):
	cap = cv2.VideoCapture(video)
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	_, frame = cap.read()
	cv2.imwrite("%s/%s.png" % (outdir, 0), frame)

def main():
	data_dir = "data/"
	output_dir = "one_frame/"
	videos = [f for f in os.listdir(data_dir) if f.endswith(".mp4") ]
	for video in videos:
		get_single_frame(data_dir + video, output_dir + video)

main()