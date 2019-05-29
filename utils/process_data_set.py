import cv2
import numpy as np
import os
import subprocess
from librosa import load
from scipy.stats import zscore
import math
import json
from random import shuffle
from pandas import read_csv


def generate_dummy_data(batch_size, ts_size):
	face_data = np.random.rand(batch_size, ts_size, 128, 128, 3).astype("uint8")
	game_data = np.random.rand(batch_size, ts_size, 128, 128, 3).astype("uint8")
	audio_data = np.random.rand(batch_size, ts_size, 40, 1).astype("float64")
	y_data = np.random.rand(batch_size, ts_size, 224, 224, 3).astype("uint8")
	return face_data, game_data, audio_data, y_data


def get_face_location(streamer_id):
	bounding_boxes = json.loads("../webcam_boxes.json")
	return bounding_boxes[streamer_id]


def get_streamer_id(video_str):
	streamer_id = ""
	started = False
	for char in video_str:
		if char == '_': 
			if not started:
				started = True
			else:
				return streamer_id
		elif started:
			streamer_id += char


def process_frames(video, outdir, face_loc):
	cap = cv2.VideoCapture(video)
	face_imgs = []
	game_imgs = []

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	n_frames = 20
	curr_frame = 0

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	while True:
		if curr_frame >= n_frames:
			break
		curr_frame += 1
		_, frame = cap.read()
		face = np.array(frame[face_loc[1]:face_loc[3], face_loc[0]:face_loc[2], :])
		cv2.rectangle(frame, (face_loc[0], face_loc[1]), (face_loc[2], face_loc[3]), -1, -1)

		face = cv2.resize(face, (64, 64))
		game = cv2.resize(frame, (128, 128))
		face_imgs.append(face)
		game_imgs.append(game)

	processed_face = np.array(face_imgs)/255
	np.save("%s/face.npy" % outdir, processed_face)

	processed_game = np.array(game_imgs)/255
	np.save("%s/game.npy" % outdir, processed_game)


def process_audio(video, outdir):
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	os.system("ffmpeg -hide_banner -loglevel panic -y -i %s -vn -acodec copy %s" % (video, outdir + "/raw_audio.mp4"))
	wav, rate = load(outdir + "/raw_audio.mp4")
	rate_pf = int(rate/4)
	wav = zscore(wav)
	audio_slices = []
	for i in range(0, 20):
		start = i*rate_pf
		end = i*rate_pf+rate_pf
		if end < len(wav):
			audio_frame = wav[start:end]
			audio_slices.append(audio_frame)
		else:
			audio_frame = wav[start:len(wav)]
			missing = end - len(wav)
			audio_frame = np.concatenate((audio_frame, np.zeros((missing,)).astype("float32")), axis=0)
			audio_slices.append(audio_frame)
	processed_array = np.expand_dims(np.array(audio_slices), axis=2)
	np.save("%s/audio.npy" % outdir, processed_array)


def main():
	data_dir = "../data/"
	output_dir = "../processed/"
	videos = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
	
	for video in videos:
		streamer_id = get_streamer_id(video)
		process_frames(data_dir + video, output_dir + video, get_face_location(streamer_id))
		process_audio(data_dir + video, output_dir + video)


if __name__ == "__main__":
	main()
