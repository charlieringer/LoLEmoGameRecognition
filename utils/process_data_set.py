import cv2
import numpy as np
import os
import subprocess
from librosa import load
from scipy.stats import zscore
import math
from random import shuffle
from pandas import read_csv


def generate_dummy_data(batch_size, ts_size):
	face_data = np.random.rand(batch_size, ts_size, 128, 128, 3).astype("uint8")
	game_data = np.random.rand(batch_size, ts_size, 128, 128, 3).astype("uint8")
	audio_data = np.random.rand(batch_size, ts_size, 40, 1).astype("float64")
	y_data = np.random.rand(batch_size, ts_size, 224, 224, 3).astype("uint8")
	return face_data, game_data, audio_data, y_data


def get_face_location(streamer_id):
	if streamer_id == "01":
		return 424, 235, 543, 360
	if streamer_id == "02":
		return 416, 264, 540, 360
	if streamer_id == "03":
		return 403, 254, 546, 360
	if streamer_id == "04":
		return 405, 280, 550, 360
	if streamer_id == "05":
		return 437, 284, 551, 360
	if streamer_id == "06":
		return 424, 279, 549, 360
	if streamer_id == "07":
		return 0, 225, 166, 360
	if streamer_id == "08":
		return 0, 236, 137, 360
	if streamer_id == "09":
		return 0, 235, 195, 360
	if streamer_id == "010":
		return 415, 284, 543, 360


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
		cv2.imwrite("%s/face_%i.png" % (outdir, curr_frame), face)
		game = cv2.resize(frame, (128, 128))
		cv2.imwrite("%s/game_%i.png" % (outdir, curr_frame), game)
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


def split_data():
	videos = [f.name for f in os.scandir("out/") if f.is_dir()]
	shuffle(videos)
	if not os.path.exists("out/train"):
		os.makedirs("out/train")
	if not os.path.exists("out/test"):
		os.makedirs("out/test")

	split_indx = int(len(videos)*0.8)
	for video in videos[split_indx:]:
		os.rename("out/%s" % video, "out/test/%s" % video)

	for video in videos[:split_indx]:
		os.rename("out/%s" % video, "out/train/%s" % video)


def split_csv(in_file, out_train, out_test, test_split=0.8):
	data = read_csv(in_file)
	data = data[data[' Misc'] != 1]
	n_data = len(data.index)
	split = int(n_data*test_split)
	data = data.sample(frac=1)

	train_split = data[:split]
	test_split = data[split:]

	train_split.to_csv(out_train)
	test_split.to_csv(out_test)


def main():
	split_csv("../master.csv", "train.csv", "test.csv")
	return
	data_dir = "data/"
	output_dir = "out_test/"
	videos = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
	print(len(videos))
	
	for video in videos:
		streamer_id = get_streamer_id(video)
		process_frames(data_dir + video, output_dir + video, get_face_location(streamer_id))
		process_audio(data_dir + video, output_dir + video)


if __name__ == "__main__":
	main()
