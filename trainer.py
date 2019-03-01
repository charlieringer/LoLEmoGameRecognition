from models import get_latent_fusion_model, get_late_fusion_model
from data_set_descriptors import calculate_class_weights
import numpy as np
import os
from random import shuffle
from pandas import read_csv
from conf_mats import get_conf_matrx
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


def process_labels(raw_y_data):
    valence = np.array([raw_y_data[' V_Neg'].values[0], raw_y_data[' V_Neut'].values[0], raw_y_data[' V_Pos'].values[0]])
    arousal = np.array([raw_y_data[' A_Neg'].values[0], raw_y_data[' A_Neut'].values[0], raw_y_data[' A_Pos'].values[0]])

    laning = raw_y_data[' Laning'].values[0]
    shopping = raw_y_data[' Shopping'].values[0]
    returning = raw_y_data[' Returning'].values[0]
    roaming = raw_y_data[' Roaming'].values[0]
    fighting = raw_y_data[' Fighting'].values[0]
    pushing = raw_y_data[' Pushing'].values[0]
    defending = raw_y_data[' Defending'].values[0]
    dead = raw_y_data[' Dead'].values[0]
    misc = raw_y_data[' Misc'].values[0]

    game_events = np.array([laning, shopping, returning, roaming, fighting, pushing, defending, dead, misc])
    return valence, arousal, game_events


def data_generator(data_dir, labels, batch_size, model_flag, shuffle_data=True):
    videos = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    if shuffle_data:
        shuffle(videos)
    video_inx = 0
    annotations = read_csv(labels)
    while True:
        face_data = []
        game_data = []
        audio_data = []
        valence_labels = []
        arousal_labels = []
        game_event_labels = []

        i = 0
        while i < batch_size:
            if video_inx >= len(videos):
                video_inx = 0
                if shuffle_data:
                    shuffle(videos)

            raw_y_data = annotations.loc[annotations['File'] == videos[video_inx]]
            labels = process_labels(raw_y_data)
            if labels[2][8] == 1:
                video_inx += 1
                continue

            valence_labels.append(labels[0])
            arousal_labels.append(labels[1])
            game_event_labels.append(labels[2])

            face_data.append(np.load("%s/%s/face.npy" % (data_dir, videos[video_inx])))
            game_data.append(np.load("%s/%s/game.npy" % (data_dir, videos[video_inx])))
            audio_data.append(np.load("%s/%s/audio.npy" % (data_dir, videos[video_inx])))

            if audio_data[i].shape != (20, 5512, 1):
                print("Audio error with: ", videos[video_inx])
            video_inx += 1
            i += 1
        if model_flag == "both":
                yield [np.array(face_data), np.array(game_data), np.array(audio_data)], \
                    [np.array(valence_labels), np.array(arousal_labels), np.array(game_event_labels)]
        elif model_flag == "face":
                yield [np.array(face_data), np.array(audio_data)], [np.array(valence_labels), np.array(arousal_labels)]
        elif model_flag == "game":
                yield [np.array(game_data), np.array(audio_data)], [np.array(game_event_labels)]


def get_generator(data_dir, labels, batch_size, model_flag):	
    """Makes and returns a generator which loads data from the supplied dir as and when needed
    
    :param data_dir: Folder containing the folders with the targets (data)
    :param labels: Width of the image
    :param batch_size: Number of images per batch
    :param model_flag: Number of images per batch
    :return: A generator to be used by a Keras model to read data
    
    Note: This is the preferred way to process the data when there is a large dataset
    Note: This requires the data to be inside folders inside the supplied dir
    """
    number_of_videos = len([f.path for f in os.scandir(data_dir) if f.is_dir()])
    return data_generator(data_dir, labels, batch_size, model_flag), number_of_videos


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


def train_late_fusion_model(ts_size, batch_size, epochs, train_dir, test_dir,
                            labels, class_weights, model_flag, model_out):
    train_gen, n_train = get_generator(train_dir, labels, batch_size, model_flag=model_flag)
    test_gen, n_test = get_generator(test_dir, labels, batch_size, model_flag=model_flag)

    model = get_late_fusion_model(ts_size, model_flag)
    model.fit_generator(train_gen, steps_per_epoch=int(n_train / batch_size), epochs=epochs, class_weight=class_weights)
    model.save_weights(model_out)
    get_conf_matrx(model, test_gen, batch_size, n_test, model_flag)


def train_latent_fusion_model(ts_size, batch_size, epochs, train_dir, test_dir,
                              labels, class_weights, model_flag, model_out):
    train_gen, n_train = get_generator(train_dir, labels, batch_size, model_flag=model_flag)
    test_gen, n_test = get_generator(test_dir, labels, batch_size, model_flag=model_flag)

    model = get_latent_fusion_model(ts_size, model_flag)
    model.fit_generator(train_gen, steps_per_epoch=int(n_train / batch_size), epochs=epochs, class_weight=class_weights)
    model.save_weights(model_out)
    get_conf_matrx(model, test_gen, batch_size, n_test, model_flag)


def train_all_latent(ts_size, batch_size, epochs, train_dir, test_dir, labels):
    v_weights, a_weights, g_weights = calculate_class_weights(labels)

    train_latent_fusion_model(ts_size, batch_size, epochs, train_dir, test_dir, labels,
                              [v_weights, a_weights, g_weights], "both", "both_latent_fusion.h5")
    train_latent_fusion_model(ts_size, batch_size, epochs, train_dir, test_dir, labels,
                              [g_weights], "game", "game_latent_fusion.h5")
    train_latent_fusion_model(ts_size, batch_size, epochs, train_dir, test_dir, labels,
                              [v_weights, a_weights], "face", "face_latent_fusion.h5")


def train_all_late(ts_size, batch_size, epochs, train_dir, test_dir, labels):
    v_weights, a_weights, g_weights = calculate_class_weights(labels)

    train_late_fusion_model(ts_size, batch_size, epochs, train_dir, test_dir, labels,
                              [v_weights, a_weights, g_weights], "both", "both_late_fusion.h5")
    train_late_fusion_model(ts_size, batch_size, epochs, train_dir, test_dir, labels,
                              [g_weights], "game", "game_late_fusion.h5")
    train_late_fusion_model(ts_size, batch_size, epochs, train_dir, test_dir, labels,
                              [v_weights, a_weights], "face", "face_late_fusion.h5")


def main():
    ts_size = 20
    batch_size = 5
    epochs = 100
    train_dir = "out/train"
    test_dir = "out/test"
    labels = "master.csv"
    train_late = False

    if train_late:
        train_all_late(ts_size, batch_size, epochs, train_dir, test_dir, labels)
    else:
        train_all_latent(ts_size, batch_size, epochs, train_dir, test_dir, labels)


if __name__ == "__main__":
    main()

