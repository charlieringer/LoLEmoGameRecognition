from utils.models import get_latent_fusion_model, get_late_fusion_model
from utils.data_set_descriptors import calculate_class_weights
import numpy as np
import os
from pandas import read_csv
from utils.conf_mats import get_conf_matrx
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


def process_labels(raw_y_data):
    valence = np.array([raw_y_data[' V_Neg'], raw_y_data[' V_Neut'], raw_y_data[' V_Pos']])
    arousal = np.array([raw_y_data[' A_Neg'], raw_y_data[' A_Neut'], raw_y_data[' A_Pos']])

    laning = raw_y_data[' Laning']
    shopping = raw_y_data[' Shopping']
    returning = raw_y_data[' Returning']
    roaming = raw_y_data[' Roaming']
    fighting = raw_y_data[' Fighting']
    pushing = raw_y_data[' Pushing']
    defending = raw_y_data[' Defending']
    dead = raw_y_data[' Dead']
    misc = raw_y_data[' Misc']

    game_events = np.array([laning, shopping, returning, roaming, fighting, pushing, defending, dead, misc])
    return valence, arousal, game_events


def data_generator(data_dir, labels, batch_size, model_flag, shuffle_data=True):
    y_data = read_csv(labels)
    if shuffle_data:
        y_data = y_data.sample(frac=1)
    index = 0
    while True:
        face_data = []
        game_data = []
        audio_data = []
        valence_labels = []
        arousal_labels = []
        game_event_labels = []
        curr_batch_i = 0
        while curr_batch_i < batch_size:
            if index >= len(y_data.index):
                index = 0
                if shuffle_data:
                    y_data = y_data.sample(frac=1)
            raw_y_data = y_data.iloc[index]
            proc_y_data = process_labels(raw_y_data)

            valence_labels.append(proc_y_data[0])
            arousal_labels.append(proc_y_data[1])
            game_event_labels.append(proc_y_data[2])
            video_file = y_data.iloc[index]['File']
            face_data.append(np.load("%s/%s/face.npy" % (data_dir, video_file)))
            game_data.append(np.load("%s/%s/game.npy" % (data_dir, video_file )))
            audio_data.append(np.load("%s/%s/audio.npy" % (data_dir, video_file)))
            index += 1
            curr_batch_i += 1
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
    """
    number_of_videos = len(read_csv(labels).index)
    return data_generator(data_dir, labels, batch_size, model_flag), number_of_videos


def train_late_fusion_model(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                            class_weights, model_flag, model_out):
    train_gen, n_train = get_generator(data_dir, train_labels, batch_size, model_flag=model_flag)
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag=model_flag)

    model = get_late_fusion_model(ts_size, model_flag)
    model.fit_generator(train_gen, steps_per_epoch=int(n_train / batch_size), epochs=epochs, class_weight=class_weights)
    model.save_weights(model_out)
    get_conf_matrx(model, test_gen, batch_size, n_test, model_flag)


def train_latent_fusion_model(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                              class_weights, model_flag, model_out):
    train_gen, n_train = get_generator(data_dir, train_labels, batch_size, model_flag=model_flag)
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag=model_flag)

    model = get_latent_fusion_model(ts_size, model_flag)
    model.fit_generator(train_gen, steps_per_epoch=int(n_train / batch_size), epochs=epochs, class_weight=class_weights)
    model.save_weights(model_out)
    get_conf_matrx(model, test_gen, batch_size, n_test, model_flag)


def train_all_latent(ts_size, batch_size, epochs, data_dir, train_labels, test_labels):
    v_weights, a_weights, g_weights = calculate_class_weights(train_labels)

    train_latent_fusion_model(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                              [v_weights, a_weights, g_weights], "both", "trained_models/both_latent_fusion.h5")
    train_latent_fusion_model(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                              [g_weights], "game", "trained_models/game_latent_fusion.h5")
    train_latent_fusion_model(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                              [v_weights, a_weights], "face", "trained_models/face_latent_fusion.h5")


def train_all_late(ts_size, batch_size, epochs, data_dir, train_labels, test_labels):
    v_weights, a_weights, g_weights = calculate_class_weights(train_labels)

    train_late_fusion_model(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                              [v_weights, a_weights, g_weights], "both", "trained_models/both_late_fusion.h5")
    train_late_fusion_model(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                              [g_weights], "game", "trained_models/game_late_fusion.h5")
    train_late_fusion_model(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                              [v_weights, a_weights], "face", "trained_models/face_late_fusion.h5")


def get_conf_mat_all_late(ts_size, data_dir, test_labels, batch_size):
    model = get_late_fusion_model(ts_size, "both")
    model.load_weights("trained_models/both_late_fusion.h5")
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag="both")
    get_conf_matrx(model, test_gen, batch_size, n_test, "both")

    model = get_late_fusion_model(ts_size, "game")
    model.load_weights("trained_models/game_late_fusion.h5")
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag="game")
    get_conf_matrx(model, test_gen, batch_size, n_test, "game")

    model = get_late_fusion_model(ts_size, "face")
    model.load_weights("trained_models/face_late_fusion.h5")
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag="face")
    get_conf_matrx(model, test_gen, batch_size, n_test, "face")


def get_conf_mat_all_latent(ts_size, data_dir, test_labels, batch_size):
    model = get_latent_fusion_model(ts_size, "both")
    model.load_weights("trained_models/both_latent_fusion.h5")
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag="both")
    get_conf_matrx(model, test_gen, batch_size, n_test, "both")

    model = get_latent_fusion_model(ts_size, "game")
    model.load_weights("trained_models/game_latent_fusion.h5")
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag="game")
    get_conf_matrx(model, test_gen, batch_size, n_test, "game")

    model = get_latent_fusion_model(ts_size, "face")
    model.load_weights("trained_models/face_latent_fusion.h5")
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag="face")
    get_conf_matrx(model, test_gen, batch_size, n_test, "face")


def main():
    ts_size = 20
    batch_size = 5
    epochs = 50
    data_dir = "out"
    train_labels = "train.csv"
    test_labels = "test.csv"
    train_late = False

    # train_all_late(ts_size, batch_size, epochs, data_dir, train_labels, test_labels)
    # train_all_latent(ts_size, batch_size, epochs, data_dir, train_labels, test_labels)

    get_conf_mat_all_late(ts_size, data_dir, test_labels, batch_size)
    get_conf_mat_all_latent(ts_size, data_dir, test_labels, batch_size)


if __name__ == "__main__":
    main()

