from utils.models import feature_fusion_model, late_fusion_model, tt_fusion_model
from utils.data_set_descriptors import calculate_class_weights
import numpy as np
import os
from pandas import read_csv
from utils.conf_mats import get_conf_matrx
import pickle
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


def process_labels(raw_y_data):
    valence = np.array([raw_y_data[' V_Neg'], raw_y_data[' V_Neut'], raw_y_data[' V_Pos']])
    arousal = np.array([raw_y_data[' A_Neut'], raw_y_data[' A_Pos']])

    laning = raw_y_data[' Laning']
    shopping = raw_y_data[' Shopping']
    returning = raw_y_data[' Returning']
    roaming = raw_y_data[' Roaming']
    fighting = raw_y_data[' Fighting']
    pushing = raw_y_data[' Pushing']
    defending = raw_y_data[' Defending']
    dead = raw_y_data[' Dead']

    game_events = np.array([laning, shopping, returning, roaming, fighting, pushing, defending, dead])
    return valence, arousal, game_events


def data_generator(data_dir, labels, batch_size, model_flag, include_tt_bias=False, shuffle_data=True):
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
            game_data.append(np.load("%s/%s/game.npy" % (data_dir, video_file)))
            audio_data.append(np.load("%s/%s/audio.npy" % (data_dir, video_file)))
            index += 1
            curr_batch_i += 1

        if include_tt_bias:
            inputs = [np.array(face_data), np.array(game_data), np.array(audio_data), np.ones((batch_size, 1))]
        else:
            inputs = [np.array(face_data), np.array(game_data), np.array(audio_data)]

        if model_flag == "both":
            outputs = [np.array(valence_labels), np.array(arousal_labels), np.array(game_event_labels)]
        elif model_flag == "face":
            outputs = [np.array(valence_labels), np.array(arousal_labels)]
        else:
            outputs = [np.array(game_event_labels)]

        yield inputs, outputs


def get_generator(data_dir, labels, batch_size, model_flag, include_tt_bias=False):
    """Makes and returns a generator which loads data from the supplied dir as and when needed
    
    :param data_dir: Folder containing the folders with the targets (data)
    :param labels: Width of the image
    :param batch_size: Number of images per batch
    :param include_tt_bias: Number of images per batch
    :return: A generator to be used by a Keras model to read data
    """
    number_of_videos = len(read_csv(labels).index)
    return data_generator(data_dir, labels, batch_size, model_flag, include_tt_bias), number_of_videos


def run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   class_weights, model_flag, fusion_type, model_out, history_out, test_out):
    train_gen, n_train = get_generator(data_dir, train_labels, batch_size, model_flag, fusion_type == "tt")
    test_gen, n_test = get_generator(data_dir, test_labels, batch_size, model_flag, fusion_type == "tt")

    if fusion_type == "tt":
        model = tt_fusion_model(ts_size, model_flag)
    elif fusion_type == "late":
        model = late_fusion_model(ts_size, model_flag)
    else:
        model = feature_fusion_model(ts_size, model_flag)

    history = model.fit_generator(train_gen, steps_per_epoch=int(n_train / batch_size),
                                  epochs=epochs, class_weight=class_weights)

    with open('%s' % history_out, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save_weights(model_out)
    get_conf_matrx(model, test_gen, batch_size, n_test, model_flag, test_out)


def train_all_feature(ts_size, batch_size, epochs, data_dir, train_labels, test_labels):
    v_weights, a_weights, g_weights = calculate_class_weights(train_labels)

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [v_weights, a_weights, g_weights],
                   "both", "feature",
                   "trained_models/both_feature_fusion.h5", "training_history/both_feature_fusion.p",
                   "results/both_feature_fusion.txt")

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [g_weights],
                   "game", "feature",
                   "trained_models/game_feature_fusion.h5", "training_history/game_feature_fusion.p",
                   "results/game_feature_fusion.txt")

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [v_weights, a_weights],
                   "face", "feature",
                   "trained_models/face_feature_fusion.h5", "training_history/face_feature_fusion.p",
                   "results/face_feature_fusion.txt")


def train_all_late(ts_size, batch_size, epochs, data_dir, train_labels, test_labels):
    v_weights, a_weights, g_weights = calculate_class_weights(train_labels)

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [v_weights, a_weights, g_weights],
                   "both", "late",
                   "trained_models/both_late_fusion.h5", "training_history/both_late_fusion.p",
                   "results/both_late_fusion.txt")

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [g_weights],
                   "game", "late",
                   "trained_models/game_late_fusion.h5", "training_history/game_late_fusion.p",
                   "results/game_late_fusion.txt")

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [v_weights, a_weights],
                   "face", "late",
                   "trained_models/face_late_fusion.h5", "training_history/face_late_fusion.p",
                   "results/face_late_fusion.txt")


def train_all_tt(ts_size, batch_size, epochs, data_dir, train_labels, test_labels):
    v_weights, a_weights, g_weights = calculate_class_weights(train_labels)

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [v_weights, a_weights, g_weights],
                   "both", "tt",
                   "trained_models/both_tt_fusion.h5", "training_history/both_tt_fusion.p",
                   "results/both_tt_fusion.txt")

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [g_weights],
                   "game", "tt",
                   "trained_models/game_tt_fusion.h5", "training_history/game_tt_fusion.p",
                   "results/game_tt_fusion.txt")

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [v_weights, a_weights],
                   "face", "tt",
                   "trained_models/face_tt_fusion.h5", "training_history/face_tt_fusion.p",
                   "results/face_tt_fusion.txt")


def main():
    ts_size = 20
    batch_size = 5
    epochs = 100
    data_dir = "out"
    train_labels = "train_augmented.csv"
    test_labels = "test.csv"

    train_all_feature(ts_size, batch_size, epochs, data_dir, train_labels, test_labels)
    train_all_late(ts_size, batch_size, epochs, data_dir, train_labels, test_labels)
    train_all_tt(ts_size, batch_size, epochs, data_dir, train_labels, test_labels)


if __name__ == "__main__":
    main()

