from models.models import early_fusion_model, late_fusion_model, tt_fusion_model
from utils.utils import calculate_class_weights, get_conf_matrx
import numpy as np
import os
from pandas import read_csv
import pickle

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


def process_labels(raw_y_data):
    """Takes raw labels and seperates them into the correct 'heads'
    
    :param raw_y_data: Raw inputd ata
    :return: A tuple of arrays, where each array contains all lables for a given output
    """
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
    """Makes and returns a generator which loads data from the supplied dir as and when needed
    
    :param data_dir: Folder containing the folders with the targets (data)
    :param labels: Width of the image
    :param batch_size: Number of images per batch
    :param model_flag: Which outputs you want
    :param include_tt_bias: Number of images per batch
    :param shuffle_data: Bool is you want the data shuffled or not
    :return: A generator to be used by a Keras model to read data
    """
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
        elif model_flag == "emo":
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
    :return: A generator to be used by a Keras model to read data + the number of vidoes 
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
        model = early_fusion_model(ts_size, model_flag)

    history = model.fit_generator(train_gen, steps_per_epoch=int(n_train / batch_size),
                                  epochs=epochs, class_weight=class_weights)

    with open('%s' % history_out, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save_weights(model_out)
    get_conf_matrx(model, test_gen, batch_size, n_test, model_flag, test_out)


def train_all(ts_size, batch_size, epochs, data_dir, train_labels, test_labels, model):
    v_weights, a_weights, g_weights = calculate_class_weights(train_labels)

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [v_weights, a_weights, g_weights],
                   "both", model,
                   "trained_models/both_%s_fusion.h5" % model, 
                   "training_history/both_%s_fusion.p" % model,
                   "results/both_%s_fusion.txt" % model)

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [g_weights],
                   "game", model,
                   "trained_models/game_%s_fusion.h5" % model, 
                   "training_history/game_%s_fusion.p" % model,
                   "results/game_%s_fusion.txt" % model)

    run_experiment(ts_size, batch_size, epochs, data_dir, train_labels, test_labels,
                   [v_weights, a_weights],
                   "emo", model,
                   "trained_models/emo_%s_fusion.h5" % model, 
                   "training_history/emo_%s_fusion.p" % model,
                   "results/emo_%s_fusion.txt" % model)


def main():
    if not os.path.exists("trained_models/"):
        os.makedirs("trained_models/")
    if not os.path.exists("training_history/"):
        os.makedirs("training_history/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    ts_size = 20
    batch_size = 5
    epochs = 1
    data_dir = "processed"
    train_labels = "train.csv"
    test_labels = "test.csv"

    train_all(ts_size, batch_size, epochs, data_dir, train_labels, test_labels, "early")
    train_all(ts_size, batch_size, epochs, data_dir, train_labels, test_labels, "late")
    train_all(ts_size, batch_size, epochs, data_dir, train_labels, test_labels, "tt")


if __name__ == "__main__":
    main()

