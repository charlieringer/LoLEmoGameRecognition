from utils.base_models import td_facenet, td_gamenet, td_audionet
from keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Concatenate, Lambda, Dot, Reshape
from keras.models import Model
from keras.optimizers import adam
from keras.utils import plot_model
from keras.regularizers import l2
import os
from extern.TTLayer import TT_Layer

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


def early_fusion_model(ts_size, model_flag):
	"""Builds and returns a full early fusion model

    :param ts_size: The length of the video time series sequence  
    :param model_flag: Which outputs the model should use ("both", "game", "emo")
    :return: The model (as a Keras model)
    """
	face_input = Input((ts_size, 64, 64, 3))
	game_input = Input((ts_size, 128, 128, 3))
	audio_input = Input((ts_size, 5512, 1))

	face_features = td_facenet((64, 64, 3), "face_")(face_input)
	game_features = td_gamenet((128, 128, 3), "game_")(game_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)

	hidden_feats = Concatenate()([face_features, game_features, audio_features])
	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	hidden_feats = LSTM(384, return_sequences=True)(hidden_feats)
	hidden_feats = LSTM(384, return_sequences=False)(hidden_feats)
	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	outputs = []

	if model_flag == "both" or model_flag == "emo":
		val_feats = Dense(128, activation="relu")(hidden_feats)
		valence = Dense(3, activation='softmax')(val_feats)
		outputs.append(valence)

		aro_feats = Dense(128, activation="relu")(hidden_feats)
		arousal = Dense(2, activation='softmax')(aro_feats)
		outputs.append(arousal)

	if model_flag == "both" or model_flag == "game":
		game_feats = Dense(128, activation="relu")(hidden_feats)
		game_events = Dense(8, activation='softmax')(game_feats)
		outputs.append(game_events)

	model = Model([face_input, game_input, audio_input], outputs)

	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/%s_early.png' % model_flag, show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'])
	print(model.summary())
	return model


def late_fusion_model(ts_size, model_flag):
	"""Builds and returns a full late fusion model

    :param ts_size: The length of the video time series sequence  
    :param model_flag: Which outputs the model should use ("both", "game", "emo")
    :return: The model (as a Keras model)
    """
	face_input = Input((ts_size, 64, 64, 3))
	game_input = Input((ts_size, 128, 128, 3))
	audio_input = Input((ts_size, 5512, 1))

	face_features = td_facenet((64, 64, 3), "face_")(face_input)
	game_features = td_gamenet((128, 128, 3), "game_")(game_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)

	face_features = BatchNormalization()(face_features)
	face_features = Dropout(0.2)(face_features)
	face_features = LSTM(128, return_sequences=True)(face_features)
	face_features = LSTM(128, return_sequences=False)(face_features)

	game_features = BatchNormalization()(game_features)
	game_features = Dropout(0.2)(game_features)
	game_features = LSTM(128, return_sequences=True)(game_features)
	game_features = LSTM(128, return_sequences=False)(game_features)

	audio_features = BatchNormalization()(audio_features)
	audio_features = Dropout(0.2)(audio_features)
	audio_features = LSTM(128, return_sequences=True)(audio_features)
	audio_features = LSTM(128, return_sequences=False)(audio_features)

	hidden_feats = Concatenate()([face_features, game_features, audio_features])
	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	outputs = []

	if model_flag == "both" or model_flag == "emo":
		val_feats = Dense(128, activation="relu")(hidden_feats)
		valence = Dense(3, activation='softmax')(val_feats)
		outputs.append(valence)

		aro_feats = Dense(128, activation="relu")(hidden_feats)
		arousal = Dense(2, activation='softmax')(aro_feats)
		outputs.append(arousal)

	if model_flag == "both" or model_flag == "game":
		game_feats = Dense(128, activation="relu")(hidden_feats)
		game_events = Dense(8, activation='softmax')(game_feats)
		outputs.append(game_events)

	model = Model([face_input, game_input, audio_input], outputs)

	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/%s_late.png' % model_flag, show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'])
	print(model.summary())
	return model


def tt_fusion_model(ts_size, model_flag):
	"""Builds and returns a full tt fusion model

    :param ts_size: The length of the video time series sequence  
    :param model_flag: Which outputs the model should use ("both", "game", "emo")
    :return: The model (as a Keras model)
    """
	face_input = Input((ts_size, 64, 64, 3))
	game_input = Input((ts_size, 128, 128, 3))
	audio_input = Input((ts_size, 5512, 1))
	bias = Input((1,))

	face_features = td_facenet((64, 64, 3), "face_")(face_input)
	game_features = td_gamenet((128, 128, 3), "game_")(game_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)

	face_features = BatchNormalization()(face_features)
	face_features = Dropout(0.2)(face_features)
	face_features = LSTM(128, return_sequences=True)(face_features)
	face_features = LSTM(128, return_sequences=False)(face_features)

	game_features = BatchNormalization()(game_features)
	game_features = Dropout(0.2)(game_features)
	game_features = LSTM(128, return_sequences=True)(game_features)
	game_features = LSTM(128, return_sequences=False)(game_features)

	audio_features = BatchNormalization()(audio_features)
	audio_features = Dropout(0.2)(audio_features)
	audio_features = LSTM(128, return_sequences=True)(audio_features)
	audio_features = LSTM(128, return_sequences=False)(audio_features)

	reshape_1 = Reshape((1, 129))(Concatenate()([bias, face_features]))
	reshape_2 = Reshape((1, 129))(Concatenate()([bias, game_features]))
	reshape_3 = Reshape((1, 129))(Concatenate()([bias, audio_features]))

	x = Dot(axes=1)([reshape_1, reshape_2])
	x = Reshape((1, 129 * 129))(x)
	x = Dot(axes=1)([x, reshape_3])
	hidden_feats = Reshape((129, 129, 129))(x)

	print("Tensor Shape: ", hidden_feats.shape)
	hidden_feats = TT_Layer(tt_input_shape=[3, 43, 129, 43, 3],
	                        tt_output_shape=[2, 4, 4, 4, 3],
	                        tt_ranks=[1, 2, 4, 4, 2, 1],
	                        activation='relu', kernel_regularizer=l2(5e-4), debug=False)(hidden_feats)
	print("After TT Shape: ", hidden_feats.shape)

	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	outputs = []

	if model_flag == "both" or model_flag == "emo":
		val_feats = Dense(128, activation="relu")(hidden_feats)
		valence = Dense(3, activation='softmax')(val_feats)
		outputs.append(valence)

		aro_feats = Dense(128, activation="relu")(hidden_feats)
		arousal = Dense(2, activation='softmax')(aro_feats)
		outputs.append(arousal)

	if model_flag == "both" or model_flag == "game":
		game_feats = Dense(128, activation="relu")(hidden_feats)
		game_events = Dense(8, activation='softmax')(game_feats)
		outputs.append(game_events)

	model = Model([face_input, game_input, audio_input, bias], outputs)

	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/%s_tt.png' % model_flag, show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'])
	print(model.summary())
	return model
