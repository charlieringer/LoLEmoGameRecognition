from utils.base_models import td_facenet, td_gamenet, td_audionet
from keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Concatenate, Lambda, Dot, Reshape
from keras.models import Model
from keras.optimizers import adam
from keras.utils import plot_model
from keras.regularizers import l2
import keras.backend as K
import os
from utils.extern.TTLayer import TT_Layer

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


def game_feature_fusion_model(ts_size):
	game_input = Input((ts_size, 128, 128, 3))
	audio_input = Input((ts_size, 5512, 1))
	game_features = td_gamenet((128, 128, 3), "game_")(game_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)
	hidden_feats = Concatenate()([game_features, audio_features])
	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)
	x = LSTM(256, return_sequences=True)(hidden_feats)
	x = LSTM(256, return_sequences=False)(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	game_feats = Dense(128, activation="relu")(x)
	game_events = Dense(8, activation='softmax')(game_feats)
	model = Model([game_input, audio_input], [game_events])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/game_feature.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'])
	# print(model.summary())
	return model


def face_feature_fusion_model(ts_size):
	face_input = Input((ts_size, 64, 64, 3))
	audio_input = Input((ts_size, 5512, 1))
	face_features = td_facenet((64, 64, 3), "face_")(face_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)
	hidden_feats = Concatenate()([face_features, audio_features])
	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)
	x = LSTM(256, return_sequences=True)(hidden_feats)
	x = LSTM(256, return_sequences=False)(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	val_feats = Dense(128, activation="relu")(x)
	aro_feats = Dense(128, activation="relu")(x)
	valence = Dense(3, activation='softmax')(val_feats)
	arousal = Dense(3, activation='softmax')(aro_feats)
	model = Model([face_input, audio_input], [valence, arousal])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/face_feature.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'], loss_weights=[0.5, 0.5])
	# print(model.summary())
	return model


def both_feature_fusion_model(ts_size):
	face_input = Input((ts_size, 64, 64, 3))
	game_input = Input((ts_size, 128, 128, 3))
	audio_input = Input((ts_size, 5512, 1))

	face_features = td_facenet((64, 64, 3), "face_")(face_input)
	game_features = td_gamenet((128, 128, 3), "game_")(game_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)
	hidden_feats = Concatenate()([face_features, game_features, audio_features])
	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)
	x = LSTM(384, return_sequences=True)(hidden_feats)
	x = LSTM(384, return_sequences=False)(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	val_feats = Dense(128, activation="relu")(x)
	aro_feats = Dense(128, activation="relu")(x)
	game_feats = Dense(128, activation="relu")(x)
	valence = Dense(3, activation='softmax')(val_feats)
	arousal = Dense(3, activation='softmax')(aro_feats)
	game_events = Dense(8, activation='softmax')(game_feats)
	model = Model([face_input, game_input, audio_input], [valence, arousal, game_events])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/both_feature.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'], loss_weights=[0.33, 0.33, 0.33])
	# print(model.summary())
	return model


def get_feature_fusion_model(ts_size, model_flag):
	if model_flag == "both":
		return both_feature_fusion_model(ts_size)
	if model_flag == "game":
		return game_feature_fusion_model(ts_size)
	if model_flag == "face":
		return face_feature_fusion_model(ts_size)
	print("Error, could not find a model that satisfies model flag %s" % model_flag)


def game_late_fusion_model(ts_size):
	game_input = Input((ts_size, 128, 128, 3))
	audio_input = Input((ts_size, 5512, 1))
	game_features = td_gamenet((128, 128, 3), "game_")(game_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)

	game_features = BatchNormalization()(game_features)
	game_features = Dropout(0.2)(game_features)
	game_features = LSTM(128, return_sequences=True)(game_features)
	game_features = LSTM(128, return_sequences=False)(game_features)

	audio_features = BatchNormalization()(audio_features)
	audio_features = Dropout(0.2)(audio_features)
	audio_features = LSTM(128, return_sequences=True)(audio_features)
	audio_features = LSTM(128, return_sequences=False)(audio_features)

	hidden_feats = Concatenate()([game_features, audio_features])
	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	hidden_feats = Dense(128, activation="relu")(hidden_feats)
	game_events = Dense(8, activation='softmax')(hidden_feats)
	model = Model([game_input, audio_input], [game_events])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/game_late.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'])
	# print(model.summary())
	return model


def face_late_fusion_model(ts_size):
	face_input = Input((ts_size, 64, 64, 3))
	audio_input = Input((ts_size, 5512, 1))

	face_features = td_facenet((64, 64, 3), "face_")(face_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)

	face_features = BatchNormalization()(face_features)
	face_features = Dropout(0.2)(face_features)
	face_features = LSTM(128, return_sequences=True)(face_features)
	face_features = LSTM(128, return_sequences=False)(face_features)

	audio_features = BatchNormalization()(audio_features)
	audio_features = Dropout(0.2)(audio_features)
	audio_features = LSTM(128, return_sequences=True)(audio_features)
	audio_features = LSTM(128, return_sequences=False)(audio_features)

	hidden_feats = Concatenate()([face_features, audio_features])
	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	val_feats = Dense(128, activation="relu")(hidden_feats)
	aro_feats = Dense(128, activation="relu")(hidden_feats)
	valence = Dense(3, activation='softmax')(val_feats)
	arousal = Dense(3, activation='softmax')(aro_feats)
	model = Model([face_input, audio_input], [valence, arousal])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/emo_late.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'], loss_weights=[0.5, 0.5])
	# print(model.summary())
	return model


def both_late_fusion_model(ts_size):
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

	val_feats = Dense(128, activation="relu")(hidden_feats)
	aro_feats = Dense(128, activation="relu")(hidden_feats)
	game_feats = Dense(128, activation="relu")(hidden_feats)
	valence = Dense(3, activation='softmax')(val_feats)
	arousal = Dense(3, activation='softmax')(aro_feats)
	game_events = Dense(8, activation='softmax')(game_feats)
	model = Model([face_input, game_input, audio_input], [valence, arousal, game_events])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/both_late.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'], loss_weights=[0.33, 0.33, 0.33])
	print(model.summary())
	return model


def get_late_fusion_model(ts_size, model_flag):
	if model_flag == "both":
		return both_late_fusion_model(ts_size)
	if model_flag == "game":
		return game_late_fusion_model(ts_size)
	if model_flag == "face":
		return face_late_fusion_model(ts_size)
	print("Error, could not find a model that satisfies model flag %s" % model_flag)


def game_tt_fusion_model(ts_size):
	game_input = Input((ts_size, 128, 128, 3))
	audio_input = Input((ts_size, 5512, 1))
	bias = Input((1,))

	game_features = td_gamenet((128, 128, 3), "game_")(game_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)

	game_features = BatchNormalization()(game_features)
	game_features = Dropout(0.2)(game_features)
	game_features = LSTM(128, return_sequences=True)(game_features)
	game_features = LSTM(128, return_sequences=False)(game_features)

	audio_features = BatchNormalization()(audio_features)
	audio_features = Dropout(0.2)(audio_features)
	audio_features = LSTM(128, return_sequences=True)(audio_features)
	audio_features = LSTM(128, return_sequences=False)(audio_features)

	reshape_1 = Reshape((1, 129))(Concatenate()([bias, game_features]))
	reshape_2 = Reshape((1, 129))(Concatenate()([bias, audio_features]))

	x = Dot(axes=1)([reshape_1, reshape_2])
	x = Reshape((1, 129 * 129))(x)
	hidden_feats = Reshape((129, 129))(x)
	print("Tensor Shape: ", hidden_feats.shape)
	hidden_feats = TT_Layer(tt_input_shape=[3, 1849, 3],
	                        tt_output_shape=[4, 8, 4],
	                        tt_ranks=[1, 4, 4, 1],
	                        activation='relu', kernel_regularizer=l2(5e-4), debug=False)(hidden_feats)
	print("After TT Shape: ", hidden_feats.shape)

	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	hidden_feats = Dense(128, activation="relu")(hidden_feats)
	game_events = Dense(8, activation='softmax')(hidden_feats)
	model = Model([game_input, audio_input, bias], [game_events])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/game_late.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'])
	# print(model.summary())
	return model


def face_tt_fusion_model(ts_size):
	face_input = Input((ts_size, 64, 64, 3))
	audio_input = Input((ts_size, 5512, 1))
	bias = Input((1,))

	face_features = td_facenet((64, 64, 3), "face_")(face_input)
	audio_features = td_audionet((5512, 1), 512)(audio_input)

	face_features = BatchNormalization()(face_features)
	face_features = Dropout(0.2)(face_features)
	face_features = LSTM(128, return_sequences=True)(face_features)
	face_features = LSTM(128, return_sequences=False)(face_features)

	audio_features = BatchNormalization()(audio_features)
	audio_features = Dropout(0.2)(audio_features)
	audio_features = LSTM(128, return_sequences=True)(audio_features)
	audio_features = LSTM(128, return_sequences=False)(audio_features)

	reshape_1 = Reshape((1, 129))(Concatenate()([bias, face_features]))
	reshape_2 = Reshape((1, 129))(Concatenate()([bias, audio_features]))

	x = Dot(axes=1)([reshape_1, reshape_2])
	x = Reshape((1, 129 * 129))(x)
	hidden_feats = Reshape((129, 129))(x)
	print("Tensor Shape: ", hidden_feats.shape)
	hidden_feats = TT_Layer(tt_input_shape=[3, 1849, 3],
	                        tt_output_shape=[4, 8, 4],
	                        tt_ranks=[1, 4, 4, 1],
	                        activation='relu', kernel_regularizer=l2(5e-4), debug=False)(hidden_feats)
	print("After TT Shape: ", hidden_feats.shape)

	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	val_feats = Dense(128, activation="relu")(hidden_feats)
	aro_feats = Dense(128, activation="relu")(hidden_feats)
	valence = Dense(3, activation='softmax')(val_feats)
	arousal = Dense(3, activation='softmax')(aro_feats)
	model = Model([face_input, audio_input, bias], [valence, arousal])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/emo_late.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'], loss_weights=[0.5, 0.5])
	# print(model.summary())
	return model


def both_tt_fusion_model(ts_size):
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
	                        tt_output_shape=[2, 3, 4, 4, 4],
	                        tt_ranks=[1, 2, 4, 4, 2, 1],
	                        activation='relu', kernel_regularizer=l2(5e-4), debug=False)(hidden_feats)
	print("After TT Shape: ", hidden_feats.shape)

	hidden_feats = BatchNormalization()(hidden_feats)
	hidden_feats = Dropout(0.2)(hidden_feats)

	val_feats = Dense(128, activation="relu")(hidden_feats)
	aro_feats = Dense(128, activation="relu")(hidden_feats)
	game_feats = Dense(128, activation="relu")(hidden_feats)
	valence = Dense(3, activation='softmax')(val_feats)
	arousal = Dense(3, activation='softmax')(aro_feats)
	game_events = Dense(8, activation='softmax')(game_feats)
	model = Model([face_input, game_input, audio_input, bias], [valence, arousal, game_events])
	opt = adam(lr=0.0005)
	plot_model(model, to_file='model_imgs/both_late.png', show_shapes=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['acc'], loss_weights=[0.33, 0.33, 0.33])
	print(model.summary())
	return model


def get_tt_fusion_model(ts_size, model_flag):
	if model_flag == "both":
		return both_tt_fusion_model(ts_size)
	if model_flag == "game":
		return game_tt_fusion_model(ts_size)
	if model_flag == "face":
		return face_tt_fusion_model(ts_size)
	print("Error, could not find a model that satisfies model flag %s" % model_flag)
