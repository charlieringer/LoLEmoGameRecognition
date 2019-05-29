from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, LSTM, \
    BatchNormalization, Flatten, TimeDistributed, GlobalAveragePooling2D, Activation, Add, ZeroPadding2D, Concatenate
from keras.models import Model
from keras.optimizers import adam
from keras.utils import plot_model
import keras_metrics
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


def audionet(input_shape, n_features):
    input_tensor = Input(input_shape)
    x = Conv1D(128, 5, padding='same', activation='relu')(input_tensor)
    x = MaxPooling1D(pool_size=8)(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=8)(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=8)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(n_features, activation='relu')(x)
    model = Model(input_tensor, x)
    return model


def td_audionet(input_shape, n_features):
    return TimeDistributed(audionet(input_shape, n_features))


def _add_norm_relu(x):
    """Adds a Batch Normalization and ReLU layer to the input X

    :param x: Input to these layers
    :return: x with Batch Norm and ReLU applied
    """
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def _add_residual_block(x, filters_in, filters_out, block_id):
    """Generates a bottleneck residual block for x with the supplied number of filters

    :param x: Input to these layers
    :param filters_in: Number of bottleneck filters to use in res block
    :param filters_out: number of filters for the output (should be the same as the number of filters x has)
    :param block_id: The id of this block (for weight loading and saving)
    :return: x with res block added
    """
    input_tensor = x
    x = Conv2D(filters_in, (1, 1), name=block_id + '_1')(x)
    x = _add_norm_relu(x)
    x = Conv2D(filters_in, (3, 3), padding='same', name=block_id + '_2')(x)
    x = _add_norm_relu(x)
    x = Conv2D(filters_out, (1, 1), name=block_id + '_3')(x)
    x = BatchNormalization()(x)
    x = Add()([input_tensor, x])
    x = Activation('relu')(x)
    return x


def _add_conv_block(x, filters_in, filters_out, block_id):
    """Generates a conv block for x with the supplied number of filters

    :param x: Input to these layers
    :param filters_in: Number of bottleneck filters to use in res block
    :param filters_out: number of filters for the output (should be the same as the number of filters x has)
    :param block_id: The id of this block (for weight loading and saving)
    :return: x with res block added
    """
    input_tensor = x
    x = Conv2D(filters_in, (1, 1), strides=(2, 2), name=block_id + '_1')(x)
    x = _add_norm_relu(x)
    x = Conv2D(filters_in, (3, 3), padding='same', name=block_id + '_2')(x)
    x = _add_norm_relu(x)
    x = Conv2D(filters_out, (1, 1), name=block_id + '_3')(x)
    shortcut = Conv2D(filters_out, (1, 1), strides=(2, 2), name=block_id + '_shortcut')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def facenet(img_input, network_id=""):
    """Builds and returns a res face model

    :param img_input: Shape of the input image. Expected 256x256x3 as that provides a 1x1x1024 latent space encoding
    :param network_id: ID of network
    :return: The encoder (as a Keras model)
    """
    # img_input = Input(img_input_shape)
    x = ZeroPadding2D(padding=(3, 3), name='%spre_conv_pad' % network_id)(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='%sconv1' % network_id)(x)
    x = _add_norm_relu(x)
    x = ZeroPadding2D(padding=(1, 1), name='%spre_pool_pad' % network_id)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='%spool1' % network_id)(x)

    x = _add_conv_block(x, 64, 256, '%sres1_1' % network_id)
    x = _add_residual_block(x, 64, 256, '%sres1_2' % network_id)
    # x = _add_residual_block(x, 64, 256, '%sres1_3' % network_id)

    x = _add_conv_block(x, 256, 512, '%sres4_1' % network_id)
    x = _add_residual_block(x, 256, 512, '%sres4_2' % network_id)

    x = GlobalAveragePooling2D()(x)
    # instantiate encoder model
    model = Model(img_input, x, name='%s_resnet' % network_id)
    # plot_model(encoder, to_file='encoder.png', show_shapes=True)
    return model


def td_facenet(img_input_s, network_id=""):
    return TimeDistributed(facenet(Input(img_input_s), network_id))


def gamenet(img_input, network_id=""):
    """Builds and returns a resnet 50 model

    :param img_input: Shape of the input image. Expected 256x256x3 as that provides a 1x1x1024 latent space encoding
    :param network_id: ID of network
    :return: The encoder (as a Keras model)
    """
    x = ZeroPadding2D(padding=(3, 3), name='%spre_conv_pad' % network_id)(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='%sconv1' % network_id)(x)
    x = _add_norm_relu(x)
    x = ZeroPadding2D(padding=(1, 1), name='%spre_pool_pad' % network_id)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='%spool1' % network_id)(x)

    x = _add_conv_block(x, 64, 128, '%sres1_1' % network_id)
    x = _add_residual_block(x, 64, 128, '%sres1_2' % network_id)
    # x = _add_residual_block(x, 64, 128, '%sres1_3' % network_id)

    x = _add_conv_block(x, 128, 256, '%sres2_1' % network_id)
    x = _add_residual_block(x, 128, 256, '%sres2_2' % network_id)
    # x = _add_residual_block(x, 128, 256, '%sres2_3' % network_id)

    x = _add_conv_block(x, 256, 512, '%sres4_1' % network_id)
    x = _add_residual_block(x, 256, 512, '%sres4_2' % network_id)
    # x = _add_residual_block(x, 256, 512, '%sres4_3' % network_id)
    x = GlobalAveragePooling2D()(x)
    # instantiate encoder model
    model = Model(img_input, x, name='%s_resnet' % network_id)
    # plot_model(encoder, to_file='encoder.png', show_shapes=True)
    return model


def td_gamenet(img_input_s, network_id=""):
    return TimeDistributed(gamenet(Input(img_input_s), network_id))
