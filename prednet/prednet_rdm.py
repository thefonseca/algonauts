from keras.models import Model
from keras.layers import Flatten, Dense, TimeDistributed, LSTM
from keras.layers import Input, Masking, Lambda, Dropout
from keras.layers import Bidirectional, concatenate, average
import prednet_model
import numpy as np


def crop(dimension, start=None, end=None, stride=1, name=None):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    # See https://github.com/keras-team/keras/issues/890
    def func(x):
        if dimension == 0:
            return x[start:end:stride]
        if dimension == 1:
            return x[:, start:end:stride]
        if dimension == 2:
            return x[:, :, start:end:stride]
        if dimension == 3:
            return x[:, :, :, start:end:stride]
        if dimension == 4:
            return x[:, :, :, :, start:end:stride]

    return Lambda(func, name=name)


def lstm_layer(tensor, mask_value, hidden_dims, dropout, name, bidirectional=False, return_sequences=False):
    x = TimeDistributed(Flatten(), name='flatten_' + name)(tensor)
    if mask_value is not None:
        x = Masking(mask_value=mask_value)(x)
    for dim in hidden_dims:
        if bidirectional:
            x = Bidirectional(LSTM(dim, return_sequences=return_sequences, dropout=dropout),
                              merge_mode='concat', name='BiLSTM_' + name)(x)
        else:
            x = LSTM(dim, return_sequences=return_sequences, dropout=dropout)(x)
    return x


def create_model(input_shape, hidden_dims, drop_rate=0.5, mask_value=None, train=False, freeze_prednet=True,
                 output_mode='representation_and_error', prediction_error_weight=0.9, rdm_error_weight=0.1, **config):
    if config is None:
        config = {}

    config['input_width'] = input_shape[1]
    config['input_height'] = input_shape[2]
    config['input_channels'] = input_shape[3]

    prednet = prednet_model.create_model(train=train, output_mode=output_mode, **config)
    prednet_layer = prednet.layers[1]
    prednet_layer.trainable = not freeze_prednet

    image_a = Input(shape=input_shape, name='image_a')
    image_b = Input(shape=input_shape, name='image_b')

    # Shared PredNet model
    if output_mode == 'representation_and_error':
        prednet_out_a = prednet(image_a)
        prednet_out_b = prednet(image_b)
        error_a = crop(2, start=-prednet_layer.nb_layers, name='crop_error_a')(prednet_out_a)
        error_a = prednet_model.get_error_layer(error_a, config['n_timesteps'], prednet_layer.nb_layers)
        rep_a = crop(2, start=0, end=-prednet_layer.nb_layers, name='crop_rep_a')(prednet_out_a)
        error_b = crop(2, start=-prednet_layer.nb_layers, name='crop_error_b')(prednet_out_b)
        error_b = prednet_model.get_error_layer(error_b, config['n_timesteps'], prednet_layer.nb_layers)
        rep_b = crop(2, start=0, end=-prednet_layer.nb_layers, name='crop_rep_b')(prednet_out_b)
        prednet_error = average([error_a, error_b], name='prednet_error')
    else:
        rep_a = prednet(image_a)
        rep_b = prednet(image_b)

    last_layer_shape = prednet_layer._PredNet__compute_layer_shape(input_shape, layer_num=prednet_layer.nb_layers - 1)
    last_layer_dim = int(np.prod(last_layer_shape))

    # Get last timestep
    # rep_a = crop(1, start=-1)(rep_a)
    # rep_b = crop(1, start=-1)(rep_b)

    # Get last layer representation
    out_a = crop(2, start=-last_layer_dim, name='crop_last_rep_a')(rep_a)
    out_b = crop(2, start=-last_layer_dim, name='crop_last_rep_b')(rep_b)

    # Shared LSTM model
    lstm_input = Input(shape=(input_shape[0], last_layer_dim))
    lstm_out = lstm_layer(lstm_input, mask_value, hidden_dims, drop_rate,
                          'lstm', return_sequences=False)
    lstm_model = Model(lstm_input, lstm_out, name='lstm')
    out_a = lstm_model(out_a)
    out_b = lstm_model(out_b)

    concatenated = concatenate([out_a, out_b], name='image_pair_representation')
    rdm_output = Dense(1, name="rdm_prediction")(concatenated)

    if output_mode == 'representation_and_error':
        model = Model([image_a, image_b], [prednet_error, rdm_output])

        if train:
            # define two dictionaries: one that specifies the loss method for
            # each output of the network along with a second dictionary that
            # specifies the weight per loss
            losses = {
                "prednet_error": "mean_absolute_error",
                "rdm_prediction": "mean_absolute_error",
            }
            loss_weights = {"prednet_error": prediction_error_weight,
                            "rdm_prediction": rdm_error_weight}

            print('Compiling model with loss weights:\n', loss_weights)
            # initialize the optimizer and compile the model
            model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights,
                          metrics=['mean_absolute_error', 'mean_squared_error'])
    else:
        model = Model([image_a, image_b], rdm_output)

        if train:
            model.compile(loss='mean_absolute_error', optimizer='adam',
                          metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()
    return model
