from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, TimeDistributed, Dense, Flatten
import numpy as np
import h5py

from prednet import PredNet


def load_model(model_json_file, model_weights_file, **extras):
    print('Loading model: {}'.format(model_weights_file))
    with open(model_json_file, 'r') as f:
        json_string = f.read()
        train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
        train_model.load_weights(model_weights_file)
    return train_model


def create_model(model_json_file=None, model_weights_file=None,
                 train=False, **config):
    if model_json_file and model_weights_file:
        pretrained_model = load_model(model_json_file, model_weights_file)
        model = pretrained_prednet(pretrained_model, train=train, **config)
    else:
        model = random_prednet(train=train, **config)
    return model


def pretrained_prednet(pretrained_model, n_timesteps, output_mode='error',
                       train=False, stateful=False, batch_size=None,
                       trainable_layers=None, trainable_units=None,
                       **config):
    prednet_model = pretrained_model

    if 'stack_sizes' not in prednet_model.layers[1].get_config():
        for layer in pretrained_model.layers:
            if 'prednet' in layer.name.lower():
                print('Found PredNet in layer', layer.name)
                prednet_model = layer
                break

    layer_config = prednet_model.layers[1].get_config()
    layer_config['output_mode'] = output_mode
    layer_config['stateful'] = stateful
    prednet = PredNet(weights=prednet_model.layers[1].get_weights(),
                      trainable_layers=trainable_layers,
                      trainable_units=trainable_units, **layer_config)
    input_shape = list(prednet_model.layers[0].batch_input_shape[1:])
    input_shape[0] = n_timesteps
    inputs = get_input_layer(batch_size, tuple(input_shape))
    outputs = get_output_layer(prednet, inputs, n_timesteps, train, output_mode)
    model = Model(inputs=inputs, outputs=outputs, name='PredNet')
    return model


def random_prednet(input_channels, input_height, input_width,
                   n_timesteps, stack_sizes=(48, 96, 192),
                   train=False, output_mode='error', stateful=False,
                   batch_size=None, trainable_layers=None,
                   trainable_units=None, **config):
    # Model parameters
    if K.image_data_format() == 'channels_first':
        input_shape = (input_channels, input_height, input_width)
    else:
        input_shape = (input_height, input_width, input_channels)

    stack_sizes = (input_channels,) + stack_sizes
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3,) * (len(stack_sizes) - 1)
    Ahat_filt_sizes = (3,) * len(stack_sizes)
    R_filt_sizes = (3,) * len(stack_sizes)
    prednet = PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode=output_mode, return_sequences=True,
                      stateful=stateful, trainable_layers=trainable_layers,
                      trainable_units=trainable_units, name='prednet_layer')
    input_shape = (n_timesteps,) + input_shape
    inputs = get_input_layer(batch_size, input_shape)
    outputs = get_output_layer(prednet, inputs, n_timesteps, train, output_mode)
    model = Model(inputs=inputs, outputs=outputs, name='PredNet')
    return model


def get_input_layer(batch_size, input_shape):
    if batch_size:
        input_shape = (batch_size,) + input_shape
        inputs = Input(batch_shape=input_shape)
    else:
        inputs = Input(shape=input_shape)
    return inputs


def get_output_layer(prednet, inputs, n_timesteps, train, output_mode):
    outputs = prednet(inputs)
    if train:
        if output_mode not in ['error', 'representation_and_error']:
            raise ValueError('When training, output_mode must be equal to "error"')
    if output_mode == 'error':
        outputs = get_error_layer(outputs, n_timesteps, prednet.nb_layers)
    return outputs


def get_error_layer(outputs, n_timesteps, nb_layers):
    # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.array([0.1] * nb_layers)
    layer_loss_weights[0] = 1
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
    # equally weight all timesteps except the first
    time_loss_weights = 1. / (n_timesteps - 1) * np.ones((n_timesteps, 1))
    time_loss_weights[0] = 0

    # calculate weighted error by layer
    errors_by_time = TimeDistributed(Dense(1, trainable=False),
                                     weights=[layer_loss_weights, np.zeros(1)],
                                     trainable=False)(outputs)
    errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, n_timesteps)
    final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)],
                         trainable=False)(errors_by_time)  # weight errors by time
    return final_errors


def load_prednet_weights(weights_path, prednet_model):
    """
    Load model weights from HDF5 directly and set weights tensor by tensor.
    This implementation is specific for a PredNet model.
    This is workaround for some issues related to HDF5 loading on Google Colab:
    OSError: Unable to open file (File signature not found)

    :param prednet_model: Keras model instance to set the weights
    :param weights_path: path for HDF5 weights file
    """

    weights = h5py.File(weights_path)
    weight_keys = list(weights.keys())
    if 'model_weights' in weight_keys:
        weights = weights['model_weights']
        weight_keys = list(weights.keys())

    prednet_key = None
    for key in weight_keys:
        if 'prednet' in key.replace('_', ''):
            prednet_key = key
            break

    if prednet_key is None:
        raise ValueError('Weights for prednet layer not found')

    prednet_weights_keys = list(weights[prednet_key].keys())
    prednet_weights = weights[prednet_key][prednet_weights_keys[0]]

    total_weights = 0
    weight_value_tuples = []
    for layer_key in prednet_weights:
        layer_weights = prednet_weights[layer_key]
        total_weights += len(layer_weights)

        for weight_key in layer_weights:
            # print(f'Loading weights for layer {layer_key}, {weight_key} shape = {layer_weights[weight_key].shape}')
            tensor_name = f'{layer_key}/{weight_key}'

            for tensor in prednet_model.layers[1].weights:
                if tensor_name in tensor.name:
                    # print(f'Setting tensor {tensor.name} to value in weights[{tensor_name}] - shape = {layer_weights[weight_key].shape}')
                    weight_value_tuples.append((tensor, layer_weights[weight_key]))

    if weight_value_tuples:
        K.batch_set_value(weight_value_tuples)
    print(f'{len(weight_value_tuples)}/{total_weights} weights loaded from {weights_path}')


def load_model_weights(weights_path, target_layer, source_weights_key=None):
    """
    Load model weights from HDF5 directly and set weights tensor by tensor.
    This is workaround for some issues related to HDF5 loading on Google Colab:
    OSError: Unable to open file (File signature not found)

    :param weights_path: path for HDF5 weights file
    :param target_layer: layer for which we want to set the weights
    :param source_weights_key: some specific HDF5 key for the weights
    """
    weights = h5py.File(weights_path)
    weight_keys = list(weights.keys())
    if 'model_weights' in weight_keys:
        weights = weights['model_weights']
        weight_keys = list(weights.keys())

    if source_weights_key is None:
        source_weights_key = target_layer.name

    model_weights_key = None
    for key in weight_keys:
        if source_weights_key in key:
            model_weights_key = key
            break

    if model_weights_key is None:
        raise ValueError(f'Weights for layer {target_layer.name} not found')

    layer_weights_keys = list(weights[model_weights_key].keys())
    all_layer_weights = weights[model_weights_key][layer_weights_keys[0]]

    total_weights = 0
    weight_value_tuples = []
    for layer_key in all_layer_weights:
        layer_weights = all_layer_weights[layer_key]
        total_weights += len(layer_weights)

        if type(layer_weights[0]) == str:
            for weights in layer_weights:
                # print(f'Loading weights for layer {layer_key}, {weights} shape = {layer_weights[weights].shape}')
                tensor_name = f'{layer_key}/{weights}'
                weights = layer_weights[weights]

                for tensor in target_layer.weights:
                    print(tensor_name, tensor.name)
                    if tensor_name in tensor.name:
                        weight_value_tuples.append((tensor, weights))
                        break

        else:
            weights = layer_weights
            # print(f'Loading weights for layer {layer_key} shape = {weights.shape}')
            tensor_name = f'{layer_key}'

            for tensor in target_layer.weights:
                print(tensor_name, tensor.name)
                if tensor_name in tensor.name:
                    # print(f'Setting tensor {tensor.name} to value in weights[{tensor_name}] - shape = {weights.shape}')
                    weight_value_tuples.append((tensor, weights))
                    break

    if weight_value_tuples:
        K.batch_set_value(weight_value_tuples)
    print(f'{len(weight_value_tuples)}/{total_weights} weights loaded from {weights_path}')