
FRAMES_PER_VIDEO = 90
AUDIO_FRAMES_PER_VIDEO = 30
SEQUENCES_PER_VIDEO = 5


def add_config(_configs, name, config, base_config=None):
    new_config = dict()
    if base_config:
        new_config.update(base_config)       
    new_config.update(config)
    new_config.update(models[new_config['model_name']])
    _configs[name] = new_config
    

configs = {}

models = {
    'prednet_kitti': {
        'model_weights_file': './kitti/model_data/kitti_keras/prednet_kitti_weights.hdf5',
        'model_json_file': './kitti/model_data/kitti_keras/prednet_kitti_model.json',
    },
    'prednet_random': {
        'model_weights_file': None,
        'model_json_file': None
    },
    'prednet_moments': {
        'model_weights_file': './models/prednet_moments__model{}/weights.hdf5',
        'model_json_file': './models/prednet_moments__model{}/model.json',
    },
    'prednet_kitti_finetuned_moments': {
        'model_weights_file': './models/prednet_kitti__moments__model{}/weights.hdf5',
        'model_json_file': './models/prednet_kitti__moments__model{}/model.json',
    },
    'prednet_random_finetuned_moments': {
        'model_weights_file': './models/prednet_random__moments__model{}/weights.hdf5',
        'model_json_file': './models/prednet_random__moments__model{}/model.json',
    },
    'prednet_moments_finetuned_algonauts': {
        'model_weights_file': './models/prednet_finetuned_moments__algonauts/weights.hdf5',
        'model_json_file': './models/prednet_finetuned_moments__algonauts/model.json',
    },
    'prednet_random_finetuned_algonauts': {
        'model_weights_file': './models/prednet_random__algonauts/weights.hdf5',
        'model_json_file': './models/prednet_random__algonauts/model.json',
    }
}

eval_base_config = {
    'n_timesteps': 10,
    'timestep_start': -1,
    'batch_size': 1,
    'stateful': False,
    'input_channels': 3, 
    'input_height': 256,
    'input_width': 256,
    'rescale': 1./255,
    'shuffle': False,
    'workers': 4,
    # DATA
    'pretrained': '10c',
    # RESULTS
    'base_results_dir': './feats/',
    'n_plot': 20
}

add_config(configs, 'prednet_random__prediction',
           { 'description': 'Using PredNet with random weights to evaluate predictions.',
             'model_name': 'prednet_random',
             'pretrained': None,
             'output_mode': 'prediction' }, eval_base_config)

add_config(configs, 'prednet_kitti__prediction',
           { 'description': 'Using PredNet pre-trained on KITTI dataset to evaluate predictions.',
             'model_name': 'prednet_kitti',
             'output_mode': 'prediction' }, eval_base_config)

add_config(configs, 'prednet_random__representation',
           { 'description': 'Using PredNet with random weights to extract features.',
             'model_name': 'prednet_random',
             'pretrained': None,
             'output_mode': 'representation',
           }, eval_base_config)

add_config(configs, 'prednet_random__error',
           { 'description': 'Using PredNet with random weights to extract features.',
             'model_name': 'prednet_random',
             'pretrained': None,
             'output_mode': 'prediction_error',
           }, eval_base_config)

add_config(configs, 'prednet_kitti__representation',
           { 'description': 'Using PredNet pre-trained on KITTI dataset to extract features.',
             'model_name': 'prednet_kitti',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_moments__representation',
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_moments',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__error',
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'output_mode': 'prediction_error' }, eval_base_config)

add_config(configs, 'prednet_kitti_finetuned_moments__prediction', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_kitti_finetuned_moments',
             'output_mode': 'prediction' }, eval_base_config)

add_config(configs, 'prednet_random_finetuned_moments__representation', 
           { 'description': 'Using PredNet pre-trained on Moments in Time dataset to extract features.',
             'model_name': 'prednet_random_finetuned_moments',
             'output_mode': 'representation' }, eval_base_config)

add_config(configs, 'prednet_random_finetuned_algonauts__representation',
           { 'description': 'Using PredNet pre-trained on Algonauts to extract features.',
             'model_name': 'prednet_random_finetuned_algonauts',
             'pretrained': None,
             'output_mode': 'representation',
           }, eval_base_config)

add_config(configs, 'prednet_moments_finetuned_algonauts__representation',
           { 'description': 'Using PredNet pre-trained on Algonauts dataset to extract features.',
             'model_name': 'prednet_moments_finetuned_algonauts',
             'pretrained': None,
             'output_mode': 'representation' }, eval_base_config)


train_base_config = dict()
train_base_config.update(eval_base_config)
train_base_config.update({
    'output_mode': 'error',
    'epochs': 50,
    'batch_size': 5,
    'shuffle': True,
    'training_data_dir': '../Training_Data/118_Image_set',
    'validation_data_dir': '../Training_Data/92_Image_set',
    'base_results_dir': './models',
    # 'gpus': 2,
    # 'stopping_patience': 100,
    'stack_sizes': (48, 96, 192, 192)
})

add_config(configs, 'prednet_kitti__algonauts',
           { 'description': 'Training PredNet (pre-trained on KITTI) on Algonauts dataset.',
             'model_name': 'prednet_kitti' }, train_base_config)

add_config(configs, 'prednet_random__algonauts',
           { 'description': 'Training PredNet from scratch on Algonauts dataset.',
             'pretrained': None,
             'model_name': 'prednet_random' }, train_base_config)

add_config(configs, 'prednet_finetuned_moments__algonauts',
           { 'description': 'Training PredNet from scratch on Algonauts dataset.',
             'model_name': 'prednet_kitti_finetuned_moments' }, train_base_config)

