from tensorflow import data as tf_data
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import Model as tf_Model 
from tensorflow.keras.optimizers import Adam as tf_Adam

from QFlow import config

def input_fn(features, labels, shuffle=True, batch_size=64,
                repeat=False, seed=None):
    '''
    A function for converting data into training/evaluation tf.Dataset

    inputs:
        features: np.ndarray containing features.
        labels : np.ndarray containing labels for all examples.
        shuffle : bool indicates whether to shuffle the dataset.
        batch_size : int indicating the desired batch size.
        repeat: bool specifying whether to repeat dataset.
        seed: int seed used for shuffling dataset.
    outputs:
        ds : dataset ready for training/evaluation
    '''
    # Convert the inputs to a Dataset.
    shuffle_buffer_size = 100
    ds = tf_data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size, seed=seed).batch(batch_size)
    else:
        ds = ds.batch(batch_size)

    if repeat:
        ds = ds.repeat()

    return ds

def create_model(model_type='state_estimator', 
    model_opt='best_noise_opt'):
    '''
    inputs:
        model_type: str specifying either 'state_estimator' or 
            'quality_control' type machine learning model.
        model_opt: str specifying dataset the model parameters were optimized 
            on. Valid options for 'state_estimator' model_type: 
            'noiseless_opt' or 'best_noise_opt'. Valid options for 
            'quality_control' type: 'uniform_noise_dist_opt'.
    '''
    valid_model_types = ['state_estimator','quality_control']
    if model_type not in valid_model_types:
        raise ValueError(
            'model_type not recognized: ', model_type,
            ' Valid values: ', valid_model_types)

    valid_model_opts = {
        'state_estimator': ['noiseless_opt', 'best_noise_opt'],
        'quality_control': ['uniform_noise_dist_opt']}
    if model_opt not in valid_model_opts[model_type]:
        raise ValueError(
            'model_opt not recognized: ', model_opt,
            ' Valid values: ', valid_model_opts[model_type])

    if model_type=='state_estimator' and model_opt=='best_noise_opt':
        lr = 1.21e-3
        k_size = [[7, 7], [7, 7]]
        cnn_maxpool = False
        cnn_stack = 2
        n_cnn = 2
        # these lists should be length n_cnn
        n_filters = [[22, 22], [35, 35]]
        drop_rates = [[0.655,0.655], [0.194, 0.194]]
        layer_norm = False
        ave_pool = True
        activation='relu'
        dense_n = 0
    elif model_type=='state_estimator' and model_opt == 'noiseless_opt':
        lr = 3.45e-3
        k_size = [[5], [5], [5]]
        cnn_maxpool=False

        cnn_stack = 1
        n_cnn = 3

        n_filters = [[23], [7], [18]]
        drop_rates = [[0.12], [0.28], [0.30]]
        layer_norm = True
        ave_pool = True
        activation = 'relu'

        dense_n = 0
    elif model_type=='quality_control' and model_opt=='uniform_noise_dist_opt':
        lr = 2.65e-4
        k_size = [[7, 3]]
        cnn_maxpool = True
        cnn_stack = 2
        n_cnn = 1

        n_filters = [[184, 249]]
        drop_rates = [[0.05, 0.0]]
        layer_norm = True
        ave_pool = True
        activation='swish'

        dense_n = 1
        dense_dropout = [0.6]
        dense_units = [161]

    # set stride to 2 if not using maxpool as size reduction
    if cnn_maxpool:
        cnn_stride=1
    else:
        cnn_stride=2

    # input layer
    inputs = tf_layers.Input(shape=(config.SUB_SIZE,config.SUB_SIZE,1))
    x = inputs
    for i in range(n_cnn):
        for j in range(cnn_stack):
            if j==cnn_stack-1:
                stride = cnn_stride
            else:
                stride=1
            x = tf_layers.Conv2D(
                filters=n_filters[i][j],
                kernel_size=k_size[i][j],
                padding='same',
                strides=stride)(x)
            x = tf_layers.Dropout(rate=drop_rates[i][j])(x)
            if layer_norm:
                x = tf_layers.LayerNormalization()(x)
            x = tf_layers.Activation(activation)(x)
        if cnn_maxpool:
            x = tf_layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

    if ave_pool:
        x = tf_layers.GlobalAvgPool2D()(x)

    x = tf_layers.Flatten()(x)

    for i in range(dense_n):
        x = tf_layers.Dense(units=dense_units[i],activation=activation)(x)
        x = tf_layers.Dropout(rate=dense_dropout[i])(x)

    if model_type=='state_estimator':
            outputs = tf_layers.Dense(
                units=config.NUM_STATES, activation='softmax')(x)
            model = tf_Model(inputs, outputs, 
            name='device_state_estimator_'+model_opt)

            model.compile(
                optimizer=tf_Adam(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    elif model_type=='quality_control':
        outputs = tf_layers.Dense(
            units=config.NUM_QUALITY_CLASSES, activation='softmax')(x)
        model = tf_Model(
            inputs=inputs, outputs=outputs, 
            name='data_quality_control_'+model_opt)
        model.compile(
            optimizer=tf_Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    return model