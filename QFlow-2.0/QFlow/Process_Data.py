import numpy as np
import random
from scipy.stats import skew as scipy_skew
from skimage.transform import resize as skimage_resize

from QFlow import config

## set of functions for loading and preparing a dataset for training.
def get_num_min_class(labels):
    '''
    Get the number of the minimum represented class in label vector.
    Used for resampling data.

    input:
        labels: np.ndarray of labels
    
    outputs:
        num_samples: int number of samples for minimum class
    '''
    # use argmax as example's class
    argmax_labels = np.argmax(labels, axis=-1)

    # max of num_samples is all one label
    num_samples = labels.shape[0]
    for i in range(labels.shape[-1]):
        lab_elems = np.sum(argmax_labels==i)
        if lab_elems < num_samples:
            num_samples = lab_elems

    return num_samples

def resample_data(features, state_labels, labels=None, seed=None):
    '''
    Resample data to be evenly distributed across classes in labels by cutting
    number of examples for each class to be equal to the number of examples
    in the least represented class. (classes assumed to be last axis of
    labels). Shuffles after resampling.

    inputs:
        features: ndarray of features to be resampled. Resample along first axis.
        state_labels: ndarray of labels to be used for resampling
        labels: ndarray of labels to be resampled.
        return_state: bool specifying whether to return state labels
        seed: Seed of random number generator for shuffling idxs during resample
            and for shuffling resampled features and labels.
    
    outputs:
        features: list of resampled features
        labels: list of resampled labels
    '''
    rng = np.random.default_rng(seed)
    num_samples = get_num_min_class(state_labels)

    features_resamp = []; state_labels_resamp = []; labels_resamp = []
    for i in range(state_labels.shape[-1]):
        s_idxs = state_labels.argmax(axis=-1)==i

        # first get full array of single state
        features_s_full = features[s_idxs]
        state_labels_s_full = state_labels[s_idxs]
        if labels is not None:
            labels_s_full = labels[s_idxs]

        # then get idxs (0-length), shuffle, and slice to num_samples
        # shuffle idxs to be sure labels and features are shuffled together
        idxs = list(range(features_s_full.shape[0]))
        rng.shuffle(idxs)
        features_resamp.append(features_s_full[idxs[:num_samples]])
        state_labels_resamp.append(state_labels_s_full[idxs[:num_samples]])
        if labels is not None:
            labels_resamp.append(labels_s_full[idxs[:num_samples]])

    features_resamp_arr = np.concatenate(features_resamp, axis=0)
    state_labels_resamp_arr = np.concatenate(state_labels_resamp, axis=0)
    if labels is not None:
        labels_resamp_arr = np.concatenate(labels_resamp, axis=0)

    idxs = list(range(features_resamp_arr.shape[0]))
    rng.shuffle(idxs)
    if labels is not None:
        return features_resamp_arr[idxs], labels_resamp_arr[idxs]
    elif labels is None:
        return features_resamp_arr[idxs], state_labels_resamp_arr[idxs]


def noise_mag_to_class(state_labels, noise_mags,
                        low_thresholds=None, high_thresholds=None):
    '''
    Function to convert noise magnitudes to noise classes.
    Noise class thresholds are defined here. Thresholds for states
    order is: no dot, left dot, central dot, right dot, double dot
    Default low thresholds is the linear extrapolation to 100 % accuracy
    of an average noisy-trained model vs. noise_mag. Default high
    thresholds are from linear extrapolation to 0 % accuracy of an
    average noisy trained model vs. noise_mag.
    
    inputs:
        state_labels: list of state labels. shape assumed to be
            (num_examples, num_states).
        noise_mags: list of float noise_mags for state_labels. shape assumed
            to be (num_examples, ).
        low_thresholds: list of floats of shape (num_state, ) specifying
            high signal to noise class thresholds. 
        high_thresholds: list of floats of shape (num_state, ) specifying
            high signal to noise class thresholds. 
    '''
    # set number of noise classes and states.
    # length of thresholds must be equal to num_states.
    # no num_quality_classes != 3 are supported.
    num_quality_classes = config.NUM_QUALITY_CLASSES
    num_states = config.NUM_STATES

    # set default thresholds
    if high_thresholds is None:
        high_thresholds = [1.22, 1.00, 1.21, 0.68, 2.00]
    if low_thresholds is None:
        low_thresholds = [0.31, 0.32, 0.41, 0.05, 0.47]

    low_thresholds = np.array(low_thresholds)
    high_thresholds = np.array(high_thresholds)

    quality_classes = np.zeros(noise_mags.shape+(num_quality_classes,))

    # use fractional labels by taking weighted average after
    # applying thresholds
    num_states = state_labels.shape[-1]

    # get per state classes then sum across last axis later
    per_state_classes = np.zeros(
        noise_mags.shape + (num_quality_classes,) + (num_states,))

    # use boolean indexing to define classes from noise mags/threshold arrays
    for i in range(num_states):
        per_state_classes[noise_mags <= low_thresholds[i],0, i] = 1
        per_state_classes[(noise_mags > low_thresholds[i]) &\
                            (noise_mags <= high_thresholds[i]), 1, i] = 1
        per_state_classes[noise_mags > high_thresholds[i], 2, i] = 1

    # multiply each first axis element then sum across last axes
    quality_classes = np.einsum('ijk,ik->ij', per_state_classes, state_labels)

    return quality_classes

def get_data(f, train_test_split=0.9, 
             dat_key='sensor', label_key='state',
             resample=True, seed=None, 
             low_thresholds=None, high_thresholds=None):
    '''
    Reads in the subregion data and converts it to a format useful for training
    Note that the data is shuffled after reading in.

    inputs:
        f: one of: 
            str path to .npz file containing cropped data
            dict of cropped data.
        train_test_split: float fraction of data to use for training.
        resample: bool specifying whether to resample data to get even state
            representation.
        seed: int random seed for file shuffling.
        label_key: string key for data used for the label. One of: 
            'data_quality', 'noise_mag_factor', 'state'.
        low_threshold: list of noise levels to use for high/moderate signal
            to noise ratio threshold.
        high_threshold: list of noise levels to use for moderate/low signal
            to noise ratio threshold.

    outputs:
        train_data: np.ndarray of training data.
        train_labels: np.ndarray of training labels.
        eval_data: np.ndarray of training data.
        eval_labels: np.ndarray of training labels.
    '''
    # treat f as path, or if TypeError treat as dict.
    try:
        dict_of_dicts = np.load(f, allow_pickle = True)
        file_on_disk = True
    except TypeError:
        dict_of_dicts = f
        file_on_disk = False

    files = list(dict_of_dicts.keys())
    random.Random(seed).shuffle(files)
    inp = []
    oup_state = []
    # if we want a nonstate label load it so we can resample
    if label_key!='state':
        oup_labels = []
    else:
        oup_labels = None
        train_labels = None
        eval_labels = None

    # if label is noise class, we need to get noise mag labels first
    # then process to turn the mag into a class label
    if label_key == 'data_quality':
        data_quality = True
        label_key = 'noise_mag_factor'
    else:
        data_quality = False

    for file in files:
        # for compressed data, file is the key of the dict of dicts
        if file_on_disk:
            data_dict = dict_of_dicts[file].item()
        else:
            data_dict = dict_of_dicts[file]

        dat = data_dict[dat_key]

        # generates a list of arrays
        inp.append(dat.reshape(config.SUB_SIZE,config.SUB_SIZE,1))
        oup_state.append(data_dict['state']) # generates a list of arrays
        if oup_labels is not None:
            oup_labels.append(data_dict[label_key])

    inp = np.array(inp) # converts the list to np.array
    oup_state = np.array(oup_state) # converts the list to np.array
    if oup_labels is not None:
        oup_labels = np.array(oup_labels)

    # split data into train and evaluatoin data/labels
    n_samples = inp.shape[0]
    print("Total number of samples :", n_samples)
    n_train = int(train_test_split * n_samples)

    train_data = inp[:n_train]
    print("Training data info:", train_data.shape)
    train_states = oup_state[:n_train]
    if oup_labels is not None:
        train_labels = oup_labels[:n_train]

    eval_data = inp[n_train:]
    print("Evaluation data info:", eval_data.shape)
    eval_states = oup_state[n_train:]
    if oup_labels is not None:
        eval_labels = oup_labels[n_train:]

    # convert noise mag to class before resampling/getting noise mags if 
    # needed because resampling doesnt return state labels
    if data_quality:
        train_labels = noise_mag_to_class(
            train_states, train_labels,
            low_thresholds=low_thresholds,
            high_thresholds=high_thresholds,
        )
        eval_labels = noise_mag_to_class(
            eval_states, eval_labels,
            low_thresholds=low_thresholds,
            high_thresholds=high_thresholds,
        )

    # resample to make state representation even
    if resample:
        train_data, train_labels = resample_data(
            train_data, train_states, train_labels)
        eval_data, eval_labels = resample_data(
            eval_data, eval_states, eval_labels)
    elif not resample and label_key=='state':
        train_labels = train_states
        eval_labels = eval_states

    # expand dim of labels to make sure that they have proper shape
    if oup_labels is not None and len(train_labels.shape)==1:
        np.expand_dims(train_labels, 1)
    if oup_labels is not None and len(eval_labels.shape)==1:
        np.expand_dims(eval_labels, 1)

    return train_data, train_labels, eval_data, eval_labels

## preprocess functions
def gradient(x):
        '''
        Take gradient of an ndarray in specified direction. Thin wrapper around
        np.gradient(). Also note that x -> axis=1 and y-> axis=0
        
        input:
            x: An numpy ndarray to take the gradient of 
        output:
            numpy ndarray containing gradient in x direction.
        '''
        return np.gradient(x, axis=1)

def apply_threshold(x, threshold_val=10, threshold_to=0):
    '''
    Thresholds an numpy ndarray to remove
    Args:
        x = numpy array with data to be filtered
        threshold_val = percentile below which to set values to zero
    '''
    x[x < np.abs(np.percentile(x.flatten(),threshold_val))] = threshold_to
    return x

def apply_clipping(x, clip_val=3, clip_to='clip_val'):
    '''
    Clip input symmetrically at clip_val number of std devs.
    Do not zscore norm x, but apply thresholds using normed x
    '''
    x_clipped = np.copy(x)
    mean = np.mean(x)
    std = np.std(x)
    norm_x = (x - mean) / std

    # set clipped values to either the mean or clip threshold
    if clip_to.lower() == 'clip_val':
        x_clipped[norm_x < -clip_val] = -clip_val * std + mean
        x_clipped[norm_x > clip_val] = clip_val * std + mean
    elif clip_to.lower() == 'mean':
        x_clipped[norm_x < -clip_val] = mean
        x_clipped[norm_x > clip_val] = mean
    else:
        raise KeyError('"clip_to" option not valid: ' +str(clip_to) +\
            'Valid options: clip_val, mean')

    return x_clipped

def autoflip_skew(data):
    '''
    Autoflip a numpy ndarray based on the skew of the values 
    (effective for gradient data).
    '''
    skew_sign = np.sign(scipy_skew(np.ravel(data)))
    return data*skew_sign

def zscore_norm(x):
    '''
    Takes a numpy ndarray and returns a z-score normalized version
    '''
    return (x-x.mean())/x.std()


class Preprocessor():

    def __init__(self, autoflip=False, denoising=[], 
        clip_val=None, thresh_val=None):
        '''
        Class for doing preprocessing of data.

        inputs:
            autoflip: bool specifying whether to autoflip data.
            denoising: list of str specifying denoising to apply to data.
            clip_val: value for clipping denoising. Unused if 'clip' not in
                denoising.
            thresh_val
        '''
        self.autoflip = autoflip

        valid_denoising = ['threshold', 'clip']
        if not set(denoising).issubset(valid_denoising):
            raise ValueError(
                'invalid denoising ', denoising, 
                ' Valid values:', valid_denoising)
        self.denoising = denoising

        self.clip_val = clip_val
        self.thresh_val = thresh_val
    
    def proc_subimage(self, x):
        '''
        Takes the gradient of the measured data, applies denoising if specified,
        normalizes, autoflips if specified,
        and then adjusts the size (if necessary)
        Args:
            x = an array with data
        '''

        # take gradient
        x = gradient(x)

        # apply thresholding
        if 'threshold' in self.denoising:
            if self.threshold_val is not None:
                grad_x = apply_threshold(x, self.threshold_val)
            else:
                grad_x = apply_threshold(x)

        # apply clipping
        if 'clip' in self.denoising:
            if self.clip_val is not None:
                grad_x = apply_clipping(grad_x, self.clip_val)
            else:
                grad_x = apply_clipping(grad_x)

        # normalize with zscore normalization
        x = zscore_norm(x)
        
        # autoflip by skew of image gradient
        if self.autoflip:
            x = autoflip_skew(x)

        target_shape = (config.SUB_SIZE, config.SUB_SIZE, 1)
        if x.shape != target_shape:
            x = skimage_resize(x, target_shape)
        return x

    def proc_subimage_set(self, x_arr):
        '''
        Loop through subimages and apply preprocessing to each one.

        inputs:
            x: full dataset of images. First axis assumed to be example index.
        returns:
            Full dataset of images with same shape, processed.
        '''
        return np.array([self.proc_subimage(x) for x in x_arr])