
import random, h5py
import numpy as np
from skimage.transform import resize as skimage_resize

from QFlow import config

# class for preparing subimages from large scan for training.
class DataCropper():

    def __init__(self, excl_range):
        '''
        Class for cropping subimages from larg maps, keeping only data needed 
        for training.

        inputs:
            excl_range: number of pixels from the edge of the image to exclude 
                from subimages.
        '''
        # number of excluded pixels at edge of image
        self.excl_range = excl_range

    @staticmethod
    def prob_vec(x):
        '''
        Finds the label for region x. Note that data is multiplied
            by 2 to avoid fractional labels
        Args:
            x = dictionary with label data
        '''
        dist = np.histogram(2*x['state'],[0,1,2,3,4,5])[0]
        prob = dist/np.sum(dist)
        return prob

    @staticmethod
    def data_label(x):
        '''
        Generates labels to be stored in data dictionary
        Args:
            x = dictionary with label data
        '''
        x['state'] = DataCropper.prob_vec(x)
        return x

    def crop_data(self, noisy_out, noisy_state, noisy_x, noisy_y, noise_factor):
        '''
        Gets cropped data from full noisy_outputs.

        inputs:
            noisy_out: 2d ndarray of noisy sensor data.
            noisy_state: 2d ndarray of state data.
            noisy_x: 1d ndarray of x data.
            noisy_y: 1d ndarray of y data.
            noise_factor: float specifying level of noise in data (white noise,
                1/f noise, and telegraph noise [sensor jumps])
        outputs:
            data: dict containing noise_factor and cropped noisy_out, 
                noisy_state, noisy_x, and noisy_y.
        '''
        data = {}

        # function to randomly pick an origin for a subimage
        def sub_position(x):
            return random.sample(
                range(int(self.excl_range),
                int(x-config.SUB_SIZE-self.excl_range)), 1)[0]

        x_len = len(noisy_x)
        y_len = len(noisy_y)

        # get origin for cropped image with clever random sample
        x0, y0 = sub_position(x_len), sub_position(y_len)

        # get cropped data starting from randomly generated x0, y0
        data['sensor'] = noisy_out[y0:y0 + config.SUB_SIZE, x0:x0 + config.SUB_SIZE]
        data['state'] = noisy_state[y0:y0 + config.SUB_SIZE, x0:x0 + config.SUB_SIZE]

        # get noise level number 
        data['noise_mag_factor'] = noise_factor

        # save similarly cropped x and y voltages
        data['V_P1'] = np.linspace(noisy_x[x0],
                                   noisy_x[x0 + config.SUB_SIZE], config.SUB_SIZE)
        data['V_P2'] = np.linspace(noisy_y[y0],
                                   noisy_y[y0 + config.SUB_SIZE], config.SUB_SIZE)

        # get and add data label as prob vector of states
        self.data_label(data)

        return data

    def crop_full_dataset(self, h5_filename, data_key='noisy_sensor', 
            x_key='V_P1_vec', y_key='V_P2_vec', subs_per_map=10, save_data=True,
            return_data=False):
        '''
        Crop all entries in an hdf5 file into subimages.

        inputs:
            h5_filename: str path to h5 file containing training data.
            data_key: str key of z data in h5 file.
            x_key: str key of x data in h5 file.
            y_key: str key of y data in h5 file.
            subs_per_map: int number of subimages to get per map.
            save_data: bool whether to save data (as compressed .npz)
            return_data: bool whether to return data (as dict)
        outputs:
            cropped_data: np.ndarray containing training data as subimages.
        '''
        # make sure that either save_data or return_data is true
        if not save_data and not return_data:
            raise ValueError(
                'save_data False and return_data False. Choose one!')

        cropped_data_dict = {}
        with h5py.File(h5_filename, 'r') as h5f:
            for k, v in h5f.items():
                for i in range(subs_per_map):
                    data = self.crop_data(
                        v['output'][data_key][()], v['output']['state'][()], 
                        v[x_key][()], v[y_key][()], v['noise_mag_factor'][()])

                    #  create list of cropped data if wanted to save memory
                    cropped_data_dict[k+('%i'%i)] = data

        print(len(cropped_data_dict.keys()))
        if save_data:
            numpy_filename = '.'.join(h5_filename.split('.')[:-1])+'.npz'
            np.savez_compressed(numpy_filename, **cropped_data_dict)

        if return_data:
            return cropped_data_dict
        else:
            return None