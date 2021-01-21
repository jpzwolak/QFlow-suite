# Copyright 2017 The QFlow Team. All Rights Reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Last updated: April 4, 2018
# by Justyna P. Zwolak
# Info: added the progress bar
# Total time to generate 10 subregions per each data set: 489.3 s on 2017 MacBook Pro
# ==============================================================================

"""Downloading and reading QD data; setting up a folder 
structure compatible with the cnn_model_fn"""

import zipfile
import numpy as np
import os, errno
import glob
import random
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt

class QFlow():
    """
    Standard paths compatible with current version of the training code
    and parameters of the subimages
    """
     
    FOLDER = 'Data' # setting up the space for learning
    PATH_RAW = 'Data/raw_data/' # data will be extracted to this folder
    PATH_SUB = 'Data/sub_images/' # subimages will be stored in this folder
    PATH_EXP = 'Data/exp_data/' # experimental data is stored in this folder
    PATH_DATA = os.path.join(os.getcwd().replace("Code","Data")) # give a path to where the raw_data.zip file is
    # if the raw_data.zip file is in the same folder as QFlow training use 
    #PATH_DATA = os.getcwd()

    EXCL_RANGE = 10 # margin to be excluded when generating subregions, usually set to 0, bigger for the 250x250 maps
        
    def data_selection(self):
        """
        Allows user to specify which data to work with and the subregion size
        """

        var_s = eval(input('Please choose: \
                         \n\t1 for working with current data \
                         \n\t2 for working with sensor data. \
                         \nYour choice: '))
        
        if var_s == 1: 
            self.DATA_MAP = 'current_map'
        elif var_s == 2:
            self.DATA_MAP = 'sensor_map'
        
        print("You chose to work with the", self.DATA_MAP.replace('_', ' ')+'.')
        
        self.SUB_SIZE = int(input('Please, specify the size of the subregions:'))
        self.NUM_OF_SUBS = int(input('Please, specify the number of subrigon to generate per full map: '))
        
    def create_folder(self, f=FOLDER):
        """
        Creates a folder FOLDER unless it exists 
        Args:
            f = name of a folder to be created (Str)
        """
        try:
            os.makedirs(f)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
    def remove_folder(self, f=PATH_RAW):
        """
        Removes folder PATH_RAW with all files
        Args:
            f = path to a (possibly non-empty) folder that should be removed (Str)
        """
        if os.path.exists(f) and os.path.isdir(f):
            shutil.rmtree(f)
    
    def progress_bar(self, cur_iter, tot_iter, dec=1, bar_len=100, fill='█'):
        """
        Creates progress bar
        Args:
            cur_iter = current iteration (Int)
            tot_iter = total iterations (Int)
            dec = positive number of decimals in percent complete (Int)
            bar_len = character length of bar (Int)
            fill = Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(dec) + "f}").format(100 * (cur_iter / float(tot_iter)))
        filledLength = int(bar_len * cur_iter // tot_iter)
        bar = fill * filledLength + '-' * (bar_len - filledLength)
        print('\r%s |%s| %s%% %s' % ('Progress:', bar, percent, 'Complete'), end = '\r')
        if cur_iter == tot_iter: 
            print()
    
    def unzip_data(self, path_data=PATH_DATA, f=FOLDER):
        """
        Unzips the raw data
        Args:
            f = name of a folder where data will be extracted (Str)
        """
        print('Extracting raw_data.zip')
        with zipfile.ZipFile(os.path.join(path_data,'raw_data.zip'),'r') as zip_ref:
            noi = len(zip_ref.infolist())
            self.progress_bar(0, noi, bar_len = 50)
            for i in range(noi):
                zip_ref.extract(zip_ref.namelist()[i],f)
                self.progress_bar(i + 1, noi, bar_len = 50)
        print('raw_data.zip is now extracted')

    def extract_info(self, x, info='current'):
        """
        Extracts information of single type from raw files and stores as a numpy array
        Args:
            x = raw data
            info = refers to data that needs to be extracted (Str)
                   possible values: 'current', 'state', 'charge', 'sensor'
        """
        data = np.array([x['output'][i][info] for i in range(len(x['output']))])
        return data
    
    def extract_full(self, x):
        """
        Creates dicionary with data for a full region
        Args:
            x = raw data
        """
        keys = ('current_map', 'state_map', 'charge_map', 'sensor_data')
        objects = ('current', 'state', 'charge', 'sensor')
        values = [self.extract_info(x, el) for el in objects]
        data = dict(zip(keys, values))
        return data
    
    def size_full(self, x):
        """
        Retrives the size of the data
        Args:
            x = dictionary with extracted data
        """
        return len(x['V_P1_vec'])
    
    def range_full(self, x):
        """
        Retrives the voltage range from the data
        Args:
            x = dictionary with extracted data
        """
        return min(x['V_P1_vec']), max(x['V_P1_vec'])
 
    def reshape_full(self, x, size):
        """
        Reshapes the data to the desired format, finds net charge for the 'charge_map' and gradient for 
        sensor (currently dG_1/dP_1; for dG_2 use [:,:,1], for dP_2 use [0])
        Args:
            x = dictionary with extracted data
            size = the size of NumPy array for full region (Int)
        """
        data = {}
        data['current_map'] = x['current_map'].reshape(size,size)
        data['state_map'] = x['state_map'].reshape(size,size)
        #flatenning labels
        data['state_map'][data['state_map'] <= 0] = 0
        data['net_charge_map'] = np.array(list(map(sum,x['charge_map']))).reshape(size,size)
        data['sensor_map'] = np.gradient(x['sensor_data'].reshape(size,size,-1)[:,:,0])[1]
        return data

    def prob_vec(self, x):
        """
        Finds the label for subregions
        Args:
            x = dictionary with subregion data
        """
        dist = np.histogram(x['state_map'],[0,1,2,3])[0] #when using 4 lables as for current in need [-1,0,1,2,3]
        prob = dist/np.sum(dist)
        return prob
    
    def data_label(self, x):
        """
        Generates labels to be stored in data dictionary
        Args:
            x = dictionary with subregion data
        """
        x['label'] = self.prob_vec(x)
        return x
    
    def subimages(self, x, file, size, r_min, r_max, excl_range=EXCL_RANGE, f=PATH_SUB):
        """
        Generates subregions while keeping track of the plungers ranges
        Args:
            x = dictionary with full size data
            excl_range = margin to be excluded when generating subregions (Int)
            n = number of subimages per image (Int)
            f = path to a folder where subregion data will be stored (Str)
        """
        self.create_folder(f)
        for num in range(int(self.NUM_OF_SUBS)):
            x0,y0 =  random.sample(range(int(excl_range),int(size-self.SUB_SIZE-excl_range)), 2)
            keys = x.keys()
            values = [x[key][x0 : x0 + self.SUB_SIZE, y0 : y0 + self.SUB_SIZE] for key in keys]
            data = dict(zip(keys, values))    
            data['V_P1'] = np.linspace(r_min,r_max,size)[x0 : x0 + self.SUB_SIZE]
            data['V_P2'] = np.linspace(r_min,r_max,size)[y0 : y0 + self.SUB_SIZE]
            self.data_label(data)
            np.save(f+os.path.basename(file)[:-4]+"_"+str(self.SUB_SIZE)+"_subimage_"+ str(num),data)
            
    def slice_data(self, path_data=PATH_DATA, f=PATH_RAW):
        """
        Creating data set for machine learning (once the subregions are ready, the raw data is deleted)
        Args:
            f = path to raw data (Str)
        """
        if os.path.exists(os.path.join(path_data,'Data/sub_images/')):
            print('Subregions are ready')
            return
        elif os.path.exists(os.path.join(path_data,'raw_data/')):
            print('raw_data is already extracted')
            f = os.path.join(path_data,'raw_data/')
        else:
            self.unzip_data(path_data)            
        files = glob.glob(f + "*.npy")
        noi = len(files)
        size = self.size_full(np.load(files[0], allow_pickle=True).item())
        r_min, r_max = self.range_full(np.load(files[0], allow_pickle=True).item())  
        print('Generating subregions')
        self.progress_bar(0, noi, bar_len = 50)
        for i, file in enumerate(files):
            dat = np.load(file, allow_pickle=True).item()
            dat = self.extract_full(dat)
            dat = self.reshape_full(dat, size)
            dat = self.subimages(dat, file, size, self.SUB_SIZE, r_min, r_max)
            self.progress_bar(i + 1, noi, bar_len = 50)
        print('Done generating subregions')
        self.remove_folder()
        
    def dot_search(self, n=1, f=PATH_SUB):
        """
        Finds within the subregion files one that has at least 95% of the desired number of dots
        Args:
            n = number of dots (Int)
            f = path to subregion data (Str)
        """
        files = glob.glob(f + "*.npy")
        i = np.random.randint(int(len(files)/2))
        load_dat = np.load(files[i], allow_pickle=True).item()
        while load_dat['label'][n] < 0.95: 
            i += 1
            load_dat = np.load(files[i], allow_pickle=True).item()
        else:
            dot_dat = load_dat[self.DATA_MAP]
            dot_lab = np.round(load_dat['label'],2)
        return dot_dat, dot_lab  
        
    def data_preview(self):
        """
        Generates a preview of a single dot and a double dot region together with labels for each region
        """
        sd_dat, sd_lab = self.dot_search(1)
        dd_dat, dd_lab = self.dot_search(2)
        plt.figure(1,figsize=(12,4))
        
        if self.DATA_MAP == 'current_map':
            scaling = 10**4
        else:
            scaling = 1

        plt.subplot(121)
        plt.pcolor(scaling*sd_dat)
        plt.title('Single dot sub-region: '+ str(sd_lab))
        bar = plt.colorbar() 

        plt.subplot(122)
        plt.pcolor(scaling*dd_dat)
        plt.title('Double dot sub-region: '+ str(dd_lab))
        bar = plt.colorbar() 
        plt.show()
        
    def get_data(self, f=PATH_SUB):
        """
        Reads in the subregion data and converts it to a format useful for learning with cnn_model_fn
        Args:
            f = path to subregion data (Str)
        """
        files = glob.glob(f + "*.npy")
        inp = []
        oup = []
        
        if self.DATA_MAP == 'current_map':
            scaling = 10**4
        else:
            scaling = 1

        for file in files:
            data_dict = np.load(file, allow_pickle=True).item()
            inp += [data_dict[self.DATA_MAP]*scaling] # generates a list of arrays
            oup += [data_dict['label']] # generates a list of arrays
        
        inp = np.array(inp) # converts the list to np.array
        oup = np.array(oup) # converts the list to np.array
        n_samples = inp.shape[0]
        print("Total number of samples :", n_samples)
        n_train = int(0.9 * n_samples)
        n_eval = int(0.1 * n_samples)
    
        train_data = inp[:n_train].reshape(n_train,-1)
        train_labels = oup[:n_train].reshape(n_train,-1)

        eval_data = inp[n_train:(n_train+n_eval)].reshape(n_eval,-1)
        eval_labels = oup[n_train:(n_train+n_eval)].reshape(n_eval,-1)
    
        return train_data, train_labels, eval_data, eval_labels
        
    def train_net(self, classifier, batch_size=50, steps=5000):
        """
        Trains the network
        Args:
            classifier = estimator for model training
        """
        train_data, train_labels = self.get_data()[:2]
        print("Training samples :", len(train_labels))
        # Set up logging for predictions
        tensors_to_log = {}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True)
    
        classifier.train(
            input_fn=train_input_fn,
            steps=steps,
            hooks=[logging_hook])
    
    def eval_net(self, classifier, i=""):
        """
        Evaluates the network
        Args:
            classifier = estimator for model training
        """
        eval_data, eval_labels = self.get_data()[2:]
        print("Evaluating samples:", len(eval_labels))
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        np.save('Data/evaluation_data', eval_labels)
        print("Evaluation:", eval_results)
    
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            num_epochs=1,
            shuffle=False)

        predictions = list(classifier.predict(input_fn=predict_input_fn))
        np.save('Data/predictions_data', predictions)
        
    def evaluation_visual(self, i=""):
        """
        Generates a histogram of actual and predicted labels for the test set of simulated data
        """
        evals=np.load(os.path.join("Data/evaluation_data.npy"), allow_pickle=True)
        preds=np.load(os.path.join("Data/predictions_data.npy"), allow_pickle=True)

        
        sess = tf.Session()
        vec_true = sess.run(tf.argmax(input=evals, axis=1))

        vec_preds = []
        for i in range(len(preds)):
            vec_preds.append(preds[i]["state"])
    
        data = np.vstack([vec_true, vec_preds]).T
        ind = np.arange(4) 

        plt.hist(data, ind, alpha=0.7, label=['true', 'preds'])
        plt.legend(loc='upper left')
        plt.xticks(ind+0.5, ('none', 'SD','DD')) 
        plt.show()
        
    def get_exp_data(self, scaling, f=PATH_EXP): 
        """
        Reads in the subregion data and converts it to a format useful for learning with cnn_model_fn
        Args:
            scaling = factor used to rescale the experimental data to the right range (Int)
            data_map = defines what data to use (Str)
            f = path to experimental data (Str)
        """
        files = glob.glob(f + "*.npy")
        inp = []
        states = []

        for file in files:
            data_dict = np.load(file, allow_pickle=True).item()
            states += [os.path.basename(file)[:2]]
            inp += [data_dict[self.DATA_MAP]*scaling] 
        
        inp = np.array(inp) 
        n_test = inp.shape[0]
        test_data = inp.reshape(n_test,-1)

        return test_data, states

    def eval_exp(self, scaling, classifier):
        """
        Classifies the experimental data
        Args:
            scaling = a scaling factor to bring the data within a range consistent with the training data set
            classifier = estimator for model training
        """
        if self.DATA_MAP == "sensor_map":
            print("There is no experimental sensor data to evaluate at this point")
            return
        else:
            exp_data, exp_labels = self.get_exp_data(scaling)
    
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": exp_data},
                num_epochs=1,
                shuffle=False)
        
            return list(classifier.predict(input_fn=predict_input_fn))
    
    def exp_visual(self, exp_preds, f=PATH_EXP):
        """
        Args:
            f = path to experimental data (Str)
        """
        if self.DATA_MAP == "sensor_map":
            return
        else:
            files = glob.glob(f + "*.npy")
        
            for i in range(len(files)):
                load_dat = np.load(files[i], allow_pickle=True).item()
                exp_lab = np.round(exp_preds[i]['probabilities'],2)
                if exp_preds[i]['state'] == 2:
                    exp_state = 'double dot'
                elif exp_preds[i]['state'] == 1:
                    exp_state = 'sinlge dot'
                elif exp_preds[i]['state'] == 0:
                    exp_state = 'no dot'
               # else:
                #    exp_state = 'short circuit'
                
                plt.pcolor(load_dat['current_map'])
                bar = plt.colorbar() 
                plt.show()
                print('This image is classified as a', exp_state + ' (sub-region label: ', str(exp_lab) + ').\n' )
