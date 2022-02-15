
This document is intended as a User Guide for anyone interested in running the machine learning code on the quantum dot data.
For more details about the project, pease refer to the Project description document and to the manuscript posted on [arXiv:1712.04914](https://arxiv.org/abs/1712.04914), as well as the manuscript posted on [arXiv:2108.00043](https://arxiv.org/abs/2108.00043).

#### I. Software dependencies: 
+ Python 3.7 
+ Tensorflow 2.4 ([www.tensorflow.org](https://www.tensorflow.org/install/)) [installed in virtual enviroment]
+ Jupyter ([jupyter.readthedocs.io](http://jupyter.readthedocs.io/en/latest/install.html))


#### II. Data included in the QFlow lite

The QFlow lite repository contains the following files:

1) `readme.md` file.

2) `QFlow\config.py` — sets global variables for QFlow classes for size of subregions, number of state classes, and number of noise classes.

3) `QFlow\Crop_Data.py` — a set of Python functions to generate subregions from `hdf5` files with state labels (by default 10 subregions of size 30x30 pixels are generated per image).

4) `QFlow\Process_Data.py` — a set of Python functions to load and preprocess data saved as subregions. Preprocessing includes: resampling data to ensure even state distribution in the dataset, conversion of data `noise_level` to a `noise_class`, taking the gradient of data, noise reduction preprocessing, autoflipping, and zscore normalization.
 
5) `QFlow\Process_Data.py` — a set of Python functions for preparing to train machine learning model. `input_fn()` prepares data as a `Tensorflow.Dataset()` and `create_model()` returns a specified type of machine learning model.

6) `QF_TrainModel_.ipynb` — a Jupyter Notebook with code to generate a training data set, train the neural network, and test it on experimental data.

	- cell [1]: Import utility modules, plotting modules, and the QFlow modules.

	- cell [2]: Create random subregions for training from an `hdf5` file. This cell needs to be run only once if `save_data` is set to `True` as it will then store the created training data set locally in the same directory where the source `hdf5` file is stored.

	- cell [3]: Loading the training and evaluation datasets. If cropped data was not saved to disk, file argument may be replaced by a the `cropped_data` `dict`. Specifying the `label_key` will return different labels for the data for training either a state estimator or data quality control module. The format of the state label is `[ND, LD, CD, RD, DD]`. The format of the quality label is `[High, Moderate, Low]`.
    
    - cell [5]: Previewing the subregions. This cell allows to preview the training data of a selected subimage.

    - cell [6]: Preprocessing of the data. Preprocessing can be set as needed to have thresholding of clipping denoising, or autoflipping. Z-score normalization and an x gradient are always applied.

    - cell [7]: Previewing the processed subregions. This cell allows to preview the training data of a selected subimage as processed.

    - cell [8]: Create `TensorFlow.Dataset` from the processed training data.

	- cell [9]: Definition of the `model`, i.e., the network to train to classify the data and training of the network. Network parameters, such as the number of convolutional and pooling layers and the number and size of dense layers can be set to the values used in [arxiv.org/abs/2108.00043](https://arxiv.org/abs/2108.00043). Epochs are set low for demonstration purposes, but should be set to 30 for full convergence.

	- cell [10]: Visualization of the accuracy as the model is trained. This should show smooth convergence if models are performing as expected.

	- cell [11]: Evaluation of the trained network on experimental data. The code reads the experimental data from a local folder ‘Data/exp_data’. 


#### III. Training data

Data needed for training and evaluation can be found on the [NIST public data repository](https://doi.org/10.18434/T4/1423788). The data included a set of 20 `hdf5` files, 10 each intended for training state estimator and data quality control modules. The sets labeled as `normal_1.5m_0.5std_noisy_data_X.hdf5` are intended for training state estimators and the sets labeled as `uniform_noisy_data_quantile_X.X-X.X.hdf5` are intended for training data quality control. Each of these `.hdf5` files contain 1599 large simulated scans. The experimental data is stored as `.npy` files. 


#### IV. Getting your data ready

In order to use the trained network with your data you need to convert it to a format compatible with the `Tensorflow` `model`. The data needs to be stored as either a compressed set of 30 x 30 NumPy arrays (`*.npz`) or be loaded as a `dict`. The data should include enough features to make the distinction between each of the possible quantum dot states possible.


#### V. Typical workflow:

Execute the following steps within the QFlow training.ipynb

+ **Step 1:** Generate subimages
+ **Step 2:** Train CNN 
+ **Step 3:** Test CNN on experimental data provided with the package
+ **Step 4:** (optional): Prepare your own data following Sec. IV Getting your data ready and test CNN on it.

