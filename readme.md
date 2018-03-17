
This document is intended as a User Guide for anyone interested in running the machine learning code on the quantum dot data.
For more details about the project, pease refer to the Project description document and to the manuscript posted on [arXiv:1712.04914](https://arxiv.org/abs/1712.04914).

#### I. Software dependencies: 
+ Python 3.5 (Python 3.6 gives a warning `~/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6` but it does not affect the accuracy or performance)
+ Tensorflow 1.6 ([www.tensorflow.org](https://www.tensorflow.org/install/)) [installed in virtual enviroment]
+ Jupyter ([jupyter.readthedocs.io](http://jupyter.readthedocs.io/en/latest/install.html))


#### II. Data included in the QFlow lite

The QFlow lite repository contains the following files:

1) `readme.md` file.

2) `Project description.pdf` file.

3) `QFlow_class.py` — a set of Python functions to generate subregions (by default 10 subregions of size 30x30 pixels are generated per image) and read in data for training and evaluation. Upon execution, `QFlow_class` will create a file structure compatible with the training algorithm.

4) `QFlow training.ipynb` — a Jupyter Notebook with code to generate a training data set and to train the network

	- cell [1]: Import the QFlow class. In Python 3.6 it might return a warning described above but it can be ignored (it does not affect the accuracy or performance).

	- cell [2]: Create random subregions for training. This cell needs to be run only once, as it stores created training data set locally in the `Data/sub_images`. Argument `sub_size` allows to set the subregion size. It needs to be consistent with the `cnn_model_fn`. The progress of extracting and slicing data is visualized with a progress bar. Total time to generate 10 subregions, 30 x 30 pixels, per each data file on a 2017 MacBook Pro is about 7 minutes.

	- cell [2a]: Previewing the subregions. This cell allows to preview the training data together with the label for a given subregion. The format of the label is `[SC, QPC, SD, DD]`. It can be run as many times as needed. Each time the shown sub-region is selected at random from all subregions generated in Step 2.

	- cell [3]: Definition of the `cnn_model_fn`, i.e., the network used to train to classify the data. Network parameters, such as the number of convolutional and pooling layers and the number and size of dense layers can be modified to improve the performance of the network. With the default architecture, on a 2017 MacBook Pro the network trains on 30 x 30 pixel subregions to the accuracy of about 96.6% in about 10 minutes, with accuracy defined as the percentage of correctly classified subregions.

	- cell [4]: Execution of training and evaluation on simulated data. The code reads the subregion data and splits it into two sets (90% for training, 10% for evaluation). Then, using the `cnn_model_fn`, the network is trained to distinguish between single dots, double dots, short circuit and a barrier states. The trained network is stored in the `Data/trained_model` folder (created during training). To re-train the network on a new data set this folder needs to be removed or emptied.

	- cell [5]: Visualization of the classification accuracy for evaluation. A histogram showing the comparison of true data labels from the evaluation set (evals) and the labels predicted by the trained network (preds). The vertical axis represents the number of images in each category.

	- cell [6]: Evaluation of the trained network on experimental data. The code reads the experimental from a folder ‘Data/exp_data’. 

5) A `sample_data.csv` file that gives a preview of one full simulated map. Note that the "state" column identifies the actual state of the simulated device (-1: SC, 0: QPC, 1:SD, 2:DD).

6) `Data/exp_data` — a folder with four sample images generated from experimental data. 


#### III. Training data

Data neded for trainig can be foun at (https://doi.org/10.18434/T4/1423788). The `data.zip` folder contains a `data_structure.pdf` file, `license.pdf`, a copy of the `project_description.pdf` file and `raw_data` folder with the training data (1001 raw NumPy files storing the simulated 5-gate devices information). 


#### IV. Typical workflow:

Execute the following steps within the QFlow training.ipynb

+ **Step 1:** Generate subimages
+ **Step 2:** Train CNN 
+ **Step 3:** Test CNN on training set
+ **Step 4:** Test CNN on experimental data provided with the package
+ **Step 5:** (optional): Prepare your own data following Sec. III Getting your data ready and test CNN on it.

