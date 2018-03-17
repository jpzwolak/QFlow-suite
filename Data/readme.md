#### Getting your data ready

In order to use the trained network to your data you need to convert it to a format compatible with the ‘cnn_model_fn’. The data needs to be stored as a 30 x 30 NumPy array (`*.npy`) inside of a dictionary and labelled as `current_map`, and include enough features to make the distinction between single and double dot possible. Also, the file must be created/saved using python 3+. 

See the jupiter notebook for examples of good images. Moreover, since the network is trained to recognize the features qualitatively (and not quantitatively), the data need to be rescaled. In the training data set the background is around 0.0 and the peaks are around 1.0. The scaling factor has to be fitted heuristically and should be fixed for a given device. In our case, we multiplied the experimental data by a constant factor of 1e2.
