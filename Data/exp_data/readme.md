#### Getting your data ready

In order to use the trained network to your data you need to convert it to a format compatible with the `cnn_model_fn`. The data needs to be stored as a 30 x 30 NumPy array (`*.npy`) and include enough features to make the distinction between single and double dot possible. In the training data set the background is around 0.0 and the peaks are around 1.0. See the jupiter notebook for examples of well formatted images. Moreover, since the network is trained to recognize the features qualitatively (and not quantitatively), the data need to be rescaled. The scaling factor has to be fitted heuristically and should be fixed for a given device. In our case, we multiplied the experimental data by a constant factor of 1e2.

