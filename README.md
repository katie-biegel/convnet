
Files:
cnn.py - python script to run neural network
README.md - this README file

Example Data files:
h1waves_test_csv - h1 component for testing data
h1waves_train.csv - h1 component for training data
h2waves_test.csv - h2 component for testing data
h2waves_train.csv - h2 component for training data
parrivals_test.csv - p arrival picks for testing data
parrivals_train.csv - p arrival picks for training data
sarrivals_test.csv - s arrival picks for testing data
sarrivals_train.csv - s arrival picks for training data
timing_test.csv - waveform timing for testing data
timing_train.csv - waveform timing for training data
Zwaves_test.csv - vertical component for testing data
Zwaves_train.csv - vertical component for training data

Outputs:
figures - figures included in final paper
results - outcome figures for running neural network (will overwrite if code is run)
predicted_model.npy - python numpy file for the network output arrays.  These are probability outcomes for p and s arrivals for each testing waveform.
weight.h5 - neural network model weights found from the fitting training data to model


To run cnn.py the following python packages are required in addition to the regular Anaconda 3 installation:
- tqdm
- skimage
- kera
- tensorflow

