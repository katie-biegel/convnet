#!/usr/bin/env python
import sys
import numpy as np
import math
import os
import collections
import subprocess
import csv
import random
import warnings
import pandas as pd

from matplotlib import pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense, Flatten, Convolution1D, Dropout
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv1D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import concatenate, Concatenate
#from keras.layers import'relu
from keras.optimizers import SGD
from keras.initializers import random_uniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.layers.advanced_activations import'relu
from keras import backend as K

import tensorflow as tf


####################################
# Function Declarations
####################################

# Function for fining the value nearest in array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Function to translate Conv2DTranspose in Keras to Conv1DTranspose
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


#############################################
# Build Data from TOC2ME Waveforms
#############################################


# Read in Data
###

# Read in timing
times = pd.read_csv('timing_train.csv', sep=',', header=None)
times = times.values

# Read in 3C waveforms (6 different types)
waveformZ = pd.read_csv('Zwaves_train.csv', sep=',', header=None)
waveformZ = waveformZ.values
waveformh1 = pd.read_csv('h1waves_train.csv', sep=',', header=None)
waveformh1 = waveformh1.values
waveformh2 = pd.read_csv('h2waves_train.csv', sep=',', header=None)
waveformh2 = waveformh2.values

# Read in estimated p-arrivals for waveforms
parrivals = pd.read_csv('parrivals_train.csv', sep=',', header=None)
parrivals = parrivals.values

# Read in estimated s-arrivals for waveforms
sarrivals = pd.read_csv('sarrivals_train.csv', sep=',', header=None)
sarrivals = sarrivals.values

# Produce synthetic dataset
###

n = 10000  # number of synthetic waves to produce
noise_levels = [0.05, 0.1, 0.3, 0.5]  # noise percentages to add
lengths = 3744


# Training datasets
X_train = np.zeros((n,lengths,3),dtype=np.float16)
Y_train = np.zeros((n,lengths,2),dtype=np.float16)
timings = np.zeros((n,lengths,1),dtype=np.float16)

for i in range(0,n):
	# Pull random wave and noise level
	randwave = random.randint(0,5)
	randnoise = random.randint(0,3)

	# Add gaussian noise to wave
	noise = np.random.normal(0,1,lengths)
	waveZ = waveformZ[0:lengths,randwave]+noise_levels[randnoise]*noise*max(waveformZ[0:lengths,randwave])
	waveh1 = waveformh1[0:lengths,randwave]+noise_levels[randnoise]*noise*max(waveformh1[0:lengths,randwave])
	waveh2 = waveformh2[0:lengths,randwave]+noise_levels[randnoise]*noise*max(waveformh2[0:lengths,randwave])
	time = times

	# Pull arrival indexes
	std = 10
	randwigglep = random.randint(-std,std)
	randwiggles = random.randint(-std,std)
	indexp = find_nearest(time,parrivals[0,randwave])
	indexp = indexp + randwigglep
	x = np.linspace(0,lengths,lengths,dtype=np.float16)
	pgauss = np.exp(-np.power(x - indexp, 2.) / (2 * np.power(std, 2.)))
	indexs = find_nearest(time,sarrivals[0,randwave])
	indexs = indexs + randwiggles
	sgauss = np.exp(-np.power(x - indexs, 2.) / (2 * np.power(std, 2.)))

	# Add to training dataset
	Y_train[i] = np.stack([pgauss,sgauss],axis=-1)
	X_train[i] = np.stack([waveZ/max(waveZ), waveh1/max(waveh1), waveh2/max(waveh2)], axis=-1)
	timings[i] = time[0:lengths]


# Read in testing datasets
# Read in timing
times_test = pd.read_csv('timing_test.csv', sep=',', header=None)
times_test = times_test.values

# Read in 3C waveforms (6 different types)
waveformZ_test = pd.read_csv('Zwaves_test.csv', sep=',', header=None)
waveformZ_test = waveformZ_test.values
waveformh1_test = pd.read_csv('h1waves_test.csv', sep=',', header=None)
waveformh1_test = waveformh1_test.values
waveformh2_test = pd.read_csv('h2waves_test.csv', sep=',', header=None)
waveformh2_test = waveformh2_test.values

# Read in estimated p-arrivals for waveforms
parrivals_test = pd.read_csv('parrivals_test.csv', sep=',', header=None)
parrivals_test = parrivals_test.values

# Read in estimated s-arrivals for waveforms
sarrivals_test = pd.read_csv('sarrivals_test.csv', sep=',', header=None)
sarrivals_test = sarrivals_test.values


# Test dataset
m = 24
X_test = np.zeros((m,lengths,3),dtype=np.float16)
timings_test = np.zeros((m,lengths,1),dtype=np.float16)
arr = np.zeros((m,2),dtype=np.float16)

for j in range(0,m):
	# Pull random wave and noise level
	#randwave = random.randint(0,23)
	randnoise = random.randint(0,3) 
	if j<6:
		time = times_test[:,0]
	elif j<12:
		time = times_test[:,1]
	elif j<18:
		time = times_test[:,2]
	else:
		time = times_test[:,3]

	# Add gaussian noise to wave
	noise = np.random.normal(0,1,lengths)
	waveZ_test = waveformZ_test[0:lengths,j]+noise_levels[randnoise]*noise*max(waveformZ_test[0:lengths,j])
	waveh1_test = waveformh1_test[0:lengths,j]+noise_levels[randnoise]*noise*max(waveformh1_test[0:lengths,j])
	waveh2_test = waveformh2_test[0:lengths,j]+noise_levels[randnoise]*noise*max(waveformh2_test[0:lengths,j])

	# Add to test dataset
	X_test[j] = np.stack([waveZ_test/max(waveZ_test), waveh1_test/max(waveh1_test), waveh2_test/max(waveh2_test)], axis=-1)
	timings_test[j] = np.reshape(time[0:lengths],(lengths,1))
	arr[j][0] = parrivals_test[0][j]
	arr[j][1] = sarrivals_test[0][j]


###############################
# Build UNET
###############################

# Set data parameters
data_height = lengths


warnings.filterwarnings('ignore',category=UserWarning,module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


## Build and train neural network loosely based on U-net 
#  (modified from Amdal-Saevik and from Jocic Marko code available on kaggle and github respectively)

# Build u-net model
inputs = Input((data_height,3))
#s = Lambda(lambda x: x / 255) (inputs)

conv1 = Conv1D(32, 3, activation='elu',kernel_initializer='he_normal', padding='same')(inputs)
conv1 = Dropout(0.1) (conv1)
conv1 = Conv1D(32, 3, activation='elu',kernel_initializer='he_normal', padding='same')(conv1)
pool1 = MaxPooling1D(pool_size=2)(conv1)

conv2 = Conv1D(64, 3, activation='elu',kernel_initializer='he_normal', padding='same')(pool1)
conv2 = Dropout(0.1) (conv2)
conv2 = Conv1D(64, 3, activation='elu', kernel_initializer='he_normal',padding='same')(conv2)
pool2 = MaxPooling1D(pool_size=2)(conv2)

conv3 = Conv1D(128, 3, activation='elu',kernel_initializer='he_normal', padding='same')(pool2)
conv3 = Dropout(0.2) (conv3)
conv3 = Conv1D(128, 3, activation='elu', kernel_initializer='he_normal',padding='same')(conv3)
pool3 = MaxPooling1D(pool_size=2)(conv3)

conv4 = Conv1D(256, 3, activation='elu', kernel_initializer='he_normal',padding='same')(pool3)
conv4 = Dropout(0.2) (conv4)
conv4 = Conv1D(256, 3, activation='elu',kernel_initializer='he_normal', padding='same')(conv4)
pool4 = MaxPooling1D(pool_size=2)(conv4)

conv5 = Conv1D(512, 3, activation='elu',kernel_initializer='he_normal', padding='same')(pool4)
conv5 = Dropout(0.3) (conv5)
conv5 = Conv1D(512, 3, activation='elu', kernel_initializer='he_normal',padding='same')(conv5)

#up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
up6 = Conv1DTranspose(conv5,256,2,strides=2,padding='same')
up6 = concatenate([up6,conv4],axis=-1)
conv6 = Conv1D(256, 3, activation='elu',kernel_initializer='he_normal', padding='same')(up6)
conv6 = Dropout(0.2) (conv6)
conv6 = Conv1D(256, 3, activation='elu', kernel_initializer='he_normal',padding='same')(conv6)

#up7 = concatenate([conv6, conv3], axis=-1)
up7 = Conv1DTranspose(conv6,128,2,strides=2,padding='same')
up7 = concatenate([up7,conv3],axis=-1)
conv7 = Conv1D(128, 3, activation='elu', kernel_initializer='he_normal',padding='same')(up7)
conv7 = Dropout(0.2) (conv7)
conv7 = Conv1D(128, 3, activation='elu', kernel_initializer='he_normal',padding='same')(conv7)

#up8 = concatenate([conv7, conv2], axis=-1)
up8 = Conv1DTranspose(conv7,64,2,strides=2,padding='same')
up8 = concatenate([up8,conv2],axis=-1)
conv8 = Conv1D(64, 3, activation='elu',kernel_initializer='he_normal', padding='same')(up8)
conv8 = Dropout(0.1) (conv8)
conv8 = Conv1D(64, 3, activation='elu', kernel_initializer='he_normal',padding='same')(conv8)

#up9 = concatenate([conv8, conv1], axis=-1)
up9 = Conv1DTranspose(conv8,32,2,strides=2,padding='same')
up9 = concatenate([up9,conv1],axis=-1)
conv9 = Conv1D(32, 3, activation='elu', kernel_initializer='he_normal',padding='same')(up9)
conv9 = Dropout(0.1) (conv9)
conv9 = Conv1D(32, 3, activation='elu',kernel_initializer='he_normal', padding='same')(conv9)

conv10 = Conv1D(2, 1, activation='sigmoid')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()


###############################
# Test and Validate Models
###############################

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('weights.h5',monitor='val_loss', verbose=1, save_best_only=True)
model.fit(X_train, Y_train, validation_split=0.1, batch_size=50 ,verbose=1, shuffle=True, epochs=50 ,callbacks=[checkpointer,earlystopper])
model.load_weights('weights.h5')

# Run model
predicted_model = model.predict(X_test, verbose=1)
np.save('predicted_model.npy', predicted_model)

"""
# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5')
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),(3744, 1), mode='constant', preserve_range=True))

"""

for i in range(0,np.shape(predicted_model)[0]):
	wave = X_test[i]
	result = predicted_model[i]
	time = timings_test[i]

	plt.figure()
	plt.subplot(311)
	plt.plot(time,wave[:,1])
	plt.plot(time,result[:,0],color='k',linestyle='dashed')
	plt.plot(time,result[:,1],color='g',linestyle='dashed')
	plt.axvline(x=arr[i,0],color='k')
	plt.axvline(x=arr[i,1],color='g')
	plt.yticks([])
	plt.title('Testing Data Wave %i' % i)
	plt.ylabel('H1 Component')

	plt.subplot(312)
	plt.plot(time,wave[:,2])
	plt.plot(time,result[:,0],color='k',linestyle='dashed')
	plt.plot(time,result[:,1],color='g',linestyle='dashed')
	plt.axvline(x=arr[i,0],color='k')
	plt.axvline(x=arr[i,1],color='g')
	plt.yticks([])
	plt.ylabel('H2 Component')

	plt.subplot(313)
	plt.plot(time,wave[:,0])
	plt.plot(time,result[:,0],color='k',linestyle='dashed')
	plt.plot(time,result[:,1],color='g',linestyle='dashed')
	plt.axvline(x=arr[i,0],color='k')
	plt.axvline(x=arr[i,1],color='g')
	plt.yticks([])
	plt.xlabel('Times (s)')
	plt.ylabel('Z Component')

	plt.savefig('results/wave%i.pdf'%(i))
	plt.close()


################################
# Plotting Results
################################


# Plotting Example Waveforms
plt.figure()
plt.subplot(311)
plt.plot(times,waveformh1[:,0])
plt.axvline(x=parrivals[0,0],color='k',linewidth=2)
plt.axvline(x=sarrivals[0,0],color='k',linewidth=2)
plt.yticks([])
plt.title('Example Input Waveform')
plt.ylabel('H1 Component')

plt.subplot(312)
plt.plot(times,waveformh2[:,0])
plt.axvline(x=parrivals[0,0],color='k',linewidth=2)
plt.axvline(x=sarrivals[0,0],color='k',linewidth=2)
plt.yticks([])
plt.ylabel('H2 Component')

plt.subplot(313)
plt.plot(times,waveformZ[:,0])
plt.axvline(x=parrivals[0,0],color='k',linewidth=2)
plt.axvline(x=sarrivals[0,0],color='k',linewidth=2)
plt.yticks([])
plt.xlabel('Times (s)')
plt.ylabel('Z Component')

plt.show()
plt.savefig('figures/ex_wave.pdf')
plt.close()

