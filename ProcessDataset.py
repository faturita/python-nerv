# coding: latin-1

import scipy.io
import numpy as np
mat = scipy.io.loadmat('/Users/rramele/GoogleDrive/BCI.Dataset/008-2014/A01.mat')

mat['data'][0][0][0]


# Labels
mat['data'][0][0][0][0][0]

# Data points
mat['data'][0][0][1]

# Targets / No tagets
mat['data'][0][0][2]

# Data point zero for the eight channels.
mat['data'][0][0][1][0]


import matplotlib.pyplot as plt

data=mat['data'][0][0][1]
cz = data[:,1]

#plt.plot(cz)
#plt.ylabel('First Channel')
#plt.show()

print cz.shape

cz = data[0:347000,:]

from scipy import stats
cz = stats.zscore(cz)

ccz = np.reshape( cz, [1000,347000/1000,8])

responses = ccz

#responses = np.concatenate( (np.ones(50),(np.ones(50)+1) )  )

type = np.concatenate ( ([True]*50,[False]*50) )

sampling_rate = 256

response_window = [50, 1000]

decimation_frequency = 32

from swlda import swlda

channels, weights = swlda(responses, type, sampling_rate, response_window, decimation_frequency)

print (channels)

print (weights.shape)

# swlda(responses, type, sampling_rate, response_window, decimation_frequency,
#     max_model_features = 60, penter = 0.1, premove = 0.15):
#     ``responses'' must be a (trials x samples x channels) array containing
#     responses to a stimulus.
#     ``type'' must be a one-dimensional array of bools of length trials.
#     ``sampling_rate'' is the sampling rate of the data.
#     ``response_window'' is of the form [begin, end] in milliseconds.
#     ``decimation_frequency'' is the frequency at which to resample.
