from sklearn.preprocessing import scale
from scipy import signal
from mne.decoding import UnsupervisedSpatialFilter
from mne.preprocessing import ICA
from sklearn.decomposition import PCA, FastICA
from mne.decoding import CSP
from pylab import *
from mne.preprocessing import ICA, create_ecg_epochs
from numpy.testing import assert_array_equal
from mne.io import concatenate_raws, read_raw_edf
from numpy.testing import assert_array_equal
#from mne_bids.copyfiles import copyfile_brainvision

import numpy as np
import scipy.io as sio
#import sympy
import matplotlib.pyplot as plt
import mne
import os
import time
import os.path as op
import pandas as pd
import pyedflib

#from mne_bids.copyfiles import copyfile_brainvision
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import csv


path_open='OpenBCI-BDF-2019-11-26_16-07-54.bdf'
#pathbdf_open='C:/Users/Nicola/Documents/Datos/OpenBCI_Giuli/OpenBCI-BDF-2019-11-26_16-17-14.bdf'

raw=mne.io.read_raw_bdf(path_open)
#data=raw.get_data()     

data, times = raw[:, :]  #'numpy.ndarray'>
info =raw.info
print(info)
time_secs = raw.times
channels = raw.ch_names

annot = mne.read_annotations(path_open)
anotaciones=raw.set_annotations(annot)
#NO FUNCIONA -->anot=mne.events_from_annotations(raw)
#                    raw.set_annotations(anot)

print('Anotaciones',anotaciones)

#events, event_ids = mne.events_from_annotations(raw) #events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
#print('Eventos', events, event_ids)


#No funciona por que no encuentra argumentos
# epochs = mne.Epochs(raw, events, event_ids, preload=True)

print(data)
print(times)
print('tiempo minimo:',times.min(),'tiempo maximo:',times.max())

print('Raw')
raw.plot(n_channels=1,scalings=dict(eeg=20e-2))
plt.show()

# NO FUNCIONA -->raw.plot()

