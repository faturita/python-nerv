# coding: latin-1

import mne
mne.set_log_level('WARNING')

import scipy.io
import numpy as np

mat = scipy.io.loadmat('/Users/rramele/work/EEGWave/signals/p300-subject-drug-21.mat')


# dtype=[('X', 'O'), ('y', 'O'), ('y_stim', 'O'), ('trial', 'O'), ('flash', 'O')])
mat['data'][0][0][0]

# Data points
mat['data'][0][0][0]

# Targets / No tagets
mat['data'][0][0][1]

# Stims/ No Stims
mat['data'][0][0][2]

# Trials
mat['data'][0][0][3]

# Flash matrix
mat['data'][0][0][4]

# Data point zero for the eight channels.  Should be in V.
signal = mat['data'][0][0][0] * pow(10,6)

print (signal.shape)

ch_names=[ 'Fz'  ,  'Cz',    'P3' ,   'Pz'  ,  'P4'  ,  'PO7'   , 'PO8'   , 'Oz']
ch_types= ['eeg'] * signal.shape[1]

info = mne.create_info(ch_names, 250, ch_types=ch_types)

eeg_mne = mne.io.array.RawArray(signal.T, info)

eeg_mne.plot_psd()

eeg_mne.filter(1,20)

eeg_mne.plot_psd()

ch_names_events = ch_names + ['t_stim']+ ['t_type']
ch_types_events = ch_types + ['misc'] + ['misc']

t_stim = mat['data'][0][0][2]
t_type = mat['data'][0][0][1]

signal_events = np.concatenate([signal, t_stim, t_type],1)

info_events = mne.create_info(ch_names_events,250, ch_types_events)

eeg_events = mne.io.RawArray(signal_events.T, info_events)

a = eeg_events.plot(show_options=False,title='EEG signals',start=28,duration=10,n_channels=10, scalings='auto')


input('Press a key to continue...')