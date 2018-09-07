# coding: latin-1

import mne
mne.set_log_level('WARNING')

import scipy.io
import numpy as np
mat = scipy.io.loadmat('/Users/rramele/GoogleDrive/BCI.Dataset/008-2014/A03.mat')

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

print signal.shape

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

a = eeg_events.plot(start=28,duration=10,n_channels=10, scalings='auto')



event_times = mne.find_events(eeg_events, stim_channel='t_stim')

print('Found %s events, first five:' % len(event_times))
print(event_times[:5])


event_times = mne.find_events(eeg_events, stim_channel='t_type')

tmin = 0
tmax = 0.8

epochs = mne.Epochs(eeg_mne, event_times, { 'second':2 }, tmin, tmax)


print ('Hits:')
print ('Epochs x channels x time')
print epochs.get_data().shape

evoked = epochs.average()
evoked.plot()

event_times = mne.find_events(eeg_events, stim_channel='t_type')

tmin = 0
tmax = 0.8

epochs = mne.Epochs(eeg_mne, event_times, {'first':1}, tmin, tmax)

print ('Nohits:')
print ('Epochs x channels x time')
print epochs.get_data().shape

evoked = epochs.average()
evoked.plot()

eeg_mne.plot(start=28, duration=10,scalings='auto',n_channels=8,events=event_times)

montage = mne.channels.read_montage('standard_1020')

eeg_mne.set_montage(montage)

eeg_mne.plot_sensors()

eeg_mne.plot(start=28,duration=10,scalings='auto',block=True)
a.savefig("singlegain.eps")
