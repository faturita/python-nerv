# Original Authors: Hari Bharadwaj <hari@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#
# License: BSD (3-clause)

# %%
import numpy as np
from matplotlib import pyplot as plt

from mne import create_info, EpochsArray
from mne.baseline import rescale
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet)

import mne

print(__doc__)

sfreq = 1000.0              # Hz
n_epochs = 40

ch_names = ['Pz', 'Fz']
ch_types = ['eeg','eeg']
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

n_times = 1024              # Epoch length

seed = 42                   # I wonder why everybody uses 42 as seed. Is that the answer to something :)?
rng= np.random.RandomState(seed)
noise = rng.randn(n_epochs, len(ch_names), n_times)

# Add a 50 Hz sinusoidarl signal 
t = np.arange(n_times, dtype = np.float) / sfreq 
signal = np.sin( np.pi * 2. * 50. * t)          # 50 Hz signal
signal[np.logical_or(t < 0.45, t > 0.55)] = 0.
on_time = np.logical_and( t >= 0.45, t<= 0.55)
signal[on_time] *= np.hanning(on_time.sum())        # Ramping
data = noise + signal

reject = dict(eeg=4000)
events = np.empty((n_epochs, 3), dtype=int)
first_event_sample = 100
event_id = dict(sin50hz=1)
for k in range(n_epochs):
    events[k,:] = first_event_sample + k * n_times, 0, event_id['sin50hz']

epochs = EpochsArray(data=data, info=info, events=events, event_id=event_id, reject=reject)


# %%
data = np.reshape( data, (1024*40,2))
signal_events = np.concatenate( [ data],1)
eeg = mne.io.RawArray(signal_events.T, info)

# %%
eeg.plot(scalings='auto')


# %%

freqs = np.arange(5., 100., 3.)
vmin, vmax = -3.,3.

n_cycles = freqs / 2.
time_bandwidth = 2.0
power = tfr_multitaper(epochs, freqs = freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False)
power.plot([0], baseline=(0.,0.1), mode='mean', vmin=vmin, vmax=vmax, title='Sim: Least smoothing, most variance')

# %% Less frequency smoothing, more time smoothing
n_cycles = freqs  # Increase time-window length to 1 second.
time_bandwidth = 4.0  # Same frequency-smoothing as (1) 3 tapers.
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=vmin, vmax=vmax,
           title='Sim: Less frequency smoothing, more time smoothing')


# %% Less time smooting, more frequency smoothing.
n_cycles = freqs / 2.
time_bandwidth = 8.0  # Same time-smoothing as (1), 7 tapers.
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=vmin, vmax=vmax,
           title='Sim: Less time smoothing, more frequency smoothing')