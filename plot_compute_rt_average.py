# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.realtime import RtEpochs, MockRtClient

print(__doc__)

# Fiff file to simulate the realtime client
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# select gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=True, exclude=raw.info['bads'])

# select the left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5

# create the mock-client object
rt_client = MockRtClient(raw)

# create the real-time epochs object
rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                     decim=1, reject=dict(grad=4000e-13, eog=150e-6))

# start the acquisition
rt_epochs.start()

# send raw buffers
rt_client.send_data(rt_epochs, picks, tmin=0, tmax=150, buffer_size=1000)
for ii, ev in enumerate(rt_epochs.iter_evoked()):
    print("Just got epoch %d" % (ii + 1))
    ev.pick_types(meg=True, eog=False)  # leave out the eog channel
    if ii == 0:
        evoked = ev
    else:
        evoked = mne.combine_evoked([evoked, ev], weights='nave')
    plt.clf()  # clear canvas
    evoked.plot(axes=plt.gca())  # plot on current figure
    plt.pause(0.05)
