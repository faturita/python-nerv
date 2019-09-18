# coding: latin-1

import mne
mne.set_log_level('WARNING')

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import Bbox

import scipy.io
import numpy as np
mat = scipy.io.loadmat('/Users/rramele/GoogleDrive/BCI.Dataset/008-2014/A03.mat')

mat = scipy.io.loadmat('/Users/rramele/work/EEGWave/signals/p300-subject-drug-21.mat')

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()

    items = ax.get_xticklabels() + ax.get_yticklabels()
    #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def savesubfigure(a,filename,xposition):
    allaxes = a.get_axes()
    subfig = allaxes[0]

    # Save just the portion _inside_ the second axis's boundaries
    extent = full_extent(subfig).transformed(a.dpi_scale_trans.inverted())

    for item in ([subfig.title, subfig.xaxis.label, subfig.yaxis.label] +
                 subfig.get_xticklabels() + subfig.get_yticklabels()):
        item.set_fontsize(14)

    if (len(xposition)!=0):
        for xc in xposition:
            subfig.axvline(x=xc, color='k', linestyle='-')

    # Alternatively,
    # extent = ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    a.savefig(filename, bbox_inches=extent)



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

#eeg_mne.plot_psd()

eeg_mne.filter(1,20)

#eeg_mne.plot_psd()

ch_names_events = ch_names + ['S']+ ['L']
ch_types_events = ch_types + ['misc'] + ['misc']

t_stim = mat['data'][0][0][2]
t_type = mat['data'][0][0][1]*3

signal_events = np.concatenate([signal, t_stim, t_type],1)

info_events = mne.create_info(ch_names_events,250, ch_types_events)

eeg_events = mne.io.RawArray(signal_events.T, info_events)

a = eeg_events.plot(show_options=True,title='',start=28,duration=10,n_channels=10, scalings='auto')
#savesubfigure(a,'singlegain.eps')
#plt.plot([30, -30000], [30, 200000], 'k-', lw=5)
#plt.plot([-30000, 30], [300000, 30], 'k-', lw=5)

savesubfigure(a,'singlegain.eps',[31.4,31.65])
eeg_mne.plot_psd()
