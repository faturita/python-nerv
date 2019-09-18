# coding: latin-1

import mne
mne.set_log_level('WARNING')

import scipy.io
import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import Bbox
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

def savesubfigure(a,filename):
    allaxes = a.get_axes()
    subfig = allaxes[0]

    # Save just the portion _inside_ the second axis's boundaries
    extent = full_extent(subfig).transformed(a.dpi_scale_trans.inverted())

    for item in ([subfig.title, subfig.xaxis.label, subfig.yaxis.label] +
                 subfig.get_xticklabels() + subfig.get_yticklabels()):
        item.set_fontsize(14)

    # Alternatively,
    # extent = ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    a.savefig(filename, bbox_inches=extent)


# Python3 implementation of the approach 
from math import sqrt 

# Function to find the circle on 
# which the given three points lie 
def findCircle(x1, y1, x2, y2, x3, y3): 
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2 
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2 
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    if ((y31) * (x12) - (y21) * (x13) == 0):
        f = 1
    else:
        f = (((sx13) * (x12) + (sy13) *
            (x12) + (sx21) * (x13) +
            (sy21) * (x13)) // (2 *
            ((y31) * (x12) - (y21) * (x13))))
            
    if ((x31) * (y12) - (x21) * (y13)==0):
        g = 1
    else:
        g = (((sx13) * (y12) + (sy13) * (y12) +
            (sx21) * (y13) + (sy21) * (y13)) //
            (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
        2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0 
    # where centre is (h = -g, k = -f) and 
    # radius r as r^2 = h^2 + k^2 - c 
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c

    # r is the radius 
    r = round(sqrt(sqr_of_r), 5)

    # if (((y3+y1)/2.0)>y2):
    #     r=r*(1)
    # else:
    #     r=r*(-1)


    print("Centre = (", h, ", ", k, ")")
    print("Radius = ", r)

    return [(h,k),r]

mat = scipy.io.loadmat('/Users/rramele/GoogleDrive/BCI.Dataset/008-2014/A03.mat')

mat = scipy.io.loadmat('/Users/rramele/work/EEGWave/signals/p300-subject-drug-21.mat')

st=131

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

#eg_mne.plot_psd()

eeg_mne.filter(1,20)

#eeg_mne.plot_psd()

ch_names_events = ch_names + ['S']+ ['L']
ch_types_events = ch_types + ['misc'] + ['misc'] 

t_stim = mat['data'][0][0][2]
t_type = mat['data'][0][0][1]*3


radiosignal = np.ones((signal.shape[0],1))
row=100

delta = int(3/3)
for i in range(delta,radiosignal.shape[0]-delta):
    x1 = row-delta ; y1 = signal[i-delta,1]
    x2 = row       ; y2 = signal[i,1]
    x3 = row+delta ; y3 = signal[i+delta,1]

    [(rows, cols),r] = findCircle(x1, y1, x2, y2, x3, y3)

    r = r

    radiosignal[i,0] = r

signal_events = np.concatenate([signal, t_stim, t_type],1)
info_events = mne.create_info(ch_names_events,250, ch_types_events)
eeg_events = mne.io.RawArray(signal_events.T, info_events)

sgn = signal

circular_events = np.concatenate([sgn, t_stim, t_type, radiosignal],1)
circular_info = mne.create_info(ch_names + ['S']+ ['L'] + ['R'],250, ch_types + ['misc'] + ['misc'] + ['misc'] )
circular = mne.io.RawArray(circular_events.T, circular_info)


#a = eeg_events.plot(show_options=False,title='EEG signals',start=28,duration=10,n_channels=10, scalings='auto')
#a = eeg_events.plot(show_options=False,title='EEG signals',start=28,duration=10,n_channels=2, order=[1,9], scalings='auto')


#eeg_events.plot(show_options=False,title='Zoom',start=31,duration=1,n_channels=2, order=[1,9], scalings='auto')
a=circular.plot  (show_options=False,title='Drug',start=st,duration=1,n_channels=2, order=[1,10], scalings='auto')
savesubfigure(a,'images/1.eps')





mat = scipy.io.loadmat('/Users/rramele/work/EEGWave/signals/p300-subject-21.mat')


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

#eg_mne.plot_psd()

eeg_mne.filter(1,20)

#eeg_mne.plot_psd()

ch_names_events = ch_names + ['S']+ ['L']
ch_types_events = ch_types + ['misc'] + ['misc'] 

t_stim = mat['data'][0][0][2]
t_type = mat['data'][0][0][1]*3


radiosignal = np.ones((signal.shape[0],1))
row=100

delta = int(3/3)
for i in range(delta,radiosignal.shape[0]-delta):
    x1 = row-delta ; y1 = signal[i-delta,1]
    x2 = row       ; y2 = signal[i,1]
    x3 = row+delta ; y3 = signal[i+delta,1]

    [(rows, cols),r] = findCircle(x1, y1, x2, y2, x3, y3)

    r = r

    radiosignal[i,0] = r

signal_events = np.concatenate([signal, t_stim, t_type],1)
info_events = mne.create_info(ch_names_events,250, ch_types_events)
eeg_events = mne.io.RawArray(signal_events.T, info_events)

sgn = signal

circular_events = np.concatenate([sgn, t_stim, t_type, radiosignal],1)
circular_info = mne.create_info(ch_names + ['S']+ ['L'] + ['R'],250, ch_types + ['misc'] + ['misc'] + ['misc'] )
circular = mne.io.RawArray(circular_events.T, circular_info)


#a = eeg_events.plot(show_options=False,title='EEG signals',start=28,duration=10,n_channels=10, scalings='auto')
#a = eeg_events.plot(show_options=False,title='EEG signals',start=28,duration=10,n_channels=2, order=[1,9], scalings='auto')

# 131  and 31
a=eeg_events.plot(show_options=False,title='Clear',start=st,duration=1,n_channels=2, order=[1,9], scalings='auto')

savesubfigure(a,'images/3.eps')

a = circular.plot  (show_options=False,title='Clear',start=st,duration=1,n_channels=2, order=[1,10], scalings='auto')
savesubfigure(a,'images/2.eps')

input("Press Enter to continue...")