# In[1]:
import mne
mne.set_log_level('WARNING')

import scipy.io
import numpy as np

# First load the template.  This is the signal that will be used to DRUG the basal EEG stream.
mat = scipy.io.loadmat('./dataset/ERPTemplate.mat')

routput = mat['routput']

# In this ERPTemplate, there are two different template signals that are good.
erptemplate1 = routput[0][7][0][1][0][0][0][7] 
erptemplate2 = routput[0][7][0][1][0][0][0][0] 

# The original ERPTemplate dataset has a sampling frequency of 256 so I need to perform a small downsampling to 250 Hz
erptemplate1 = np.delete( erptemplate1, range(0,256,43),0)
erptemplate2 = np.delete( erptemplate2, range(0,256,43),0)

# Use this for testing  (get a ZERO signal)
#erptemplate1 = np.zeros((250,8))

# Randomize amplitude and jitter.
# Find the right locations where this should be inserted in the stream.
# Insert the signal mantaining the continiuity of the EEG.
def DrugSignal(signal, t_flash):
    '''
    Randomize amplitude and jitter
    Find the right locations where this should be inserted in the stream
    Insert the template mantaining the continuity and physiological meaning of the EEG
    '''
    for i in range(0,4200):
        if (t_flash[i,3]==2):
            signal[t_flash[i,0]-1:t_flash[i,0]+250-1,:] = erptemplate1

    return signal


# Now load the basal EEG stream
mat = scipy.io.loadmat('./dataset/p300-subject-25.mat')
#mat = scipy.io.loadmat('./dataset/p300-subject-26.mat')

# In[1]:

# coding: latin-1
# Data point zero for the eight channels.  Should be in V.
signal = mat['data'][0][0][0] * pow(10,6)

# Trials
t_trials = mat['data'][0][0][3]

# Flash matrix
t_flash = mat['data'][0][0][4]

#signal = DrugSignal(signal, t_flash)

t_stim = mat['data'][0][0][2]
t_type = mat['data'][0][0][1]


ch_names=[ 'Fz'  ,  'Cz',    'P3' ,   'Pz'  ,  'P4'  ,  'PO7'   , 'PO8'   , 'Oz']
ch_types= ['eeg'] * signal.shape[1]

ch_names_events = ch_names + ['t_stim']+ ['t_type']
ch_types_events = ch_types + ['misc'] + ['misc']

#info = mne.create_info(ch_names, 250, ch_types=ch_types)
#eeg_mne = mne.io.array.RawArray(signal.T, info)

signal_events = np.concatenate([signal, t_stim, t_type],1)
info_events = mne.create_info(ch_names_events,250, ch_types_events)
eeg_events = mne.io.RawArray(signal_events.T, info_events)

# Do some basic signal processing (1-20 band pass filter)
fig=eeg_events.plot_psd()

eeg_events.filter(1,20)

fig=eeg_events.plot_psd()

eeg_events.plot(scalings='auto',n_channels=8,events=signal_events,block=True)

def getstims(eeg_mne, eeg_events):
    '''
    Get the stimulations.  These are the FLASHINGS of rows and columns.
    '''
    tmin = 0
    tmax = 0.8
    reject = None
    event_times = mne.find_events(eeg_events, stim_channel='t_stim')
    event_id = {'Row1':1,'Row2':2,'Row3':3,'Row4':4,'Row5':5,'Row6':6,'Col1':7,'Col2':8,'Col3':9,'Col4':10,'Col5':11,'Col6':12}


    epochs = mne.Epochs(eeg_mne, event_times, event_id, tmin, tmax, proj=False,
                    baseline=None, reject=reject, preload=True,
                    verbose=True)


    stims = event_times[:,-1]

    return [epochs,stims]

stimepochs, stims = getstims(eeg_events, eeg_events)

def getlabels(eeg_mne, eeg_events):
    '''
    Get the hit/no hits labels.  These are the FLASHINGS of rows and columns but selected if they are the ones that will
    trigger the P300 response or not.
    '''
    event_id = { 'first':1, 'second':2 }
    #baseline = (0.0, 0.2)
    #reject = {'eeg': 70 * pow(10,6)}
    tmin = 0
    tmax = 0.8
    reject = None
    event_times = mne.find_events(eeg_events, stim_channel='t_type')
    epochs = mne.Epochs(eeg_mne, event_times, event_id, tmin, tmax, proj=False,
                    baseline=None, reject=reject, preload=True,
                    verbose=True)
    labels = epochs.events[:, -1]
    return [epochs, labels]

epochs, labels = getlabels(eeg_events, eeg_events)

# Downsample the original FS=250 Hz signal to >>> 20 Hz
epochs.resample(20, npad="auto")
stimepochs.resample(20, npad="auto")
repetitions=120

# %%
# This is Single Flashing Classification attempt.
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# import a linear classifier from mne.decoding
from mne.decoding import LinearModel

import matplotlib.pyplot as plt

clf = LogisticRegression(solver='lbfgs')
scaler = StandardScaler()

# create a linear model with LogisticRegression
model = LinearModel(clf)

# Get the epoched data (get only the data columns)
eeg_data = epochs.get_data().reshape(len(labels), -1)
eeg_data = eeg_data[:,0:epochs.get_data().shape[2]*1]
#eeg_data[labels==2] = erptemplate1[:201,0]
#eeg_data[labels==1] = erptemplate1[:201,0]

#eeg_data[labels==2] = np.zeros((eeg_data.shape[1],))
#eeg_data[labels==1] = np.ones((eeg_data.shape[1],))

#labels = np.random.permutation(labels)

# fit the classifier on MEG data
X = scaler.fit_transform(eeg_data)

model.fit(X[0:2800], labels[0:2800])

preds = model.predict(X[2800:])

# Classification report
target_names = ['nohit', 'hit']

report = classification_report(labels[2800:], preds, target_names=target_names)
print(report)

cm = confusion_matrix(labels[2800:], preds)
print (cm)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
acc=(cm[0,0]+cm[1,1])*1.0/(np.sum(cm))

# %%
#import matplotlib.pyplot as plt
#for i in range(200,220):
#    plt.figure(figsize=(9, 3))
#    plt.plot(eeg_data[labels==1][i])
#    plt.show()


# In[1]:   Average classification x trial (unbalanced)
# Este dataset es un dataset de calibración. El registro de EEG corresponde a un experimento del Speller
# de 35 letras (7 palabras de 5 letras).   Cada una de las letras consiste en 10 repeticiones de la intensificación 
# de los 12 estímulos distintos, siendo cada estímulo el FLASHING de una de las 6 filas o 6 columnas.
# Cada vez que se repite, se hace una permutación de los 12.
# En cada una de esos 12 estímulos, dos, uno correspondiente a una fila y a una columna, corresponden 
# a la letra que la persona está prestando atención y la idea es que el sistema descubra que letra es.

# Primero tengo que agarrar la lista de labels y asignar a los 420 (35x12)
# el label que le corresponde a cada uno.  Es decir de los primeros 12, 10
# son no hits y 2 hits.

# hlbls tiene pares (r,c) que representan la fila y la columna donde está la letra
# que la persona tiene que elegir. 
hlbls = []
hpreds = []
classlabels=np.asarray([])
for trial in range(0,35):
    a=np.zeros((12*10,2))
    a[:,0] = stims[0+120*trial:0+120*trial+120]
    a[:,1] = labels[0+120*trial:0+120*trial+120]

    b=np.zeros((12,1))

    for i in range(1,13):
        b[i-1] = np.unique(a[a[:,0]==i,1])

    for i in range(0,6):
        if (b[i]==2):
            r = i+1

    for i in range(6,12):
        if (b[i]==2):
            c = i+1

    classlabels = np.append( classlabels, b )

    assert (r!=0 and c!=0), 'Error %d,%d' % (r,c) 
    hlbls.append( (r,c) )

# In[1]:  
def SpellMeLetter(row, col):
    spellermatrix = [ ['A','B','C','D','E','F'],
                    [ 'G','H','I','J','K','L'],
                [ 'M','N','O','P','Q','R'],
                [ 'S','T','U','V','W','X'],
                [ 'Y','Z','1','2','3','4'],
                [ '5','6','7','8','9','_'] ]

    return spellermatrix[row-1][col-1-6]

# Esta es la frase de 7 palabras de 5 letras que la persona tiene que producir.
for i in range(0,35):
    print(SpellMeLetter(hlbls[i][0],hlbls[i][1]),end='')

print()
# In[1]: 
# Luego necesito calcular los 420 averaging (de repetitions)
# Finalmente aprendo con 180 y me fijo si predigo los 240
# De los 240 adivino 20 letras (de a pares) y con eso calculo la performance

def getaverageepoch(singleepoch):
    '''
    Build the epochs based on each stimulation (1-12), and put all the epochs togheter.
    '''
    for trial in range(0,35):
        epochstrial = singleepoch[0+repetitions*trial:repetitions*trial+repetitions]

        epochr1 = epochstrial['Row1']
        epochr2 = epochstrial['Row2']
        epochr3 = epochstrial['Row3']
        epochr4 = epochstrial['Row4']
        epochr5 = epochstrial['Row5']
        epochr6 = epochstrial['Row6']

        epochc1 = epochstrial['Col1']
        epochc2 = epochstrial['Col2']
        epochc3 = epochstrial['Col3']
        epochc4 = epochstrial['Col4']
        epochc5 = epochstrial['Col5']
        epochc6 = epochstrial['Col6']

        if (trial==0):
            epochs_data = np.array([epochr1.average().data])
        else:
            epochs_data = np.concatenate((epochs_data, [epochr1.average().data]), axis=0)

        epochs_data = np.concatenate((epochs_data, [epochr2.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochr3.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochr4.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochr5.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochr6.average().data]), axis=0)

        epochs_data = np.concatenate((epochs_data, [epochc1.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochc2.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochc3.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochc4.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochc5.average().data]), axis=0)
        epochs_data = np.concatenate((epochs_data, [epochc6.average().data]), axis=0)

    # There are 420 epochs, which correspond to 35 letters, in groups of 12.
    events=np.array([np.arange(420),np.zeros(420), classlabels])
    events = events.T
    events = events.astype(int)

    tmin = 0
    tmax = 0.8
    event_id = { 'first':1, 'second':2 }
    info = mne.create_info(ch_names, 250, ch_types=ch_types)
    custom_epochs = mne.EpochsArray(epochs_data, info, events, tmin, event_id) 

    return custom_epochs

# avepochs contains all the 420 averaged epochs, 7 letters of 5, with 12 each.
custom_epochs = getaverageepoch(stimepochs)


# In[1]: 
# Performs the final classification, the one that allows to produce the spelled letters.
print('Performance Classification of Averaged Epochs')
clf = LogisticRegression(solver='lbfgs')
scaler = StandardScaler()

# create a linear model with LogisticRegression
model = LinearModel(clf)

training = range(0,180)
test = range(180,420)

eeg_data = custom_epochs.get_data()

eeg_data = eeg_data.reshape(420, -1)
eeg_data = eeg_data[:,0:201]

#eeg_data[classlabels==2] = np.zeros((eeg_data.shape[1],))
#eeg_data[classlabels==1] = np.ones((eeg_data.shape[1],))

#X = scaler.fit_transform(eeg_data)

X = eeg_data

cf = clf.fit(X[training], classlabels[training])

classpreds = np.empty ((420,2))

classpreds[test,:] = clf.predict_proba(X[test])

hpreds = []

for trial in range(15,35):
    #print('Row')
    for i in range(0,6):
        preds = classpreds[trial*12+i]
        #print ( preds[1] )
        labels = classlabels[trial*12+i]

    #print (  np.argmin( classpreds[trial*12+0:trial*12+6]))
    r = np.argmax( classpreds[trial*12+0:trial*12+6,1])+1
    

    #print('Col')
    for i in range(6,12):
        preds = classpreds[trial*12+i]
        #print ( preds[1] )
        labels = classlabels[trial*12+i]

    #print (  np.argmin( classpreds[trial*12+6:trial*12+12]))
    c = np.argmax( classpreds[trial*12+6:trial*12+12,1])+1

    hpreds.append( (r,c) )

# In[1]: 
for i in range(15,35):
    print(SpellMeLetter(hlbls[i][0],hlbls[i][1]),end='')

print()

# In[1]: 
for i in range(15,35):
    print(SpellMeLetter(hpreds[i-15][0],hpreds[i-15][1]),end='')

print()


# %%
for i in range(0,12):
    plt.figure(figsize=(9, 3))
    plt.plot(eeg_data[i])
    plt.show()
# %%
