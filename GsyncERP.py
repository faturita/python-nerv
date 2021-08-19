# In[1]:

# coding: latin-1

import mne
mne.set_log_level('WARNING')

import scipy.io
import numpy as np

mat = scipy.io.loadmat('/Users/rramele/./GoogleDrive/Data/P300/p300-subject-21.mat')
#mat = scipy.io.loadmat('/Users/rramele/work/gsync/.//data/p300-subject-24.mat')
mat = scipy.io.loadmat('/Users/rramele/work/gsync/.//data/p300-subject-25.mat')
#mat = scipy.io.loadmat('/Users/rramele/work/gsync/.//data/p300-subject-26.mat')

# In[1]:

# coding: latin-1
# Data point zero for the eight channels.  Should be in V.
signal = mat['data'][0][0][0] * pow(10,6)

# Trials
t_trials = mat['data'][0][0][3]

# Flash matrix
t_flash = mat['data'][0][0][4]


from DrugSignal import DrugSignal
signal = DrugSignal(signal, t_flash)

print (signal.shape)

ch_names=[ 'Fz'  ,  'Cz',    'P3' ,   'Pz'  ,  'P4'  ,  'PO7'   , 'PO8'   , 'Oz']
ch_types= ['eeg'] * signal.shape[1]

info = mne.create_info(ch_names, 250, ch_types=ch_types)

eeg_mne = mne.io.array.RawArray(signal.T, info)

fig=eeg_mne.plot_psd()

eeg_mne.filter(1,20)

fig=eeg_mne.plot_psd()

# In[1]:

ch_names_events = ch_names + ['t_stim']+ ['t_type']
ch_types_events = ch_types + ['misc'] + ['misc']

t_stim = mat['data'][0][0][2]
t_type = mat['data'][0][0][1]

# In[1]:

signal_events = np.concatenate([signal, t_stim, t_type],1)

info_events = mne.create_info(ch_names_events,250, ch_types_events)

eeg_events = mne.io.RawArray(signal_events.T, info_events)

#eeg_events.plot(n_channels=10, scalings='auto')

event_times = mne.find_events(eeg_events, consecutive=True, min_duration=0.0001, stim_channel='t_stim', shortest_event=1,verbose=True)

print('Found %s events, first five:' % len(event_times))
print(event_times[:5])

# In[1]:
np.unique(t_flash[:,0]).shape
assert  np.unique(t_flash[:,0]).shape[0] == 4200, 'Problem with experiment structure.  There aren''t enough events.'

# In[1]:

event_times = mne.find_events(eeg_events, stim_channel='t_type')

tmin = 0
tmax = 0.8

epochs = mne.Epochs(eeg_mne, event_times, { 'second':2 }, tmin, tmax, preload=True)

# In[1]:

print ('Hits:')
print ('Epochs x channels x time')
print (epochs.get_data().shape)

epochs.resample(20, npad="auto")
evoked = epochs.average()
evoked.plot()

# In[1]:
event_times = mne.find_events(eeg_events, stim_channel='t_type')

tmin = 0
tmax = 0.8

epochs = mne.Epochs(eeg_mne, event_times, {'first':1}, tmin, tmax, preload=True)
# In[1]:

print ('Nohits:')
print ('Epochs x channels x time')
print (epochs.get_data().shape)

epochs.resample(20, npad="auto")
evoked = epochs.average()
evoked.plot()


eeg_mne.plot(scalings='auto',n_channels=8,events=event_times,block=True)

# In[1]:
montage = mne.channels.read_montage('standard_1020')

eeg_mne.set_montage(montage)

eeg_mne.plot_sensors()


# In[1]:

event_id = { 'first':1, 'second':2 }
#baseline = (0.0, 0.2)
#reject = {'eeg': 70 * pow(10,6)}
reject = None
epochs = mne.Epochs(eeg_mne, event_times, event_id, tmin, tmax, proj=False,
                baseline=None, reject=reject, preload=True,
                verbose=True)



# In[1]:   Single trial classification
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Create classification pipeline
clf = make_pipeline(mne.preprocessing.Xdawn(n_components=3),
                    mne.decoding.Vectorizer(),
                    MinMaxScaler(),
                    LogisticRegression(penalty='l1')
                    )

labels = epochs.events[:, -1]
lbls = labels

# Cross validator
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

epochs.resample(20, npad="auto")

print ('Epochs x channels x time')
print (epochs.get_data().shape)


# In[1]:   Single trial classification

# Do cross-validation
preds = np.empty(len(labels))
for train, test in cv.split(epochs, labels):
    cf=clf.fit(epochs[train], labels[train])
    preds[test] = clf.predict(epochs[test])

prds = preds

# Classification report
target_names = ['nohit', 'hit']

report = classification_report(labels, preds, target_names=target_names)
print(report)

cm = confusion_matrix(labels, preds)
print (cm)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
acc=(cm[0,0]+cm[1,1])*1.0/(np.sum(cm))





# In[1]:   Average classification x trial (unbalanced)
globalavgacc=[]
print ('Averaged classification per trials (20 reps vs 100 reps)')

repetitions=120

# Extracting for each letter-trial the epochs for each class.
for trial in range(0,35):
    epochstrial = epochs[0+repetitions*trial:repetitions*trial+repetitions]

    epochstrial1 = epochstrial['first']
    epochstrial2 = epochstrial['second']

    print ('Epochs x channels x time')
    print (epochstrial.get_data().shape)

    if (trial==0):
        evoked_nohit = epochstrial1.average()
        epochs_data = np.array([evoked_nohit.data])
    else:
        epochs_data = np.concatenate((epochs_data, [epochstrial1.average().data]), axis=0)
    epochs_data = np.concatenate((epochs_data, [epochstrial2.average().data]), axis=0)


#nave = len(epochs_data)
#evokeds = mne.EvokedArray(evoked_data, info=info, tmin=-0.2,comment='Arbitrary', nave=nave)
labels = np.array([1,2]*35)

print ('Randomize values...')
#labels = np.random.randint(1,3,70)

events = np.asarray([np.arange(70)+1,np.zeros(70), np.array([1,2]*35) ])
events = events.T

events = events.astype(int)


events[:,2] = labels

# Cross validator
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

custom_epochs = mne.EpochsArray(epochs_data, info, events, tmin, event_id)

# Do cross-validation
preds = np.empty(len(labels))
for train, test in cv.split(custom_epochs, labels):
    cf=clf.fit(custom_epochs[train], labels[train])
    preds[test] = clf.predict(custom_epochs[test])


test = range(30,70)
cf = clf.fit(custom_epochs[0:30], labels[0:30])
preds[test] = clf.predict(custom_epochs[test])

preds = preds[test]
labels = labels[test]

# Classification report
target_names = ['nohit', 'hit']

report = classification_report(labels, preds, target_names=target_names)
print(report)

cm = confusion_matrix(labels, preds)
print (cm)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
acc=(cm[0,0]+cm[1,1])*1.0/(np.sum(cm))

print('Accuracy per letter trial:'+str(acc))

globalavgacc.append(acc)





# In[1]:   Average classification x trial (unbalanced)
globalperformance=[]

print ('Averaged classification per row/column')

event_times = mne.find_events(eeg_events, stim_channel='t_stim')
event_id = {'Row1':1,'Row2':2,'Row3':3,'Row4':4,'Row5':5,'Row6':6,'Col1':7,'Col2':8,'Col3':9,'Col4':10,'Col5':11,'Col6':12}


epochs = mne.Epochs(eeg_mne, event_times, event_id, tmin, tmax, proj=False,
                baseline=None, reject=reject, preload=True,
                verbose=True)


epochs.resample(20, npad="auto")

repetitions=120

stims = event_times[:,-1]




# In[1]:   Average classification x trial (unbalanced)
# Primero tengo que agarrar la lista de labels y asignar a los 420 (35x12)
# el label que le corresponde a cada uno.  Es decir de los primeros 12, 10
# son no hits y 2 hits.
hlbls = []
hpreds = []
classlabels=np.asarray([])
for trial in range(0,35):
    a=np.zeros((12*10,2))
    a[:,0] = stims[0+120*trial:0+120*trial+120]
    a[:,1] = lbls[0+120*trial:0+120*trial+120]

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




    #print(hlbls[0][1])



# In[1]:  
def SpellMeLetter(row, col):
    spellermatrix = [ ['A','B','C','D','E','F'],
                    [ 'G','H','I','J','K','L'],
                [ 'M','N','O','P','Q','R'],
                [ 'S','T','U','V','W','X'],
                [ 'Y','Z','1','2','3','4'],
                [ '5','6','7','8','9','_'] ]

    return spellermatrix[row-1][col-1-6]

for i in range(0,35):
    print(SpellMeLetter(hlbls[i][0],hlbls[i][1]),end='')




# In[1]: 
# Luego necesito calcular los 420 averaging (de repetitions)

# Finalmente aprendo con 180 y me fijo si predigo los 240

# De los 240 adivino 20 letras (de a pares) y con eso calculo la performance

for trial in range(0,35):
    epochstrial = epochs[0+repetitions*trial:repetitions*trial+repetitions]

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

event_id = { 'first':1, 'second':2 }
custom_epochs = mne.EpochsArray(epochs_data, info, events, tmin, event_id)





# In[1]: 
print('Performance Classification of Averaged Epochs')
test = range(180,420)
classpreds = np.empty(len(classlabels))
cf = clf.fit(custom_epochs[0:180], classlabels[0:180])
classpreds[test] = clf.predict(custom_epochs[test])

preds = classpreds[test]
labels = classlabels[test]


# Classification report
target_names = ['nohit', 'hit']

report = classification_report(labels, preds, target_names=target_names)
print(report)

cm = confusion_matrix(labels, preds)
print (cm)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
acc=(cm[0,0]+cm[1,1])*1.0/(np.sum(cm))

globalperformance.append(acc)



# In[1]: 

print('Performance Classification of Averaged Epochs')
test = range(180,420)
cf = clf.fit(custom_epochs[0:180], classlabels[0:180])

classpreds = np.empty ((480,2))

classpreds[test,:] = clf.predict_proba(custom_epochs[test])

hpreds = []

for trial in range(15,35):
    #print('Row')
    for i in range(0,6):
        preds = classpreds[trial*12+i]
        #print ( preds[1] )
        labels = classlabels[trial*12+i]

    #print (  np.argmin( classpreds[trial*12+0:trial*12+6]))
    r = np.argmin( classpreds[trial*12+0:trial*12+6,1])+1
    

    #print('Col')
    for i in range(6,12):
        preds = classpreds[trial*12+i]
        #print ( preds[1] )
        labels = classlabels[trial*12+i]

    #print (  np.argmin( classpreds[trial*12+6:trial*12+12]))
    c = np.argmin( classpreds[trial*12+6:trial*12+12,1])+1

    hpreds.append( (r,c) )

# In[1]: 
for i in range(15,35):
    print(SpellMeLetter(hlbls[i][0],hlbls[i][1]),end='')

print()

# In[1]: 
for i in range(15,35):
    print(SpellMeLetter(hpreds[i-15][0],hpreds[i-15][1]),end='')

print()

print(globalperformance)