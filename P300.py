# coding: latin-1

import mne
mne.set_log_level('WARNING')

import scipy.io
import numpy as np
mat = scipy.io.loadmat('/Users/rramele/GoogleDrive/BCI.Dataset/008-2014/A03.mat')

#mat = scipy.io.loadmat('/Users/rramele/work/GuessMe/signals/p300-subject-01.mat')

# General structure
mat['data'][0][0][0]

# Labels
mat['data'][0][0][0][0][0]

# Data points
mat['data'][0][0][1]

# Targets / No tagets
mat['data'][0][0][2]

# Stimulations
mat['data'][0][0][3]

# Trials start
mat['data'][0][0][4]

# Data point zero for the eight channels.  Should be in V.
signal = mat['data'][0][0][1] * pow(10,6)

print signal.shape

ch_names=[ 'Fz'  ,  'Cz',    'Pz' ,   'Oz'  ,  'P3'  ,  'P4'   , 'PO7'   , 'PO8']
ch_types= ['eeg'] * signal.shape[1]

info = mne.create_info(ch_names, 256, ch_types=ch_types)

eeg_mne = mne.io.array.RawArray(signal.T, info)

eeg_mne.plot_psd()

eeg_mne.filter(1,20)

eeg_mne.plot_psd()

eeg_mne.plot(n_channels=10, block=True)


# Now get the trial information

ch_names_events = ch_names + ['t_type']
ch_types_events = ch_types + ['misc']

# Hits and nohits
t_type = mat['data'][0][0][2]

signal_events = np.concatenate([signal, t_type],1)

info_events = mne.create_info(ch_names_events,256, ch_types_events,true)

eeg_events = mne.io.RawArray(signal_events.T, info_events)

event_times = mne.find_events(eeg_events, stim_channel='t_type')

event_id = { 'second':2 }

tmin = 0
tmax = 1

epochs = mne.Epochs(eeg_mne, event_times, event_id, tmin, tmax)

print ('Epochs x channels x time')
print epochs.get_data().shape

evoked = epochs.average()

evoked.plot()


epochsn = mne.Epochs(eeg_mne, event_times, {'first':1}, tmin, tmax)

print epochsn.get_data().shape

evokedn = epochsn.average()

evokedn.plot()


montage = mne.channels.read_montage('standard_1020')

eeg_mne.set_montage(montage)

eeg_mne.plot_sensors()

eeg_mne.plot(scalings='auto',block=True)


from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Create classification pipeline
clf = make_pipeline(mne.preprocessing.Xdawn(n_components=3),
                    mne.decoding.Vectorizer(),
                    MinMaxScaler(),
                    LogisticRegression(penalty='l1'))

event_id = { 'first':1, 'second':2 }

epochs = mne.Epochs(eeg_mne, event_times, event_id, tmin, tmax, proj=False,
                baseline=None, preload=True,
                verbose=False)

labels = epochs.events[:, -1]

# Cross validator
cv = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=42)

# Do cross-validation
preds = np.empty(len(labels))
for train, test in cv:
    cf=clf.fit(epochs[train], labels[train])
    preds[test] = clf.predict(epochs[test])

# Classification report
target_names = ['nohit', 'hit']

report = classification_report(labels, preds, target_names=target_names)
print(report)

cm = confusion_matrix(labels, preds)
print (cm)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Plot confusion matrix
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
mne.viz.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

np.where( labels == 3 )

y_score = cf.decision_function(epochs)
fpr, tpr, _ = roc_curve(labels, y_score,pos_label=2)
roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


from sklearn.decomposition import PCA, FastICA

X = epochs.get_data()

pca = mne.decoding.UnsupervisedSpatialFilter(PCA(4), average=False)
pca_data = pca.fit_transform(X)
ev = mne.EvokedArray(np.mean(pca_data, axis=0),
                     mne.create_info(4, epochs.info['sfreq'],
                                     ch_types='eeg'), tmin=tmin)
ev.plot(show=False, window_title="PCA")


ica = mne.decoding.UnsupervisedSpatialFilter(FastICA(4), average=False)
ica_data = ica.fit_transform(X)
ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),
                      mne.create_info(4, epochs.info['sfreq'],
                                      ch_types='eeg'), tmin=tmin)
ev1.plot(show=False, window_title='ICA')

plt.show()
