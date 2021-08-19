#coding: latin-1
import numpy as np

def do_corr_eog(raw):
    data = raw._data
    eog = data[raw.ch_names.index('EXG5'):raw.ch_names.index('EXG5')+4, :]
    eeg = data[0:64, :]
    le = eeg.shape[-1]
    t1 = int(le * .1)
    t2 = int(le * .9)
  
    for o in eog:
        oc = np.asarray(o[t1:t2], dtype=np.float64)
        pow_oc = np.inner(oc,oc)
        for i in range(len(eeg)):
            eeg[i,...] -= (np.inner(eeg[i,t1:t2], oc) / pow_oc)
    return raw
    
def isartifact(window, threshold=80):
    # Window is EEG Matrix

    awindow = np.asarray(window)
    ameans = np.asarray(  window   ).mean(0)
    signalaverage = ameans.tolist()
    athresholds = np.asarray([threshold]*len(signalaverage))

    #print awindow
    #print ameans
    #print athresholds

    # FIXME
    for t in range(0,len(window)):
        asample = (ameans+athresholds)-awindow[t]
        #print asample
        for c in range(0,asample.shape[0]):
            # while (ameans+athresholds)>(awindow)
            if asample[c]<0:
                return True


    return False

if __name__ == "__main__":
    window = [ [10,11],[10,12],[9,8],[10,10],[10,9],[10,11],[12,9] ]

    print (isartifact(window))
