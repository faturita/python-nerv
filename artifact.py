#coding: latin-1
import numpy as np

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
