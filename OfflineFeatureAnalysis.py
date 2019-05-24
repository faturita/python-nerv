#coding: latin-1

# Run with ann virtual environment
# EPOC Emotiv file format https://www.researchgate.net/publication/332514530_EPOC_Emotiv_EEG_Basics

import numpy as np

import serial
from struct import *

import sys, select

#import emotiv
import platform
import socket
import gevent

import time
import datetime
import os

from scipy.fftpack import fft

import math

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

from scipy.signal import butter, lfilter

import artifact as artifact

import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def psd(y):
    # Number of samplepoints
    N = 128
    # sample spacing
    T = 1.0 / 128.0
    # From 0 to N, N*T, 2 points.
    #x = np.linspace(0.0, 1.0, N)
    #y = 1*np.sin(10.0 * 2.0*np.pi*x) + 9*np.sin(20.0 * 2.0*np.pi*x)


    # Original Bandpass
    fs = 128.0
    fso2 = fs/2
    #Nd,wn = buttord(wp=[9/fso2,11/fso2], ws=[8/fso2,12/fso2],
    #   gpass=3.0, gstop=40.0)
    #b,a = butter(Nd,wn,'band')
    #y = filtfilt(b,a,y)

    y = butter_bandpass_filter(y, 8.0, 15.0, fs, order=6)


    yf = fft(y)
    #xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    #import matplotlib.pyplot as plt
    #plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    #plt.axis((0,60,0,1))
    #plt.grid()
    #plt.show()

    return np.sum(np.abs(yf[0:N/2]))

from Plotter import Plotter
from OfflineHeadset import OfflineHeadset

def process(headset):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    log = open('data/biosensor-%s.dat' % st, 'w')
    #plotter = Plotter(500,4000,5000)
    print ("Starting BioProcessing Thread...")
    readcounter=0
    iterations=0

    N = 128

    window = []
    fullsignal = []
    awindow = None
    afullsignal = None
    features = []


    while headset.running:
        packet = headset.dequeue()
        interations=iterations+1
        if (packet != None):
            datapoint = [packet.O1[0], packet.O2[0]]
            #plotter.plotdata( [packet.gyro_x, packet.O2[0], packet.O1[0]])
            log.write( str(packet.gyro_x) + "\t" + str(packet.gyro_y) + "\n" )

            window.append( datapoint )

            if len(window)>=N:
                if not artifact.isartifact(window):
                    awindow = np.asarray( window )
                    fullsignal = fullsignal + window
                    afullsignal = np.asarray( fullsignal )

                    if (len(fullsignal) > 0):
                        awindow = awindow - afullsignal.mean(0)

                    o1 = psd(awindow[:,0])
                    o2 = psd(awindow[:,1])

                    print o1, o2

                    features.append( [o1, o2] )

                # Slide window
                window = window[N/2:N]

            readcounter=readcounter+1

        if (readcounter==0 and iterations>50):
            headset.running = False
        gevent.sleep(0.001)

    log.close()

    return features

def reshapefeature(feature, featuresize):
    feature=feature[0:feature.shape[0]-(feature.shape[0]%featuresize)]
    feature = np.reshape( feature, (feature.shape[0]/(featuresize/feature.shape[1]),featuresize) )

    return feature

def classify(afeatures1, afeatures2, featuresize):

    print 'Feature 1 Size %d,%d' % (afeatures1.shape)
    print 'Feature 2 Size %d,%d' % (afeatures2.shape)

    afeatures1 = reshapefeature(afeatures1, featuresize)
    afeatures2 = reshapefeature(afeatures2, featuresize)

    featuredata = np.concatenate ((afeatures1,afeatures2))
    featurelabels = np.concatenate( (np.zeros(afeatures1.shape[0]),(np.zeros(afeatures2.shape[0])+1) )  )

    boundary = int(featuredata.shape[0]/2.0)

    print ('Boundary %d:' % boundary)

    # Reshape and shuffle the features
    reorder = np.random.permutation(featuredata.shape[0])

    trainingdata = featuredata[reorder[0:boundary]]
    traininglabels = featurelabels[reorder[0:boundary]]

    testdata = featuredata[reorder[boundary+1:featuredata.shape[0]]]
    testlabels = featurelabels[reorder[boundary+1:featuredata.shape[0]]]

    print ('Training Dataset Size %d,%d' % (trainingdata.shape))
    print 'Test Dataset Size %d,%d' % (testdata.shape)


    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential([
        Dense(64, activation='tanh', input_shape=(trainingdata.shape[1],)),
        Dense(32, activation='tanh'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    hist = model.fit(trainingdata, traininglabels,
          batch_size=10, epochs=1000*trainingdata.shape[1],verbose=0,
          validation_split=0.4)


    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(trainingdata,traininglabels)


    predlabels = clf.predict(testdata)
    C = confusion_matrix(testlabels, predlabels)
    acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
    print 'SVM Feature Dim: %d Accuracy: %f' % (featuresize,acc)
    print(C)

    predlabels = model.predict(testdata)
    #print(predlabels)
    predlabels = predlabels.round()
    #print(predlabels)
    C = confusion_matrix(testlabels, predlabels)
    acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
    print 'Keras Feature Dim: %d Accuracy: %f' % (featuresize,acc)
    print(C)

    print(model.evaluate(testdata,testlabels))
    print 'Keras Model Accuracy: %f' % (model.evaluate(testdata,testlabels)[1])

    target_names = ['Open', 'Closed']
    report = classification_report(testlabels, predlabels, target_names=target_names)
    print(report)

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()


def featureextractor():
    # Get features from label 1.
    headset = OfflineHeadset('Sleepy',1)
    features1 = process(headset)
    headset.close()
    # Get features from label 2
    headset = OfflineHeadset('Sleepy',2)
    features2 = process(headset)
    headset.close()

    afeatures1 = np.asarray(features1)
    afeatures2 = np.asarray(features2)

    print (afeatures1.mean(0))
    print (afeatures2.mean(0))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(afeatures1[:,0], afeatures1[:,1], s=10, c='b', marker="x", label='Open')
    ax1.scatter(afeatures2[:,0], afeatures2[:,1], s=10, c='r', marker="o", label='Closed')
    plt.xlabel('PSD O2')
    plt.ylabel('PSD O1')
    plt.legend(loc='upper left')
    plt.show()

    # Group time features in tuples, 4-tuples and 8-tuples and classify them
    classify(afeatures1, afeatures2,2)
    classify(afeatures1, afeatures2,4)
    classify(afeatures1, afeatures2,8)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(afeatures1[:,0], afeatures1[:,1], s=10, c='b', marker="x", label='Open')
    ax1.scatter(afeatures2[:,0], afeatures2[:,1], s=10, c='r', marker="o", label='Closed')
    plt.xlabel('PSD O2')
    plt.ylabel('PSD O1')
    plt.legend(loc='upper left')
    plt.show()



if __name__ == "__main__":

    featureextractor()
