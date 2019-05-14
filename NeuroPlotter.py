#coding: latin-1

# http://matplotlib.org/faq/virtualenv_faq.html
# Run me with frameworkpython

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

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord

from sklearn import svm
from sklearn.metrics import confusion_matrix

from scipy.signal import butter, lfilter

import artifact as artifact

import emotiv

import matplotlib.pyplot as plt

from Plotter import Plotter

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


def onemore():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    f = open('sensor.dat', 'w')
    plotter = Plotter(500,0,1000)
    print ("Starting main thread")
    readcounter=0
    iterations=0

    while headset.running:
        packet = headset.dequeue()
        interations=iterations+1
        if (packet != None):
            datapoint = [packet.O1, packet.O2]
            #print ("Packet:")
            #print (packet.O1)
            plotter.plotdata( [packet.gyro_x, packet.gyro_y, packet.O1[0]])
            f.write( str(packet.gyro_x) + "\t" + str(packet.gyro_y) + "\n" )
            readcounter=readcounter+1

        if (readcounter==0 and iterations>50):
            headset.running = False
        gevent.sleep(0)

    f.close()
    plotter.close()


if __name__ == "__main__":
    while True:
        KeepRunning = True
        headset = None
        while KeepRunning:
            try:
                headset = emotiv.Emotiv(display_output=False)
                #headset = OfflineHeadset('Rodrigo',1)
                gevent.spawn(headset.setup)
                #g = gevent.spawn(process, headset)
                g = gevent.spawn(onemore)
                gevent.sleep(0)

                gevent.joinall([g])
            except KeyboardInterrupt:
                headset.close()
                quit()
            #except Exception:
                #pass

            if (headset):
                headset.close()
                if (headset.readcounter==0):
                    print ("Restarting headset object...")
                    continue
                else:
                    quit()
