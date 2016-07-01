#coding: latin-1

# http://matplotlib.org/faq/virtualenv_faq.html

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

class Plotter:

    def __init__(self,rangeval,minval,maxval):
        # You probably won't need this if you're embedding things in a tkinter plot...
        import matplotlib.pyplot as plt
        plt.ion()

        self.x = []
        self.y = []
        self.z = []

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.line1, = self.ax.plot(self.x,'r', label='X') # Returns a tuple of line objects, thus the comma
        self.line2, = self.ax.plot(self.y,'g', label='Y')
        self.line3, = self.ax.plot(self.z,'b', label='Z')

        self.rangeval = rangeval
        self.ax.axis([0, rangeval, minval, maxval])
        self.plcounter = 0
        self.plotx = []

    def plotdata(self,new_values):
        # is  a valid message struct
        #print new_values

        self.x.append( float(new_values[0]))
        self.y.append( float(new_values[1]))
        self.z.append( float(new_values[2]))

        self.plotx.append( self.plcounter )

        self.line1.set_ydata(self.x)
        self.line2.set_ydata(self.y)
        self.line3.set_ydata(self.z)

        self.line1.set_xdata(self.plotx)
        self.line2.set_xdata(self.plotx)
        self.line3.set_xdata(self.plotx)

        self.fig.canvas.draw()
        plt.pause(0.0001)

        self.plcounter = self.plcounter+1

        if self.plcounter > self.rangeval:
          self.plcounter = 0
          self.plotx[:] = []
          self.x[:] = []
          self.y[:] = []
          self.z[:] = []

class Packet():
    def init(self):
        self.O1 = 0
        self.O2 = 0
        self.gyro_x = 0
        self.gyro_y = 0



class OfflineHeadset:
    def __init__(self, subject, label):
        # @TODO Need to parametrize this.
        self.basefilename = '/Users/rramele/Data/%s/Alfa/e.%d.l.%d.dat'
        self.readcounter = 0
        self.running = True
        self.label = label
        self.subject = subject
        self.fileindex = 0
        self.f = None

    def setup(self):
        pass

    def setupfile(self):
        self.datasetfile = self.basefilename % (self.subject,self.fileindex,self.label)
        print self.datasetfile
        if os.path.isfile(self.datasetfile):
            if self.f:
                self.f.close()
            self.f = open(self.datasetfile,'r')
            return True
        else:
            return False

    def nextline(self):
        line = None
        if self.f:
            line = self.f.readline()
        if (not line):
            self.fileindex = self.fileindex + 1

            if self.setupfile():
                return self.nextline()
            else:
                return None
        else:
            return line

    def dequeue(self):
        line = self.nextline()
        if (line):
            data = line.split('\r\n')[0].split(' ')
            packet = Packet()
            packet.O1 = [float(data[7]),0]
            packet.O2 = [float(data[8]),0]
            packet.gyro_x = 0
            packet.gyro_y = 0

            self.readcounter = self.readcounter + 1
            return packet
        else:
            self.running = False
            return None


    def close(self):
        if (self.f):
            self.f.close()

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

def classify(afeatures1, afeatures2, featuresize):

    #print 'Feature 1 Size %d,%d' % (afeatures1.shape)
    #print 'Feature 2 Size %d,%d' % (afeatures2.shape)

    #print 'Reshape %d' % (afeatures1.shape[0]/(featuresize/afeatures1.shape[1]))
    #print '%d' % featuresize
    #print 'Reshape %d' % (afeatures2.shape[0]/(featuresize/afeatures2.shape[1]))

    trainingfeatures1=afeatures1[0:afeatures1.shape[0]-(afeatures1.shape[0]%featuresize)]
    trainingfeatures2=afeatures2[0:afeatures2.shape[0]-(afeatures2.shape[0]%featuresize)]

    #print 'Feature 1 Size %d,%d' % (trainingfeatures1.shape)
    #print 'Feature 2 Size %d,%d' % (trainingfeatures2.shape)

    #print 'Reshape %d' % (trainingfeatures1.shape[0]/(featuresize/trainingfeatures1.shape[1]))
    #print '%d' % featuresize
    #print 'Reshape %d' % (trainingfeatures2.shape[0]/(featuresize/trainingfeatures2.shape[1]))

    trainingfeatures1 = np.reshape( trainingfeatures1, (trainingfeatures1.shape[0]/(featuresize/trainingfeatures1.shape[1]),featuresize) )
    trainingfeatures2 = np.reshape( trainingfeatures2, (trainingfeatures2.shape[0]/(featuresize/trainingfeatures2.shape[1]),featuresize) )

    #print 'Training 1 Size %d,%d' % (trainingfeatures1.shape)
    #print 'Training 2 Size %d,%d' % (trainingfeatures2.shape)


    trainingdata = np.concatenate ((trainingfeatures1,trainingfeatures2))
    traininglabels = np.concatenate( (np.ones(trainingfeatures1.shape[0]),(np.ones(trainingfeatures2.shape[0])+1) )  )

    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(trainingdata,traininglabels)

    testfeatures1=afeatures1[0:afeatures1.shape[0]-(afeatures1.shape[0]%featuresize)]
    testfeatures2=afeatures1[0:afeatures2.shape[0]-(afeatures2.shape[0]%featuresize)]

    testdata=np.concatenate ((trainingfeatures1,trainingfeatures2))
    testlabels=np.concatenate( (np.ones(trainingfeatures1.shape[0]),(np.ones(trainingfeatures2.shape[0])+1) )  )

    # datapoint = testfeatures1.mean(0)
    # print datapoint
    # datapoints = []
    # datapoints.append( datapoint )
    # print 'Classifying datapoints...'
    # print(clf.predict(datapoints))

    predlabels = clf.predict(testdata)
    C = confusion_matrix(testlabels, predlabels)
    acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
    print '%d Accuracy: %f' % (featuresize,acc)
    print(C)

  # raw_input('Ready?')
  # train()
  # print data
  # label = np.asarray(data)
  # y = np.ones([1,label.shape[0]])
  # raw_input('Second round?')
  # train()
  # print data
  # label = np.asarray(data)
  # y = np.concatenate((y,np.zeros([1,label.shape[0]-y.shape[1]])))
  # print data
  # print y.shape
  # y = y.reshape(1,y.shape[0]*y.shape[1])[0]
  # clf = svm.SVC(kernel='linear', C = 1.0)
  # clf.fit(data,y)
  # raw_input('Run?')
  # run(clf)
  # print(clf.predict([0.58,0.76]))
  # headset.close()


def featureextractor():
    headset = OfflineHeadset('Rodrigo',1)
    features1 = process(headset)
    headset.close()
    headset = OfflineHeadset('Rodrigo',2)
    features2 = process(headset)
    headset.close()

    afeatures1 = np.asarray(features1)
    afeatures2 = np.asarray(features2)

    print (afeatures1.mean(0))
    print (afeatures2.mean(0))


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
    plt.legend(loc='upper left');
    plt.show()


def onemore():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    f = open('sensor.dat', 'w')
    plotter = Plotter(500,4000,5000)
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
        gevent.sleep(0.001)

    f.close()


if __name__ == "__main__":

    featureextractor()




    while False:
        KeepRunning = True
        headset = None
        while KeepRunning:
            try:
                #headset = emotiv.Emotiv()
                headset = OfflineHeadset('Rodrigo',1)
                #gevent.spawn(headset.setup)
                #g = gevent.spawn(process, headset)
                gevent.spawn(featureextractor)
                gevent.sleep(0.001)

                gevent.joinall([g])
            except KeyboardInterrupt:
                headset.close()
                quit()
            except Exception:
                pass

            if (headset):
                headset.close()
                if (headset.readcounter==0):
                    print ("Restarting headset object...")
                    continue
                else:
                    quit()
