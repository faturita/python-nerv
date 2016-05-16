#coding: latin-1

import matplotlib.pyplot as plt
import numpy as np

import serial
from struct import *

import sys, select

import emotiv
import platform
import socket
import gevent

import time
import datetime


from scipy.fftpack import fft

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord

from sklearn import svm


class Plotter:

    def __init__(self):
        # You probably won't need this if you're embedding things in a tkinter plot...
        plt.ion()

        self.x = []
        self.y = []
        self.z = []

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.line1, = self.ax.plot(self.x,'r', label='X') # Returns a tuple of line objects, thus the comma
        self.line2, = self.ax.plot(self.y,'g', label='Y')
        self.line3, = self.ax.plot(self.z,'b', label='Z')

        self.ax.axis([0, 500, -5000, 5000])
        self.plcounter = 0
        self.plotx = []

    def plotdata(self,new_values):
        # is  a valid message struct
        print new_values

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

        self.plcounter = self.plcounter+1

        if self.plcounter > 500:
          self.plcounter = 0
          self.plotx[:] = []
          self.x[:] = []
          self.y[:] = []
          self.z[:] = []


def onemore():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    f = open('sensor.dat', 'w')
    plotter = Plotter()
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



while True:
    KeepRunning = True
    while KeepRunning:
        try:
            headset = emotiv.Emotiv()
            gevent.spawn(headset.setup)
            g = gevent.spawn(onemore)
            gevent.sleep(1)

            gevent.joinall([g])
        except KeyboardInterrupt:
            headset.close()
            quit()
        except Exception:
            pass

        headset.close()
        if (headset.readcounter==0):
            print ("Restarting headset object...")
            continue
        else:
            quit()
