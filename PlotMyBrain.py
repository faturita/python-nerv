# coding: latin-1
#
# Use to me to plot raw signals from the EPOC Emotiv (Pre 2016)
# (uses the modified emokit library)
# (which uses hidraw on Mac)
#
import emotiv
import platform
import numpy as np
if platform.system() == "Windows":
    import socket
import gevent

import matplotlib.pyplot as plt

import time
import datetime
import os

class Plotter:

    def __init__(self,rangeval,minval,maxval):
        # You probably won't need this if you're embedding things in a tkinter plot...
        import matplotlib.pyplot as plt
        plt.ion()

        self.x = []
        self.y = []
        self.z = []
        self.w = []

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.line1, = self.ax.plot(self.x,'r', label='X') # Returns a tuple of line objects, thus the comma
        self.line2, = self.ax.plot(self.y,'g', label='Y')
        self.line3, = self.ax.plot(self.z,'b', label='Z')
        self.line4, = self.ax.plot(self.w,'y', label='W')

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
        self.w.append( float(new_values[3]))

        self.plotx.append( self.plcounter )

        self.line1.set_ydata(self.x)
        self.line2.set_ydata(self.y)
        self.line3.set_ydata(self.z)
        self.line4.set_ydata(self.w)

        self.line1.set_xdata(self.plotx)
        self.line2.set_xdata(self.plotx)
        self.line3.set_xdata(self.plotx)
        self.line4.set_xdata(self.plotx)

        self.fig.canvas.draw()
        plt.pause(0.000001)

        self.plcounter = self.plcounter+1

        if self.plcounter > self.rangeval:
          self.plcounter = 0
          self.plotx[:] = []
          self.x[:] = []
          self.y[:] = []
          self.z[:] = []
          self.w[:] = []

if __name__ == "__main__":
  plotter = Plotter(500,0,2000)
  connected = False
  while (not connected):
    headset = emotiv.Emotiv(display_output=True)
    gevent.spawn(headset.setup)
    
    gevent.sleep(0.5)

    if (not headset.ready):
      print("Emotiv connection is not ready. Restarting...")
      headset.close()
      gevent.sleep(10)
    else:
      connected = True
  try:
    for i in range(1,128*40):
      packet = headset.dequeue()
      if (packet != None):
          pac = [packet.O1, packet.O2]
          o1 = packet.F7;
          o2 = packet.F8;
          gyx = packet.gyro_x;
          gyy = packet.gyro_y;
          plotter.plotdata( [o1[0],o2[0], gyx+100,gyy+100])
          #print f
          #print np.asarray(pac).shape
      gevent.sleep(0)
  except KeyboardInterrupt:
    headset.close()
  finally:
    headset.close()

gevent.sleep(10)
