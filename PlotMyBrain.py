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

class Plotter:

    def __init__(self,rangeval,minval,maxval):
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



if __name__ == "__main__":
  headset = emotiv.Emotiv(display_output=False)
  plotter = Plotter(500,0,2000)
  gevent.spawn(headset.setup)
  gevent.sleep(0)
  try:
    while True:
      packet = headset.dequeue()
      if (packet != None):
          a = [packet.O1, packet.O2]
          d = packet.O1;
          dd = packet.O2;
          f = packet.gyro_x;
          e = packet.gyro_y;
          plotter.plotdata( [d[0],f+100,e+100])
          print f
          print np.asarray(a).shape
      gevent.sleep(0)
  except KeyboardInterrupt:
    headset.close()
  finally:
    headset.close()
