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

import time
import datetime
import os

if __name__ == "__main__":
  connected = False
  while (not connected):
    headset = emotiv.Emotiv(display_output=True)
    headset.logdown = True
    gevent.spawn(headset.setup)
    
    gevent.sleep(0.5)

    if (not headset.ready):
      print("Emotiv connection is not ready. Restarting...")
      headset.close()
      gevent.sleep(10)
    else:
      connected = True

  try:
    for i in range(1,128*60*5):
    #while (headset.packets_processed<128*40):
      packet = headset.dequeue()
      if (packet != None):
          #pac = [packet.O1, packet.O2]
          #o1 = packet.O1;
          #o2 = packet.O2;
          #gyx = packet.gyro_x;
          #gyy = packet.gyro_y;
          #plotter.plotdata( [o1[0],gyx+100,gyy+100])
          #print f
          #print np.asarray(pac).shape
          pass
      gevent.sleep(0)
  except KeyboardInterrupt:
    headset.close()
  finally:
    headset.close()

gevent.sleep(10)