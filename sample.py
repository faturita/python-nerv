import emotiv
import platform
import numpy as np
if platform.system() == "Windows":
    import socket
import gevent

if __name__ == "__main__":
  headset = emotiv.Emotiv()
  gevent.spawn(headset.setup)
  gevent.sleep(0)
  try:
    while True:
      packet = headset.dequeue()
      a = [packet.O1, packet.O2]
      print a
      print np.asarray(a).shape
      gevent.sleep(0)
  except KeyboardInterrupt:
    headset.close()
  finally:
    headset.close()
