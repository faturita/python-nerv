# coding: latin-1

import emotiv
import platform
import socket
import gevent

import time
import datetime


from scipy.fftpack import fft
import numpy as np

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord

from sklearn import svm


def send():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 7778
    MESSAGE = '{\"status\":\"L\", \"speed\":0,\"balance\":0 }'

    print "UDP target IP:", UDP_IP
    print "UDP target port:", UDP_PORT
    print "message:", MESSAGE
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))


def psd(y):
    # Number of samplepoints
    N = 128
    # sample spacing
    T = 1.0 / 128.0
    # From 0 to N, N*T, 2 points.
    #x = np.linspace(0.0, 1.0, N)
    #y = 1*np.sin(10.0 * 2.0*np.pi*x) + 9*np.sin(20.0 * 2.0*np.pi*x)


    fs = 128.0
    fso2 = fs/2
    Nd,wn = buttord(wp=[9/fso2,11/fso2], ws=[8/fso2,12/fso2],
       gpass=3.0, gstop=40.0)

    b,a = butter(Nd,wn,'band')
    y = filtfilt(b,a,y)


    yf = fft(y)
    #xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    #import matplotlib.pyplot as plt
    #plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    #plt.axis((0,60,0,1))
    #plt.grid()
    #plt.show()

    return np.sum(np.abs(yf[0:N/2]))



def train():
    i=0
    window = []
    try:
        while i<10:
          packet = headset.dequeue()
          #print packet.gyro_x, packet.gyro_y
          window.append([ packet.O1, packet.O2 ])
          awindow = np.asarray(window)
          if ((awindow.shape[0])>128):
            data.append([ psd(awindow[:,0]), psd(awindow[:,1]) ])
            i=i+1
            print i
          gevent.sleep(0)
    except KeyboardInterrupt:
        headset.close()
        quit()
    finally:
        pass


def run(clf):
    i=0
    try:
        while True:
          packet = headset.dequeue()
          #print packet.gyro_x, packet.gyro_y
          datapoint = [packet.O1, packet.O2]
          print datapoint
          clf.predict(datapoint)
          #data.append([ packet.gyro_x, packet.gyro_y ])
          gevent.sleep(0)
    except KeyboardInterrupt:
        headset.close()
        quit()
    finally:
        pass

data=[]

X = []

def onemore():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    f = open('sensor.dat', 'w')

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
            f.write( str(packet.O1) )
            readcounter=readcounter+1

        if (readcounter==0 and iterations>50):
            headset.running = False
        gevent.sleep(0.01)

    f.close()

if __name__ == "__main__":
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


  # print 'Training...'
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
  # quit()
