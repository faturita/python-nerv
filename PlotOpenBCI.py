# coding: latin-1

import sys; sys.path.append('..') # help python find open_bci_v3.py relative to scripts folder
import open_bci_v3 as bci
import os
import logging
import time

import matplotlib.pyplot as plt

from Plotter import Plotter

plotter = Plotter(500,-200000,-150000)

ctr = 0

def printData(sample):
    #os.system('clear')
    print "----------------"
    print("%f" %(sample.id))
    print sample.channel_data
    print sample.aux_data
    plotter.plotdata( [sample.channel_data[4],sample.channel_data[5],sample.channel_data[7]])
    print "----------------"


if __name__ == '__main__':

    port = '/dev/tty.usbserial-DN0096XA'
    port = '/dev/tty.usbserial-DN0096Q1'
    baud = 115200
    logging.basicConfig(filename="test.log",format='%(message)s',level=logging.DEBUG)
    logging.info('---------LOG START-------------')
    board = bci.OpenBCIBoard(port=port, filter_data=True,scaled_output=True, log=True)

    #32 bit reset
    #board.ser.write(b'v')
    time.sleep(2)


    #connect pins to vcc
    #board.ser.write(b'p')
    #time.sleep(0.100)



    #board.print_packets_in()



    board.start_streaming(printData)
