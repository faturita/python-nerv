#coding: latin-1

import time
import datetime
import os

class Packet():
    def init(self):
        self.O1 = 0
        self.O2 = 0
        self.gyro_x = 0
        self.gyro_y = 0


class OfflineHeadset:
    def __init__(self, subject,label,paradigm='Alfa'):
        # @TODO Need to parametrize this.
        # @NOTE Search for datasets on current "Data" directory
        self.basefilename = 'Data/%s/%s/e.%d.l.%d.dat'
        self.paradigm = paradigm
        self.readcounter = 0
        self.running = True
        self.label = label
        self.subject = subject
        self.fileindex = 0
        self.f = None

    def setup(self):
        pass

    def setupfile(self):
        self.datasetfile = self.basefilename % (self.subject,self.paradigm,self.fileindex,self.label)
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
