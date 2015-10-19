#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor

def readFunc(data):
    pulses = data.input1.getPulses()
    print('pulses', len(pulses))
    points = data.input1.getPoints()
    print('points', len(points))
    recv = data.input1.getReceived()
    print(recv)
    trans = data.input1.getTransmitted()
    print(trans)
    #if len(trans) > 0:
    #    print(trans[0])
    #recv = data.input1.getReceived()
    #nrets, npulses = recv.shape
    #for n in range(npulses):
    #    ret = recv[..., n]
    #    if ret.sum() > 0:
    #        print(ret)
    
def testRead(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    
    lidarprocessor.doProcessing(readFunc, dataFiles)
    
if __name__ == '__main__':
    testRead(sys.argv[1])
        
