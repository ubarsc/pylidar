#!/usr/bin/env python

from __future__ import print_function, division

import sys
from pylidar import lidarprocessor

def readFunc(data):
    pulses1 = data.input1.getPulses()
    pulses2 = data.input2.getPulses()
    print('pulses', len(pulses1), len(pulses2))
    points1 = data.input1.getPoints()
    points2 = data.input2.getPoints()
    print('points', len(points1), len(points2))
    
def testRead(infile1, infile2):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile1, lidarprocessor.READ)
    dataFiles.input2 = lidarprocessor.LidarFile(infile2, lidarprocessor.READ)
    
    lidarprocessor.doProcessing(readFunc, dataFiles)
    
if __name__ == '__main__':
    testRead(sys.argv[1], sys.argv[2])
        
