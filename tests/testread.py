#!/usr/bin/env python

import sys
from pylidar import lidarprocessor

def readFunc(data):
    pulses = data.input1.getPulses()
    print('pulses', len(pulses))
    #points = data.input1.getPoints()
    #print('points', len(points))
    
def testRead(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    
    lidarprocessor.doProcessing(readFunc, dataFiles)
    
if __name__ == '__main__':
    testRead(sys.argv[1])
        
