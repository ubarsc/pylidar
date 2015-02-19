#!/usr/bin/env python

import sys
from pylidar import lidarprocessor

def readFunc(data):
    print(data)
    
def testRead(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    
    lidarprocessor.doProcessing(readFunc, dataFiles)
    
if __name__ == '__main__':
    testRead(sys.argv[1])
        
