#!/usr/bin/env python

from __future__ import print_function, division

import sys
from pylidar import lidarprocessor
from rios import cuiprogress

def readFunc(data):
    pulses = data.input1.getPulses()
    print('pulses', len(pulses))
    #points = data.input1.getPoints()
    #print('points', len(points))
    
def testRead(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    lidarprocessor.doProcessing(readFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testRead(sys.argv[1])
        
