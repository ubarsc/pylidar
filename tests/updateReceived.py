#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid

def updateRecvFunc(data):

    recv = data.input1.getReceived()
    recv = recv * 2
    
    data.input1.setReceived(recv)
    
def testUpdate(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.UPDATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    lidarprocessor.doProcessing(updateRecvFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testUpdate(sys.argv[1])
        
