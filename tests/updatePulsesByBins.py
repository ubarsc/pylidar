#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid

def updatePointFunc(data):

    pls = data.input1.getPulsesByBins(colNames=['X_IDX', 'USER_FIELD'])
    pls['USER_FIELD'] = 82

    data.input1.setPulses(pls)
    
def testUpdate(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.UPDATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setOverlap(10)
    
    lidarprocessor.doProcessing(updatePointFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testUpdate(sys.argv[1])
        
