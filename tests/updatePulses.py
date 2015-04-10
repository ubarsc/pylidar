#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid

def updatePointFunc(data):

    pts = data.input1.getPulses(colNames=['X_IDX', 'USER_FIELD'])
    pts['USER_FIELD'] = 81
    
    data.input1.setPulses(pts)
    
def testUpdate(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.UPDATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    lidarprocessor.doProcessing(updatePointFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testUpdate(sys.argv[1])
        
