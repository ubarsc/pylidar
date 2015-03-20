#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid

def writeImageFunc(data):

    zValues = data.input1.getPointsByBins(colNames='Z')
    (maxPts, nRows, nCols) = zValues.shape
    nullval = 0
    if maxPts > 0:
        minZ = zValues.min(axis=0)
        stack = numpy.ma.expand_dims(minZ, axis=0)
    else:
        stack = numpy.empty((1, nRows, nCols), dtype=zValues.dtype)
        stack.fill(nullval)
    data.imageOut1.setData(stack)
    
    
def testWrite(infile, imageFile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.imageOut1 = lidarprocessor.ImageFile(imageFile, lidarprocessor.CREATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    lidarprocessor.doProcessing(writeImageFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testWrite(sys.argv[1], sys.argv[2])
        
