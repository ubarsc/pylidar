#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid

def writeImageFunc(data):

    pointsByBins = data.input1.getPointsByBins()
    zValues = pointsByBins['Z']
    (maxPts, nRows, nCols) = zValues.shape
    nullval = 0
    if maxPts > 0:
        minZ = zValues.min(axis=0)
        rasterData = data.imageIn1.getData()
        minZ += rasterData[0]
        stack = numpy.ma.expand_dims(minZ, axis=0)
    else:
        stack = numpy.full((1, nRows, nCols), nullval, dtype=zValues.dtype)
    data.imageOut1.setData(stack)
    
    
def testWrite(infile, inImageFile, outImageFile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.imageIn1 = lidarprocessor.ImageFile(inImageFile, lidarprocessor.READ)
    dataFiles.imageOut1 = lidarprocessor.ImageFile(outImageFile, lidarprocessor.CREATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    lidarprocessor.doProcessing(writeImageFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testWrite(sys.argv[1], sys.argv[2], sys.argv[3])
        
