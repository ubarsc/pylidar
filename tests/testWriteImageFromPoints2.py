#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid

def writeImageFunc(data):

    pointsByBins1 = data.input1.getPointsByBins()
    pointsByBins2 = data.input2.getPointsByBins()
    zValues1 = pointsByBins1['Z']
    zValues2 = pointsByBins2['Z']
    
    stacked = numpy.ma.vstack((zValues1, zValues2))
    
    (maxPts, nRows, nCols) = stacked.shape

    if maxPts > 0:
        minZ = stacked.min(axis=0)
        minZ = numpy.ma.expand_dims(minZ, axis=0)
    else:
        minZ = numpy.zeros((1, nRows, nCols), dtype=zValues1.dtype)
    
    data.imageOut1.setData(minZ)
    
    
def testWrite(infile1, infile2, imageFile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile1, lidarprocessor.READ)
    dataFiles.input2 = lidarprocessor.LidarFile(infile2, lidarprocessor.READ)
    dataFiles.imageOut1 = lidarprocessor.ImageFile(imageFile, lidarprocessor.CREATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setFootprint(lidarprocessor.UNION)
    controls.setOverlap(10)
    
    lidarprocessor.doProcessing(writeImageFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testWrite(sys.argv[1], sys.argv[2], sys.argv[3])
        
