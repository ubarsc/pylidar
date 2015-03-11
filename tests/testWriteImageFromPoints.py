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
        minZ[minZ.mask] = nullval
        stack = numpy.expand_dims(minZ, axis=0)
    else:
        stack = numpy.full((1, nRows, nCols), nullval, dtype=zValues.dtype)
    data.imageOut1.setData(stack)
    
    
def testWrite(infile, imageFile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.imageOut1 = lidarprocessor.ImageFile(imageFile, lidarprocessor.CREATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    pixGrid = pixelgrid.PixelGridDefn(xMin=706445.0, xMax=706545.0, yMin=6153381.0, yMax=6153481.0,
                        xRes=0.5, yRes=0.5, projection='')
#    controls.setReferencePixgrid(pixGrid)
    
    lidarprocessor.doProcessing(writeImageFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testWrite(sys.argv[1], sys.argv[2])
        
