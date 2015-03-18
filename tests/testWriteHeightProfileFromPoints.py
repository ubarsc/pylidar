#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid

def writeImageFunc(data):

    pointsByBins = data.input1.getPointsByBins()
    (maxPts, nRows, nCols) = pointsByBins.shape
    nullval = 0
    bins = numpy.mgrid[500:700:5j]
    if maxPts > 0:
        
        ptsByBinsByHeights = data.input1.rebinPtsByHeight(pointsByBins, bins)
        cntByBinsByHeights = (~ptsByBinsByHeights['Z'].mask).sum(axis=0).astype(numpy.int32)
    else:
        cntByBinsByHeights = numpy.empty((len(bins)-1, nRows, nCols), dtype=numpy.int32)
        cntByBinsByHeights.fill(nullval)

    data.imageOut1.setData(cntByBinsByHeights)
    
    
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
        
