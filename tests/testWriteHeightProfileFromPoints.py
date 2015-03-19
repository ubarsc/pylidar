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
        # OK 2 issues here. 
        # 1. Masked arrays use False where valid data so we have to invert the mask
        # 2. Masked strcutured arrays seem to have a weird thing where they
        #    get a mask for each element also. So we have to extract just one
        #    field. 
        # If this is a common operation we might have to create a function
        # to make this easier for the user.
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
        
