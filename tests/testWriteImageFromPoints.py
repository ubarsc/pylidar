#!/usr/bin/env python

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress

def writeImageFunc(data):

    pointsByBins = data.input1.getPointsByBins()
    zValues = pointsByBins['Z']
    avgZ = zValues.mean(axis=0)
    avgZ = numpy.expand_dims(avgZ, axis=0)
    data.imageOut1.setData(avgZ)
    
    
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
        
