#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
import scipy.interpolate
import matplotlib.mlab
from pylidar import lidarprocessor
from pylidar.toolbox import interpolation
from rios import cuiprogress
from rios import pixelgrid

def interpGrdReturnsFunc(data, argsDict):
    ptVals = data.input1.getPoints(colNames=['X','Y','Z','CLASSIFICATION'])
    pxlCoords = data.info.getBlockCoordArrays()
    
    if ptVals.shape[0] > 0:        
        xVals = ptVals['X']
        yVals = ptVals['Y']
        zVals = ptVals['Z']
        classVals = ptVals['CLASSIFICATION']
        
        xVals = xVals[classVals == 3]
        yVals = yVals[classVals == 3]
        zVals = zVals[classVals == 3]
        
        if (numpy.var(xVals) < 4.0) & (numpy.var(yVals) < 4.0):
            out = numpy.empty((1, pxlCoords[0].shape[0], pxlCoords[0].shape[1]), dtype=numpy.float64)
            out.fill(0)
        elif xVals.shape[0] < 4:
            out = numpy.empty((1, pxlCoords[0].shape[0], pxlCoords[0].shape[1]), dtype=numpy.float64)
            out.fill(0)
        elif (numpy.var(xVals) < 4.0) | (numpy.var(yVals) < 4.0):
                out = numpy.empty((1, pxlCoords[0].shape[0], pxlCoords[0].shape[1]), dtype=numpy.float64)
                out.fill(0)
        else:
            out = interpolation.interpGrid(xVals, yVals, zVals, pxlCoords, argsDict['interp'])
            out = numpy.expand_dims(out, axis=0)
    else:
        out = numpy.empty((1, pxlCoords[0].shape[0], pxlCoords[0].shape[1]), dtype=numpy.float64)
        out.fill(0)
    
    #print(out.shape)
    data.imageOut1.setData(out)
    
    
def testInterp(infile, imageFile, overlap, interpolator):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.imageOut1 = lidarprocessor.ImageFile(imageFile, lidarprocessor.CREATE)
    
    controls = lidarprocessor.Controls()
    #controls.setReferenceResolution(1.0)
    controls.setOverlap(overlap)
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    otherArgs = dict()
    otherArgs['interp'] = interpolator
    
    lidarprocessor.doProcessing(interpGrdReturnsFunc, dataFiles, otherArgs=otherArgs, controls=controls)
    
if __name__ == '__main__':
    testInterp(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4])
        


