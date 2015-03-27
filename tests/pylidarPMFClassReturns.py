#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from pylidar.toolbox.grdfilters import pmf
from pylidar.toolbox.grdfilters import classGrdReturns
from rios import cuiprogress
from rios import pixelgrid


def applyPMFFilter(data):
    ptBinVals = data.input1.getPointsByBins()#colNames=['X','Y','Z','CLASSIFICATION'])
    pxlCoords = data.info.getBlockCoordArrays()
        
    (maxPts, nRows, nCols) = ptBinVals.shape
    nullval = 0
    if maxPts > 0:
        zValues = ptBinVals['Z']
        # Get Minimum Point Surface...
        minZArr = zValues.min(axis=0)
        # Create a mask of bins where which don't contain any points.
        noPtsMask = (~zValues.mask).sum(axis=0) != 0

        binGeoSize = data.info.getExtent().binSize
        initWinSize=1
        maxWinSize=12
        winSizeInc=1
        slope=0.3
        dh0=0.3
        dhmax=5
        
        pmfOut = pmf.applyPMF(minZArr, noPtsMask, binGeoSize, initWinSize=1, maxWinSize=12, winSizeInc=1, slope=0.3, dh0=0.3, dhmax=5, expWinSizes=False)
        
        classGrdReturns.classifyGroundReturns(ptBinVals, pmfOut, 0.2)
            
    data.input1.setPoints(ptBinVals)
    
def testPMFFilter(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.UPDATE)
    
    controls = lidarprocessor.Controls()
    #controls.setReferenceResolution(1.0)
    controls.setOverlap(5)
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    lidarprocessor.doProcessing(applyPMFFilter, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testPMFFilter(sys.argv[1])
        


