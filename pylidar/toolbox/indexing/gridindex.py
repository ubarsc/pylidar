
"""
Deals with creating a grid spatial index
"""
# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import print_function, division

import os
import copy
import numpy
import tempfile
from pylidar import lidarprocessor
from pylidar.lidarformats import spdv4
from pylidar.lidarformats import generic
from pylidar.basedriver import Extent
from rios import cuiprogress
from rios import pixelgrid

"""
Number of blocks to divide the extent up into for the longeset axis
if not given
"""
BLOCKSIZE_N_BLOCKS = 2

def createGridSpatialIndex(infile, outfile, binSize=1.0, blockSize=None, 
        tempDir='.', extent=None, indexMethod=spdv4.SPDV4_INDEX_CARTESIAN,
        wkt=None):
    """
    Creates a grid spatially indexed file from a non spatial input file.
    Currently only supports creation of a SPD V4 file.
    
    Creates a tempfile for every block and them merges them into the output.
    If blockSize isn't set then it is picked.
    
    """
    info = generic.getLidarFileInfo(infile)
    header = info.header
    
    if extent is None:
        # work out from header
        try:
            if indexMethod == spdv4.SPDV4_INDEX_CARTESIAN:
                xMax = header['X_MAX']
                xMin = header['X_MIN']
                yMax = header['Y_MAX']
                yMin = header['Y_MIN']
            elif indexMethod == spdv4.SPDV4_INDEX_SPHERICAL:
                xMax = header['AZIMUTH_MAX']
                xMin = header['AZIMUTH_MIN']
                yMax = header['ZENITH_MAX']
                yMin = header['ZENITH_MIN']
            elif indexMethod == spdv4.SPDV4_INDEX_SCAN:
                xMax = header['SCANLINE_IDX_MAX']
                xMin = header['SCANLINE_IDX_MIN']
                yMax = header['SCANLINE_MAX']
                yMin = header['SCANLINE_MIN']
            else:
                msg = 'unsupported indexing method'
                raise generic.LiDARSpatialIndexNotAvailable(msg)
        except KeyError:
            msg = 'info for creating bounding box not available'
            raise generic.LiDARFunctionUnsupported(msg)
            
        extent = Extent(xMin, xMax, yMin, yMax, binSize)
        
    else:
        # ensure that our binSize comes from their exent
        binSize = extent.binSize
        
    if wkt is None:
        wkt = header['SPATIAL_REFERENCE']

    if blockSize is None:
        minAxis = min(extent.xMax - extent.xMin, extent.yMax - extent.yMin)
        blockSize = min(minAxis / BLOCKSIZE_N_BLOCKS, 200.0)
        # make it a multiple of binSize
        blockSize = int(numpy.ceil(blockSize / binSize)) * binSize
    
    extentList = []
    subExtent = Extent(extent.xMin, extent.xMin + blockSize, 
            extent.yMax - blockSize, extent.yMax, binSize)
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)
        
    bMoreToDo = True
    while bMoreToDo:
        fd, fname = tempfile.mkstemp(suffix='.spdv4', dir=tempDir)
        os.close(fd)
        
        userClass = lidarprocessor.LidarFile(fname, generic.CREATE)
        driver = spdv4.SPDV4File(fname, generic.CREATE, controls, userClass)
        data = (copy.copy(subExtent), driver)
        extentList.append(data)

        # move it along
        subExtent.xMin += blockSize
        subExtent.xMax += blockSize

        if subExtent.xMin >= extent.xMax:
            # next line down
            subExtent.xMin = extent.xMin
            subExtent.xMax = extent.xMin + blockSize
            subExtent.yMax -= blockSize
            subExtent.yMin -= blockSize
            
        # done?
        bMoreToDo = subExtent.yMax > extent.yMin

    # ok now set up to read the input file using lidarprocessor
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    progress.setLabelText('Splitting...')
    controls.setProgress(progress)
    controls.setSpatialProcessing(False)
    
    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.outList = extentList
    otherArgs.indexMethod = indexMethod
    
    lidarprocessor.doProcessing(classifyFunc, dataFiles, controls=controls, 
                otherArgs=otherArgs)
    
    # close all the output files and re-open in read mode
    newExtentList = []
    for subExtent, driver in extentList:
        fname = driver.fname
        driver.close()
        userClass = lidarprocessor.LidarFile(fname, generic.READ)
        driver = spdv4.SPDV4File(fname, generic.READ, controls, userClass)
        
        data = (subExtent, driver)
        newExtentList.append(data)
                
    # update header
    header['INDEX_TLX'] = extent.xMin
    header['INDEX_TLY'] = extent.yMax
    # TODO: should this come from the spatial index itself?
    header['NUMBER_BINS_X'] = int(numpy.ceil((extent.xMax - extent.xMin) / binSize))
    header['NUMBER_BINS_Y'] = int(numpy.ceil((extent.yMax - extent.yMin) / binSize))
    header['INDEX_TYPE'] = indexMethod
    header['BIN_SIZE'] = binSize
                
    progress.reset()
    progress.setLabelText('Merging...')
    indexAndMerge(newExtentList, extent, wkt, outfile, header, progress)
    
    # close all inputs and delete
    for extent, driver in newExtentList:
        fname = driver.fname
        driver.close()
        os.remove(fname)

def copyScaling(input, output):
    """
    Copy the known scaling required fields accross.
    There must be a better way to do this.
    """
    for field in ('X_ORIGIN', 'Y_ORIGIN', 'Z_ORIGIN', 'H_ORIGIN', 'X_IDX',
                'Y_IDX', 'AZIMUTH', 'ZENITH'):
        try:
            gain, offset = input.getScaling(field, lidarprocessor.ARRAY_TYPE_PULSES)
            output.setScaling(field, lidarprocessor.ARRAY_TYPE_PULSES, gain, offset)
        except generic.LiDARArrayColumnError:
            pass
                
    for field in ('X', 'Y', 'Z', 'HEIGHT'):
        try:
            gain, offset = input.getScaling(field, lidarprocessor.ARRAY_TYPE_POINTS)
            output.setScaling(field, lidarprocessor.ARRAY_TYPE_POINTS, gain, offset)
        except generic.LiDARArrayColumnError:
            pass
                
    for field in ('RANGE_TO_WAVEFORM_START',):
        try:
            gain, offset = input.getScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS)
            output.setScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS, gain, offset)
        except generic.LiDARArrayColumnError:
            pass
    
def classifyFunc(data, otherArgs):
    """
    Called by lidarprocessor. Looks at the input data and splits into 
    the appropriate output files.
    """
    pulses = data.input.getPulses()
    points = data.input.getPointsByPulse()
    waveformInfo = data.input.getWaveformInfo()
    recv = data.input.getReceived()
    trans = data.input.getTransmitted()

    #xMin = None
    #xMax = None
    #yMin = None
    #yMax = None
    for extent, driver in otherArgs.outList:
    
        if data.info.isFirstBlock():
            # deal with scaling. There must be a better way to do this.
            copyScaling(data.input, driver)

        # TODO: should we always be able to rely on X_IDX, Y_IDX for
        # whatever index we are building?    
        if otherArgs.indexMethod == spdv4.SPDV4_INDEX_CARTESIAN:
            xIdx = pulses['X_IDX']
            yIdx = pulses['Y_IDX']            
        elif otherArgs.indexMethod == spdv4.SPDV4_INDEX_SPHERICAL:
            xIdx = pulses['AZIMUTH']
            yIdx = pulses['ZENITH']            
        elif otherArgs.indexMethod == spdv4.SPDV4_INDEX_SCAN:
            xIdx = pulses['SCANLINE_IDX']
            yIdx = pulses['SCANLINE']            
        else:
            msg = 'unsupported indexing method'
            raise generic.LiDARSpatialIndexNotAvailable(msg)
            
        #xIdx_min = xIdx.min()
        #xIdx_max = xIdx.max()
        #yIdx_min = yIdx.min()
        #yIdx_max = yIdx.max()
        #if xMin is None or xIdx_min < xMin:
        #    xMin = xIdx_min
        #if xMax is None or xIdx_max > xMax:
        #    xMax = xIdx_max
        #if yMin is None or yIdx_min < yMin:
        #    yMin = yIdx_min
        #if yMax is None or yIdx_max > yMax:
        #    yMax = yIdx_max
        
        #print(xIdx.min(), xIdx.max(), yIdx.min(), yIdx.max())
        mask = ((xIdx >= extent.xMin) & (xIdx < extent.xMax) & 
                (yIdx >= extent.yMin) & (yIdx < extent.yMax))
        # subset the data
        pulsesSub = pulses[mask].copy()
        pointsSub = points[..., mask].copy()
        waveformInfoSub = None
        recvSub = None
        transSub = None
        if waveformInfo is not None and waveformInfo.size > 0:
            # TODO: seem to need copy here or crash happens... weird
            # something to do with structured arrays?
            waveformInfoSub = waveformInfo[...,mask].copy()
        if recv is not None and recv.size > 0:
            recvSub = recv[...,...,mask].copy()
        if trans is not None and trans.size > 0:
            transSub = trans[...,...,mask].copy()
            
        driver.writeData(pulsesSub, pointsSub, transSub, recvSub, 
                    waveformInfoSub)
        
def indexAndMerge(extentList, extent, wkt, outfile, header, progress):
    """
    Internal method to merge all the temproray files into the output
    spatially indexing as we go.
    """
    # create output file
    
    userClass = lidarprocessor.LidarFile(outfile, generic.CREATE)
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(True)
    outDriver = spdv4.SPDV4File(outfile, generic.CREATE, controls, userClass)
    pixGrid = pixelgrid.PixelGridDefn(xMin=extent.xMin, xMax=extent.xMax,
                yMin=extent.yMin, yMax=extent.yMax, projection=wkt,
                xRes=extent.binSize, yRes=extent.binSize)
    outDriver.setPixelGrid(pixGrid)
    
    progress.setTotalSteps(len(extentList))
    progress.setProgress(0)
    nFilesProcessed = 0
    for subExtent, driver in extentList:
        # read in all the data
        npulses = driver.getTotalNumberPulses()
        pulseRange = generic.PulseRange(0, npulses)
        driver.setPulseRange(pulseRange)
        pulses = driver.readPulsesForRange()
        points = driver.readPointsByPulse()
        waveformInfo = driver.readWaveformInfo()
        recv = driver.readReceived()
        trans = driver.readTransmitted()

        outDriver.setExtent(subExtent)
        if nFilesProcessed == 0:
            copyScaling(driver, outDriver)
            outDriver.setHeader(header)
        
        # on create, a spatial index is created
        outDriver.writeData(pulses, points, trans, recv, 
                            waveformInfo)
        
        nFilesProcessed += 1
        progress.setProgress(nFilesProcessed)

    outDriver.close()
