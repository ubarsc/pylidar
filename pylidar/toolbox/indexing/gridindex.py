
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
Number of blocks to divide the extent up into for the longest axis
if not given a blockSize.
"""
BLOCKSIZE_N_BLOCKS = 2

"""
Types of spatial indices. Copied from spdv4.
"""
INDEX_CARTESIAN = spdv4.SPDV4_INDEX_CARTESIAN
INDEX_SPHERICAL = spdv4.SPDV4_INDEX_SPHERICAL
INDEX_CYLINDRICAL = spdv4.SPDV4_INDEX_CYLINDRICAL
INDEX_POLAR = spdv4.SPDV4_INDEX_POLAR
INDEX_SCAN = spdv4.SPDV4_INDEX_SCAN

"""
Types of pulse indexing methods. Copied from spdv4.
"""
PULSE_INDEX_FIRST_RETURN = spdv4.SPDV4_PULSE_INDEX_FIRST_RETURN
PULSE_INDEX_LAST_RETURN = spdv4.SPDV4_PULSE_INDEX_LAST_RETURN
PULSE_INDEX_START_WAVEFORM = spdv4.SPDV4_PULSE_INDEX_START_WAVEFORM
PULSE_INDEX_END_WAVEFORM = spdv4.SPDV4_PULSE_INDEX_END_WAVEFORM
PULSE_INDEX_ORIGIN = spdv4.SPDV4_PULSE_INDEX_ORIGIN
PULSE_INDEX_MAX_INTENSITY = spdv4.SPDV4_PULSE_INDEX_MAX_INTENSITY

def createGridSpatialIndex(infile, outfile, binSize=1.0, blockSize=None, 
        tempDir='.', extent=None, indexMethod=INDEX_CARTESIAN,
        pulseIndexMethod=PULSE_INDEX_FIRST_RETURN, wkt=None):
    """
    Creates a grid spatially indexed file from a non spatial input file.
    Currently only supports creation of a SPD V4 file.
    
    Creates a tempfile for every block (using blockSize) and them merges them into the output
    building a spatial index as it goes.
    If blockSize isn't set then it is picked using BLOCKSIZE_N_BLOCKS.
    binSize is the size of the bins to create the spatial index.
    indexMethod is on of the INDEX_* constants.
    wkt is the projection to use for the output. Copied from the input if
    not supplied.
    """
    info = generic.getLidarFileInfo(infile)
    header = info.header
    
    if extent is None:
        # work out from header
        try:
            if indexMethod == INDEX_CARTESIAN:
                xMax = header['X_MAX']
                xMin = header['X_MIN']
                yMax = header['Y_MAX']
                yMin = header['Y_MIN']
            elif indexMethod == INDEX_SPHERICAL:
                xMax = header['AZIMUTH_MAX']
                xMin = header['AZIMUTH_MIN']
                yMax = header['ZENITH_MAX']
                yMin = header['ZENITH_MIN']
            elif indexMethod == INDEX_SCAN:
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
        if len(wkt) == 0:
            wkt = getDefaultWKT()

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
    controls.setMessageHandler(lidarprocessor.silentMessageFn)
    
    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.outList = extentList
    otherArgs.indexMethod = indexMethod
    otherArgs.pulseIndexMethod = pulseIndexMethod
    
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
    header['INDEX_TYPE'] = indexMethod
    header['PULSE_INDEX_METHOD'] = pulseIndexMethod
    header['BIN_SIZE'] = binSize
                
    progress.reset()
    progress.setLabelText('Merging...')
    indexAndMerge(newExtentList, extent, wkt, outfile, header, progress)
    
    # close all inputs and delete
    for extent, driver in newExtentList:
        fname = driver.fname
        driver.close()
        os.remove(fname)

def getDefaultWKT():
    """
    When processing data in sensor or project coordinates we may not have a WKT.
    However, rios.pixelgrid requires something. For now
    return the WKT for GDA96/MGA zone 55 until we think of something
    better.
    """
    from osgeo import osr
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(28355)
    return sr.ExportToWkt()

def copyScaling(input, output):
    """
    Copy the known scaling required fields accross.
    
    Internal method. Called from classifyFunc.
    
    """
    for arrayType in (lidarprocessor.ARRAY_TYPE_PULSES, 
            lidarprocessor.ARRAY_TYPE_POINTS, lidarprocessor.ARRAY_TYPE_WAVEFORMS):
        for field in output.getScalingColumns(arrayType):
            try:
                gain, offset = input.getScaling(field, arrayType)
                output.setScaling(field, arrayType, gain, offset)
            except generic.LiDARArrayColumnError:
                pass
                
def setScalingForCoordField(driver, srcfield, coordfield):
    """
    Internal method to set the output scaling for range of data.
    """
    # srcfield and coordfield might not be the same type (pulses/points)
    # this happens when we are using point X and Y to set pulse X_IDX, Y_IDX etc
    if srcfield in spdv4.PULSE_SCALED_FIELDS:
        gain, offset = driver.getScaling(srcfield, lidarprocessor.ARRAY_TYPE_PULSES)
    elif srcfield in spdv4.POINT_SCALED_FIELDS:
        gain, offset = driver.getScaling(srcfield, lidarprocessor.ARRAY_TYPE_POINTS)

    if coordfield in spdv4.PULSE_SCALED_FIELDS:
        driver.setScaling(coordfield, lidarprocessor.ARRAY_TYPE_PULSES, gain, offset)
    elif coordfield in spdv4.POINT_SCALED_FIELDS:
        driver.setScaling(coordfield, lidarprocessor.ARRAY_TYPE_POINTS, gain, offset)
    
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

    for extent, driver in otherArgs.outList:
    
        if data.info.isFirstBlock():
            # deal with scaling. There must be a better way to do this.
            copyScaling(data.input, driver)

        # TODO: should we always be able to rely on X_IDX, Y_IDX for
        # whatever index we are building?
        # No - as the values of these columns may have to change if the
        # properties of the spatial indexing method change   
        if otherArgs.indexMethod == INDEX_CARTESIAN:
            xIdxFieldName = 'X'
            yIdxFieldName = 'Y'
            xIdx, yIdx = indexPulses(pulses, points, recv, otherArgs.pulseIndexMethod)
        elif otherArgs.indexMethod == INDEX_SPHERICAL:
            xIdxFieldName = 'AZIMUTH'
            yIdxFieldName = 'ZENITH'
            xIdx, yIdx = pulses[xIdxFieldName], pulses[yIdxFieldName]
        elif otherArgs.indexMethod == INDEX_SCAN:
            xIdxFieldName = 'SCANLINE_IDX'
            yIdxFieldName = 'SCANLINE'
            xIdx, yIdx = pulses[xIdxFieldName], pulses[yIdxFieldName]              
        else:
            msg = 'unsupported indexing method'
            raise generic.LiDARSpatialIndexNotAvailable(msg)

        # ensure the scaling of X_IDX & Y_IDX matches the data we are putting in it
        setScalingForCoordField(driver, xIdxFieldName, 'X_IDX')
        setScalingForCoordField(driver, yIdxFieldName, 'Y_IDX')
        
        # this the expression used in the spatial index building
        # so we are consistent. 
        mask = ((xIdx >= extent.xMin) & (xIdx < extent.xMax) & 
                (yIdx > extent.yMin) & (yIdx <= extent.yMax))
        # subset the data
        pulsesSub = pulses[mask]
        # this is required otherwise the pulses get stripped out
        # when we write the pulses in spatial mode (in indexAndMerge)
        pulsesSub['X_IDX'] = xIdx[mask]
        pulsesSub['Y_IDX'] = yIdx[mask]
        
        # subset the other data also to match
        pointsSub = points[..., mask]
        
        waveformInfoSub = None
        recvSub = None
        transSub = None
        if waveformInfo is not None and waveformInfo.size > 0:
            waveformInfoSub = waveformInfo[...,mask]
        if recv is not None and recv.size > 0:
            recvSub = recv[...,...,mask]
        if trans is not None and trans.size > 0:
            transSub = trans[...,...,mask]
           
        driver.writeData(pulsesSub, pointsSub, transSub, recvSub, 
                    waveformInfoSub)

def indexPulses(pulses, points, recv, pulseIndexMethod):
    """
    Internal method to assign a point coordinates to the X_IDX and Y_IDX
    columns based on the user specified pulse_index_method.
    """
    if pulseIndexMethod == PULSE_INDEX_FIRST_RETURN:
        if points.shape[0] > 0:
            xIdx = points['X'][0, ...]
            yIdx = points['Y'][0, ...]
        else:
            xIdx = numpy.zeros(points.shape[1],dtype=numpy.uint32)
            yIdx = numpy.zeros(points.shape[1],dtype=numpy.uint32)
    elif pulseIndexMethod == PULSE_INDEX_LAST_RETURN:
        if points.shape[0] > 0:
            firstfield = points.dtype.names[0]
            last = points[firstfield].count(axis=0) - 1
            idx = numpy.arange(last.size)
            xIdx = points['X'][last, idx]
            yIdx = points['Y'][last, idx]
        else:
            xIdx = numpy.zeros(points.shape[1],dtype=numpy.uint32)
            yIdx = numpy.zeros(points.shape[1],dtype=numpy.uint32)         
    else:
        msg = 'unsupported pulse indexing method'
        raise generic.LiDARPulseIndexUnsupported(msg)        

    return xIdx, yIdx

def indexAndMerge(extentList, extent, wkt, outfile, header, progress):
    """
    Internal method to merge all the temporary files into the output
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
    
    # update header
    nrows,ncols = pixGrid.getDimensions()
    header['NUMBER_BINS_X'] = ncols
    header['NUMBER_BINS_Y'] = nrows
    
    progress.setTotalSteps(len(extentList))
    progress.setProgress(0)
    nFilesProcessed = 0
    nFilesWritten = 0
    for subExtent, driver in extentList:

        # read in all the data
        npulses = driver.getTotalNumberPulses()
        if npulses > 0:
            pulseRange = generic.PulseRange(0, npulses)
            driver.setPulseRange(pulseRange)
            pulses = driver.readPulsesForRange()
            points = driver.readPointsByPulse()
            waveformInfo = driver.readWaveformInfo()
            recv = driver.readReceived()
            trans = driver.readTransmitted()

            outDriver.setExtent(subExtent)
            if nFilesWritten == 0:
                copyScaling(driver, outDriver)
                outDriver.setHeader(header)
        
            # on create, a spatial index is created
            outDriver.writeData(pulses, points, trans, recv, 
                            waveformInfo)        

            nFilesWritten += 1
            
        nFilesProcessed += 1
        progress.setProgress(nFilesProcessed)

    outDriver.close()
