
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
        tempDir=None, extent=None, indexType=INDEX_CARTESIAN,
        pulseIndexMethod=PULSE_INDEX_FIRST_RETURN, wkt=None):
    """
    Creates a grid spatially indexed file from a non spatial input file.
    Currently only supports creation of a SPD V4 file.
    
    Creates a tempfile for every block (using blockSize) and them merges them into the output
    building a spatial index as it goes.
    If blockSize isn't set then it is picked using BLOCKSIZE_N_BLOCKS.
    binSize is the size of the bins to create the spatial index.
    if tempDir is none a temporary directory will be created with tempfile.mkdtemp
    and removed at the end of processing.
    extent is an Extent object specifying the extent to work within.
    indexType is one of the INDEX_* constants.
    pulseIndexMethod is one of the PULSE_INDEX_* constants.
    wkt is the projection to use for the output. Copied from the input if
    not supplied.
    nPulsesPerChunkMerge is the number of pulses to process at a time
    when merging.

    """
    removeTempDir = False
    if tempDir is None:
        tempDir = tempfile.mkdtemp()
        removeTempDir = True

    header, extent, extentList = splitFileIntoTiles(infile, binSize=binSize, 
                blockSize=blockSize, tempDir=tempDir, extent=extent, 
                indexType=indexType, pulseIndexMethod=pulseIndexMethod)
    
    # update header
    header['INDEX_TLX'] = extent.xMin
    header['INDEX_TLY'] = extent.yMax
    header['INDEX_TYPE'] = indexType
    header['PULSE_INDEX_METHOD'] = pulseIndexMethod
    header['BIN_SIZE'] = binSize

    if wkt is None:
        wkt = header['SPATIAL_REFERENCE'] 
        if len(wkt) == 0:
            wkt = getDefaultWKT()

    indexAndMerge(extentList, extent, wkt, outfile, header) 
    
    # delete the temp files
    for fname, extent in extentList:
        os.remove(fname)

    # we must have created this directory - remove it
    if removeTempDir:
        os.rmdir(tempDir)

def splitFileIntoTiles(infile, binSize=1.0, blockSize=None, 
        tempDir='.', extent=None, indexType=INDEX_CARTESIAN,
        pulseIndexMethod=PULSE_INDEX_FIRST_RETURN):
    """
    Creates a tempfile for every block (using blockSize).
    If blockSize isn't set then it is picked using BLOCKSIZE_N_BLOCKS.
    binSize is the size of the bins to create the spatial index.
    indexType is one of the INDEX_* constants.
    pulseIndexMethod is one of the PULSE_INDEX_* constants.

    returns the header of the input file, the extent used and a list
    of (fname, extent) tuples that contain the information for 
    each tempfile.
    """
    info = generic.getLidarFileInfo(infile)
    header = info.header
    
    if extent is None:
        # work out from header
        try:
            if indexType == INDEX_CARTESIAN:
                xMax = header['X_MAX']
                xMin = header['X_MIN']
                yMax = header['Y_MAX']
                yMin = header['Y_MIN']
            elif indexType == INDEX_SPHERICAL:
                xMax = header['AZIMUTH_MAX']
                xMin = header['AZIMUTH_MIN']
                yMax = header['ZENITH_MAX']
                yMin = header['ZENITH_MIN']
            elif indexType == INDEX_SCAN:
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

        # round the coords to the nearest multiple
        xMin = numpy.floor(xMin / binSize) * binSize
        yMin = numpy.floor(yMin / binSize) * binSize
        xMax = numpy.ceil(xMax / binSize) * binSize
        yMax = numpy.ceil(yMax / binSize) * binSize
            
        extent = Extent(xMin, xMax, yMin, yMax, binSize)
        
    else:
        # ensure that our binSize comes from their exent
        binSize = extent.binSize
    
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
        userClass.setLiDARDriverOption('SCALING_BUT_NO_DATA_WARNING', False)
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
    otherArgs.indexType = indexType
    otherArgs.pulseIndexMethod = pulseIndexMethod
    
    lidarprocessor.doProcessing(classifyFunc, dataFiles, controls=controls, 
                otherArgs=otherArgs)
    
    # close all the output files and save their names to return
    newExtentList = []
    for subExtent, driver in extentList:
        fname = driver.fname
        driver.close()

        data = (fname, subExtent)
        newExtentList.append(data)

    return header, extent, newExtentList

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
    else:
        # we need to set these in case an unscaled variable is used for the spatial index
        # e.g. using pulse SCANLINE and SCANLINE_IDX to set pulse X_IDX, Y_IDX
        gain = 1.0
        offset = 0.0

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
        if otherArgs.indexType == INDEX_CARTESIAN:
            xIdxFieldName = 'X'
            yIdxFieldName = 'Y'
            xIdx, yIdx = indexPulses(pulses, points, recv, otherArgs.pulseIndexMethod)
        elif otherArgs.indexType == INDEX_SPHERICAL:
            xIdxFieldName = 'AZIMUTH'
            yIdxFieldName = 'ZENITH'
            xIdx, yIdx = pulses[xIdxFieldName], pulses[yIdxFieldName]
        elif otherArgs.indexType == INDEX_SCAN:
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
            recvSub = recv[:,:,mask]
        if trans is not None and trans.size > 0:
            transSub = trans[:,:,mask]
           
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

def indexAndMerge(extentList, extent, wkt, outfile, header):
    """
    Internal method to merge all the temporary files into the output
    spatially indexing as we go.
    """
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)

    # open in read mode
    driverExtentList = []
    for fname, subExtent in extentList:
        userClass = lidarprocessor.LidarFile(fname, generic.READ)
        driver = spdv4.SPDV4File(fname, generic.READ, controls, userClass)
        
        data = (subExtent, driver)
        driverExtentList.append(data)


    # create output file    
    userClass = lidarprocessor.LidarFile(outfile, generic.CREATE)
    userClass.setLiDARDriverOption('SCALING_BUT_NO_DATA_WARNING', False)
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

    # clobber these values since we don't want to 
    # start with the number in the original file
    # they will be reset to 0 in the new file
    del header['NUMBER_OF_POINTS']
    del header['NUMBER_OF_PULSES']
    # these too
    del header['GENERATING_SOFTWARE']
    del header['CREATION_DATETIME']
    
    progress = cuiprogress.GDALProgressBar()
    progress.setLabelText('Merging...')
    progress.setTotalSteps(len(extentList))
    progress.setProgress(0)
    nFilesProcessed = 0
    nFilesWritten = 0
    for subExtent, driver in driverExtentList:

        # read in all the data
        # NOTE: can't write data in blocks as the driver needs to be able to 
        # sort all the data in one go.
        bDataWritten = False
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
            nFilesWritten +=1 

        # close the driver while we are here
        driver.close()
        
        if bDataWritten:
            nFilesWritten += 1
            
        nFilesProcessed += 1
        progress.setProgress(nFilesProcessed)

    outDriver.close()
