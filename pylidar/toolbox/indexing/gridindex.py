
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
    info = None
    
    if extent is None:
        # work out from header
        info = generic.getLidarFileInfo(infile)
        try:
            if indexMethod == spdv4.SPDV4_INDEX_CARTESIAN:
                xMax = info.header['X_MAX']
                xMin = info.header['X_MIN']
                yMax = info.header['Y_MAX']
                yMin = info.header['Y_MIN']
            elif indexMethod == spdv4.SPDV4_INDEX_SPHERICAL:
                xMax = info.header['AZIMUTH_MAX']
                xMin = info.header['AZIMUTH_MIN']
                yMax = info.header['ZENITH_MAX']
                yMin = info.header['ZENITH_MIN']
            elif indexMethod == spdv4.SPDV4_INDEX_SCAN:
                xMax = info.header['SCANLINE_MAX']
                xMin = info.header['SCANLINE_MIN']
                yMax = info.header['SCANLINE_IDX_MAX']
                yMin = info.header['SCANLINE_IDX_MIN']
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
        if info is None:
            info = generic.getLidarFileInfo(infile)
        wkt = info.header['SPATIAL_REFERENCE']

    if blockSize is None:
        maxAxis = max(extent.xMax - extent.xMin, extent.yMax - extent.yMin)
        blockSize = min(maxAxis / BLOCKSIZE_N_BLOCKS, 200.0)
    
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
    controls.setProgress(progress)
    controls.setSpatialProcessing(False)
    
    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.outList = extentList
    
    lidarprocessor.doProcessing(classifyFunc, dataFiles, controls=controls, 
                otherArgs=otherArgs)
                
    indexAndMerge(extentList, extent, wkt, outfile)
    
def classifyFunc(data, otherArgs):
    """
    Called by lidarprocessor. Looks at the input data and splits into 
    the appropriate output files.
    """
    pulses = data.input.getPulses()
    points = data.input.getPointsByPulse()
    waveformInfo = data.input.getWaveformInfo()
    revc = data.input.getReceived()
    trans = data.input.getTransmitted()

    #xMin = None
    #xMax = None
    #yMin = None
    #yMax = None
    for extent, driver in otherArgs.outList:
    
        if data.info.isFirstBlock():
            # deal with scaling. There must be a better way to do this.
            for field in ('X_ORIGIN', 'Y_ORIGIN', 'Z_ORIGIN', 'H_ORIGIN', 'X_IDX',
                        'Y_IDX', 'AZIMUTH', 'ZENITH'):
                gain, offset = data.input.getScaling(field, lidarprocessor.ARRAY_TYPE_PULSES)
                driver.setScaling(field, lidarprocessor.ARRAY_TYPE_PULSES, gain, offset)
                
            for field in ('X', 'Y', 'Z', 'HEIGHT'):
                gain, offset = data.input.getScaling(field, lidarprocessor.ARRAY_TYPE_POINTS)
                driver.setScaling(field, lidarprocessor.ARRAY_TYPE_POINTS, gain, offset)
                
            for field in ('RANGE_TO_WAVEFORM_START',):
                try:
                    gain, offset = data.input.getScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS)
                    driver.setScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS, gain, offset)
                except generic.LiDARArrayColumnError:
                    pass
    
        xIdx = pulses['X_IDX']
        yIdx = pulses['Y_IDX']
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
        pulsesSub = pulses[mask]
        pointsSub = points[..., mask]
        print(pulsesSub.shape, pointsSub.shape)
        #waveformInfoSub = waveformInfo[mask]
        # TODO: waveforms
        driver.writeData(pulsesSub, pointsSub)
        
def indexAndMerge(extentList, extent, wkt, outfile):
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
    
    for extent, driver in extentList:
        # read in all the data
        npulses = driver.getTotalNumberPulses()
        pulseRange = generic.PulseRange(0, npulses)
        driver.setPulseRange(pulseRange)
        pulses = driver.getPulses()
        points = driver.readPointsByPulse()
    
        outDriver.setExtent(extent)
        # on create, a spatial index is created
        outDriver.writeData(points, pulses)
        