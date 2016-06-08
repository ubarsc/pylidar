"""
Handles conversion between SPDV3 and SPDV4 formats
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

import copy
import numpy
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from rios import cuiprogress

from . import translatecommon

def getInfoFromHeader(colName, header):
    """
    Guesses at the range and offset of data given
    column name and header from SPDv3
    """
    if colName.startswith('X_') or colName == 'X':
        return (header['X_MAX'] - header['X_MIN']), header['X_MIN']
    elif colName.startswith('Y_') or colName == 'Y':
        return (header['Y_MAX'] - header['Y_MIN']), header['Y_MIN']
    elif (colName.startswith('Z_') or colName.startswith('H_') or colName == 'Z' 
            or colName == 'HEIGHT'):
        return (header['Z_MAX'] - header['Z_MIN']), header['Z_MIN']
    elif colName == 'ZENITH':
        return (header['ZENITH_MAX'] - header['ZENITH_MIN']), header['ZENITH_MIN']
    elif colName == 'AZIMUTH':
        return (header['AZIMUTH_MAX'] - header['AZIMUTH_MIN']), header['AZIMUTH_MIN']
    elif colName.startswith('RANGE_'):
        return (header['RANGE_MAX'] - header['RANGE_MIN']), header['RANGE_MIN']
    else:
        return 100, 0

def setOutputScaling(header, output, scalingsDict):
    """
    Sets the output scaling using info in the SPDv3 header, 
    or in scalingsDict if present.
    """
    xOffset = header['X_MIN']
    yOffset = header['Y_MAX']
    zOffset = header['Z_MIN']
    rangeOffset = header['RANGE_MIN']

    for arrayType in (lidarprocessor.ARRAY_TYPE_PULSES, 
            lidarprocessor.ARRAY_TYPE_POINTS, 
            lidarprocessor.ARRAY_TYPE_WAVEFORMS):
        scaling = scalingsDict[arrayType]
        cols = output.getScalingColumns(arrayType)
        for col in cols:
            dtype = output.getNativeDataType(col, arrayType)
            if col in scaling:
                # use defaults, or overridden on command line first
                gain, offset, dtype = scaling[col]
                output.setScaling(col, arrayType, gain, offset)
                output.setNativeDataType(col, arrayType, dtype)
            else:
                # otherwise guess from old header
                range, offset = getInfoFromHeader(col, header)
                gain = numpy.iinfo(dtype).max / range
                output.setScaling(col, arrayType, gain, offset)

def transFunc(data, otherArgs):
    """
    Called from lidarprocessor. Does the actual conversion to SPD V4
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    waveformInfo = data.input1.getWaveformInfo()
    revc = data.input1.getReceived()
    trans = data.input1.getTransmitted()
    
    data.output1.translateFieldNames(data.input1, points, 
            lidarprocessor.ARRAY_TYPE_POINTS)
    data.output1.translateFieldNames(data.input1, pulses, 
            lidarprocessor.ARRAY_TYPE_PULSES)
            
    # work out scaling 
    if data.info.isFirstBlock():
        header = data.input1.getHeader()
        setOutputScaling(header, data.output1, otherArgs.scalingsDict)
        # write header while we are at it
        data.output1.setHeader(header)
    
    data.output1.setPoints(points)
    data.output1.setPulses(pulses)
    data.output1.setWaveformInfo(waveformInfo)
    data.output1.setReceived(revc)
    data.output1.setTransmitted(trans)

def translate(info, infile, outfile, expectRange, spatial, extent, scaling):
    """
    Main function which does the work.

    * Info is a fileinfo object for the input file.
    * infile and outfile are paths to the input and output files respectively.
    * expectRange is a list of tuples with (type, varname, min, max).
    * spatial is True or False - dictates whether we are processing spatially or not.
        If True then spatial index will be created on the output file on the fly.
    * extent is a tuple of values specifying the extent to work with. 
        xmin ymin xmax ymax
    * scaling is a list of tuples with (type, varname, gain, offset).
    """
    scalingsDict = translatecommon.overRideDefaultScalings(scaling)

    # first we need to determine if the file is spatial or not
    if spatial and not info.hasSpatialIndex:
        msg = "Spatial processing requested but file does not have spatial index"
        raise generic.LiDARInvalidSetting(msg)

    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(spatial)

    if extent is not None:
        extent = [float(x) for x in extent]
        binSize = info.header['BIN_SIZE']
        pixgrid = pixelgrid.PixelGridDefn(xMin=extent[0], yMin=extent[1], 
            xMax=extent[2], yMax=extent[3], xRes=binSize, yRes=binSize)
        controls.setReferencePixgrid(pixgrid)
        controls.setFootprint(lidarprocessor.BOUNDS_FROM_REFERENCE)

    # if they have given us an expected range
    if expectRange is not None:
        translatecommon.getRange(dataFiles.input1, controls, expectRange)

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.scalingsDict = scalingsDict
    
    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls, 
            otherArgs=otherArgs)
