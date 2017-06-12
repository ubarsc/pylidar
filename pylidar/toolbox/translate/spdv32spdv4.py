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
        translatecommon.setOutputScaling(otherArgs.scaling, data.output1)
        translatecommon.setOutputNull(otherArgs.nullVals, data.output1)

        # copy the index type accross - we can assume these values
        # are the same (at the moment)
        indexType = data.input1.getHeaderValue('INDEX_TYPE')
        data.output1.setHeaderValue('INDEX_TYPE', indexType)

    # check the range
    translatecommon.checkRange(otherArgs.expectRange, points, pulses, 
            waveformInfo)
    # any constant columns
    points, pulses, waveformInfo = translatecommon.addConstCols(otherArgs.constCols,
            points, pulses, waveformInfo)
    
    data.output1.setPoints(points)
    data.output1.setPulses(pulses)
    data.output1.setWaveformInfo(waveformInfo)
    data.output1.setReceived(revc)
    data.output1.setTransmitted(trans)

def translate(info, infile, outfile, expectRange=None, spatial=False, 
            extent=None, scaling=None, nullVals=None, constCols=None):
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
    * nullVals is a list of tuples with (type, varname, value)
    * constCols is a list of tupes with (type, varname, dtype, value)
    """
    scalingsDict = translatecommon.overRideDefaultScalings(scaling)

    # first we need to determine if the file is spatial or not
    if spatial and not info.hasSpatialIndex:
        msg = "Spatial processing requested but file does not have spatial index"
        raise generic.LiDARInvalidSetting(msg)

    if extent is not None and not spatial:
        msg = 'Extent can only be set when processing spatially'
        raise generic.LiDARInvalidSetting(msg)

    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')
    dataFiles.output1.setLiDARDriverOption('SCALING_BUT_NO_DATA_WARNING', False)

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

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.scaling = scalingsDict
    otherArgs.expectRange = expectRange
    otherArgs.nullVals = nullVals
    otherArgs.constCols = constCols
    
    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls, 
            otherArgs=otherArgs)
