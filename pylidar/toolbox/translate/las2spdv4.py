"""
Handles conversion between LAS and SPDV4 formats
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

import numpy
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from pylidar.lidarformats import las
from rios import cuiprogress
from rios import pixelgrid
from osgeo import osr

from . import translatecommon

def transFunc(data, otherArgs):
    """
    Called from lidarprocessor. Does the actual conversion to SPD V4
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    waveformInfo = data.input1.getWaveformInfo()
    revc = data.input1.getReceived()
    
    # set scaling and write header
    if data.info.isFirstBlock():
        if otherArgs.useLASScaling:
            updateScalingWithLASValues(otherArgs.scaling, data.input1, points)

        translatecommon.setOutputScaling(otherArgs.scaling, data.output1)
        translatecommon.setOutputNull(otherArgs.nullVals, data.output1)
        if otherArgs.epsg is not None:
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(otherArgs.epsg)
            data.output1.setHeaderValue('SPATIAL_REFERENCE', sr.ExportToWkt())
        else:
            data.output1.setHeaderValue('SPATIAL_REFERENCE', 
                    otherArgs.lasInfo.wkt)

        if data.info.getControls().spatialProcessing:
            # set index type if spatial - always cartesian for LAS (??)
            data.output1.setHeaderValue('INDEX_TYPE', spdv4.SPDV4_INDEX_CARTESIAN)

    # check the range
    translatecommon.checkRange(otherArgs.expectRange, points, pulses, 
            waveformInfo)
    # any constant columns
    points, pulses, waveformInfo = translatecommon.addConstCols(otherArgs.constCols,
            points, pulses, waveformInfo)

    data.output1.setPoints(points)
    data.output1.setPulses(pulses)
    if waveformInfo is not None and waveformInfo.size > 0:
        data.output1.setWaveformInfo(waveformInfo)
    if revc is not None and revc.size > 0:
        data.output1.setReceived(revc)

def updateScalingWithLASValues(scalingDict, input, pointsArray):
    """
    Updates scalingDict with scalings from input (a LAS file). pointsArray
    is needed so we know what fields exist in the input.
    """
    ptsScaling = scalingDict[lidarprocessor.ARRAY_TYPE_POINTS]
    for colName in pointsArray.dtype.names:
        try:
            gain, offset = input.getScaling(colName, 
                    lidarprocessor.ARRAY_TYPE_POINTS)

            dtype = input.getNativeDataType(colName, 
                    lidarprocessor.ARRAY_TYPE_POINTS)

            if colName in ptsScaling:
                # can't change dtype as per spdv4 spec
                dtype = ptsScaling[colName][-1]

            ptsScaling[colName] = [gain, offset, dtype]
        except generic.LiDARArrayColumnError:
            # no scaling/dtype set for this column
            pass

    # we can safely assume that X_IDX, Y_IDX on the pulses etc should be the same as 
    # X, Y on the points (the driver creates X_IDX from X etc)
    plsScaling = scalingDict[lidarprocessor.ARRAY_TYPE_PULSES]
    for inName, outName in [("X", "X_IDX"), ("Y", "Y_IDX")]:
        gain, offset = input.getScaling(inName, lidarprocessor.ARRAY_TYPE_POINTS)

        dtype = plsScaling[outName][-1]
        plsScaling[outName] = [gain, offset, dtype]

def translate(info, infile, outfile, expectRange=None, spatial=None, extent=None, 
        scaling=None, epsg=None, binSize=None, buildPulses=False, pulseIndex=None, 
        nullVals=None, constCols=None, useLASScaling=False):
    """
    Main function which does the work.

    * Info is a fileinfo object for the input file.
    * infile and outfile are paths to the input and output files respectively.
    * expectRange is a list of tuples with (type, varname, min, max).
    * spatial is True or False - dictates whether we are processing spatially or not.
        If True then spatial index will be created on the output file on the fly.
    * extent is a tuple of values specifying the extent to work with. 
        xmin ymin xmax ymax
    * scaling is a list of tuples with (type, varname, dtype, gain, offset).
    * if epsg is not None should be a EPSG number to use as the coord system
    * binSize is the used by the LAS spatial index
    * buildPulses dictates whether to attempt to build the pulse structure
    * pulseIndex should be 'FIRST_RETURN' or 'LAST_RETURN' and determines how the
        pulses are indexed.
    * nullVals is a list of tuples with (type, varname, value)
    * constCols is a list of tupes with (type, varname, dtype, value)
    * if useLASScaling is True, then the scaling used in the LAS file
        is used for columns. Overrides anything given in 'scaling'
    
    """
    scalingsDict = translatecommon.overRideDefaultScalings(scaling)

    if epsg is None and (info.wkt is None or len(info.wkt) == 0):
        msg = 'No projection set in las file. Must set EPSG on command line'
        raise generic.LiDARInvalidSetting(msg)

    if spatial and not info.hasSpatialIndex:
        msg = 'Spatial processing requested but file does not have spatial index'    
        raise generic.LiDARInvalidSetting(msg)

    if spatial and binSize is None:
        msg = "For spatial processing, the bin size must be set"
        raise generic.LiDARInvalidSetting(msg)

    if extent is not None and not spatial:
        msg = 'Extent can only be set when processing spatially'
        raise generic.LiDARInvalidSetting(msg)

    # set up the variables
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    if pulseIndex == 'FIRST_RETURN':
        dataFiles.input1.setLiDARDriverOption('PULSE_INDEX', las.FIRST_RETURN)
    elif pulseIndex == 'LAST_RETURN':
        dataFiles.input1.setLiDARDriverOption('PULSE_INDEX', las.LAST_RETURN)
    else:
        msg = "Pulse index argument not recognised."
        raise generic.LiDARInvalidSetting(msg)

    dataFiles.input1.setLiDARDriverOption('BUILD_PULSES', buildPulses)

    if spatial:
        dataFiles.input1.setLiDARDriverOption('BIN_SIZE', float(binSize))

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(spatial)

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.scaling = scalingsDict
    otherArgs.epsg = epsg
    otherArgs.expectRange = expectRange
    otherArgs.lasInfo = info
    otherArgs.nullVals = nullVals
    otherArgs.constCols = constCols
    otherArgs.useLASScaling = useLASScaling

    if extent is not None:
        extent = [float(x) for x in extent]
        pixgrid = pixelgrid.PixelGridDefn(xMin=extent[0], yMin=extent[1], 
            xMax=extent[2], yMax=extent[3], xRes=binSize, yRes=binSize)
        controls.setReferencePixgrid(pixgrid)
        controls.setFootprint(lidarprocessor.BOUNDS_FROM_REFERENCE)
    
    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')
    dataFiles.output1.setLiDARDriverOption('SCALING_BUT_NO_DATA_WARNING', False)

    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls, 
                    otherArgs=otherArgs)

