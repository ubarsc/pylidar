"""
Module for doing simple rasterization of LiDAR data.
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
import importlib
from pylidar import lidarprocessor
from rios import cuiprogress

DEFAULT_FUNCTION = "numpy.ma.min"
DEFAULT_ATTRIBUTE = 'POINT'

POINT = 0
PULSE = 1

class RasterizationError(Exception):
    "Exception type for rasterization errors"

def writeImageFunc(data, otherArgs):
    """
    Called from pylidar.lidarprocessor. Calls the nominated
    function on the data.
    """
    # get data for each file
    if otherArgs.atype == POINT:
        dataList = [input.getPointsByBins(colNames=otherArgs.attributes)
                    for input in data.inList]
    else:
        dataList = [input.getPulsesByBins(colNames=otherArgs.attributes)
                    for input in data.inList]

    # stack it so we can analyse the whole thing
    dataStack = numpy.ma.vstack(dataList)

    # create output
    nLayers = len(otherArgs.attributes)
    (maxPts, nRows, nCols) = dataStack.shape
    outStack = numpy.empty((nLayers, nRows, nCols), dtype=numpy.float64)
    if maxPts > 0:
        # a layer per attribute
        nIdx = 0
        for attribute in otherArgs.attributes:
            attributeData = dataStack[attribute]
            attributeDataFunc = otherArgs.func(attributeData, axis=0)
            outStack[nIdx] = attributeDataFunc
            # Need to manually put in the 'background' value. Masked arrays are dangerous. 
            outStack[nIdx][attributeDataFunc.mask] = otherArgs.background
            nIdx += 1
    else:
        outStack.fill(otherArgs.background)

    data.imageOut.setData(outStack)

def rasterize(infiles, outfile, attributes, function=DEFAULT_FUNCTION, 
        atype=DEFAULT_ATTRIBUTE, background=0, binSize=None, extraModule=None, 
        quiet=False, footprint=None, windowSize=None, driverName=None, driverOptions=None):
    """
    Apply the given function to the list of input files and create
    an output raster file. attributes is a list of attributes to run
    the function on. The function name must contain a module
    name and the specified function must take a masked array, plus
    the 'axis' parameter. atype should be a string containing 
    either POINT|PULSE.
    background is the background raster value to use. binSize is the bin size
    to use which defaults to that of the spatial indices used.
    extraModule should be a string with an extra module to import - use
    this for modules other than numpy that are needed by your function.
    quiet means no progress etc
    footprint specifies the footprint type
    """
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.inList = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) 
                        for fname in infiles]
    dataFiles.imageOut = lidarprocessor.ImageFile(outfile, lidarprocessor.CREATE)
    dataFiles.imageOut.setRasterIgnore(background)

    if driverName is not None:
        dataFiles.imageOut.setRasterDriver(driverName)
        
    if driverOptions is not None:
        dataFiles.imageOut.setRasterDriverOptions(driverOptions)

    # import any other modules required
    globalsDict = globals()
    if extraModule is not None:
        globalsDict[extraModule] = importlib.import_module(extraModule)

    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(True)
    if not quiet:
        progress = cuiprogress.GDALProgressBar()
        controls.setProgress(progress)

    if binSize is not None:
        controls.setReferenceResolution(binSize)
    
    if footprint is not None:
        controls.setFootprint(footprint)

    if windowSize is not None:
        controls.setWindowSize(windowSize)

    otherArgs = lidarprocessor.OtherArgs()
    # reference to the function to call
    otherArgs.func = eval(function, globalsDict)
    otherArgs.attributes = attributes
    otherArgs.background = background
    atype = atype.upper()
    if atype == 'POINT':
        otherArgs.atype = POINT
    elif atype == 'PULSE':
        otherArgs.atype = PULSE
    else:
        msg = 'Unsupported type %s' % atype
        raise RasterizationError(msg)

    lidarprocessor.doProcessing(writeImageFunc, dataFiles, controls=controls, 
            otherArgs=otherArgs)
