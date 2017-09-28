
"""
Simple testsuite that checks we can set the pixelgrid different
from the spatial index.
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
import numpy
from osgeo import gdal
from . import utils
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid

IN_FILE = 'testsuite1_idx.spd'
OUT_FILE = 'testsuite7.img'
PROJECTION_SOURCE = 'testsuite1.img'

def writeImageFunc(data):
    zValues = data.input.getPointsByBins(colNames='Z')
    (maxPts, nRows, nCols) = zValues.shape
    nullval = 0
    if maxPts > 0:
        minZ = zValues.min(axis=0)
        stack = numpy.ma.expand_dims(minZ, axis=0)
    else:
        stack = numpy.empty((1, nRows, nCols), dtype=zValues.dtype)
        stack.fill(nullval)
    data.imageOut.setData(stack)

def getProjection(imageFile):
    """
    Returns the projection of an image as a WKT
    """
    ds = gdal.Open(imageFile)
    return ds.GetProjection()

def run(oldpath, newpath):
    """
    Runs the 7th basic test suite. Tests:

    setting pixel grid different from the spatial index
    """
    input = os.path.join(oldpath, IN_FILE)
    interp = os.path.join(newpath, OUT_FILE)
    origInterp = os.path.join(oldpath, OUT_FILE)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile(input, lidarprocessor.READ)
    dataFiles.imageOut = lidarprocessor.ImageFile(interp, lidarprocessor.CREATE)

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(True)

    # can't use origInterp as projection source as this might not 
    # be created yet (eg called from testing_cmds.sh)
    projectionSource = os.path.join(oldpath, PROJECTION_SOURCE)
    wkt = getProjection(projectionSource)
    pixGrid = pixelgrid.PixelGridDefn(xMin=509199.0, yMax=6944830, xMax=509857, 
                    yMin=6944130, xRes=2.0, yRes=2.0, projection=wkt)
    controls.setFootprint(lidarprocessor.BOUNDS_FROM_REFERENCE)
    controls.setReferencePixgrid(pixGrid)

    lidarprocessor.doProcessing(writeImageFunc, dataFiles, controls=controls)

    utils.compareImageFiles(origInterp, interp)
