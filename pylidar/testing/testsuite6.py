
"""
Simple testsuite that checks we can do interpolation, set overlap
and pixelgrid.
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
from pylidar.toolbox import interpolation
from rios import cuiprogress
from rios import pixelgrid

IN_FILE = 'testsuite1_idx.spd'
INTERP_FILE = 'testsuite6.img'
PROJECTION_SOURCE = 'testsuite1.img'

REQUIRED_FORMATS = ["PYNNINTERP"]

def interpGroundReturns(data):
    # if given a list of fields, returns a structured array with all of them
    ptVals = data.input.getPoints(colNames=['X', 'Y', 'Z', 'CLASSIFICATION'])
    # create mask for ground
    mask = ptVals['CLASSIFICATION'] == lidarprocessor.CLASSIFICATION_GROUND

    # get the coords for this block
    pxlCoords = data.info.getBlockCoordArrays()

    if ptVals.shape[0] > 0:
        # there is data for this block
        xVals = ptVals['X'][mask]
        yVals = ptVals['Y'][mask]
        zVals = ptVals['Z'][mask]
        # 'pynn' needs the pynnterp module installed
        out = interpolation.interpGrid(xVals, yVals, zVals, pxlCoords, 
                method='pynn')

        # mask out where interpolation failed
        invalid = numpy.isnan(out)
        out[invalid] = 0
    else:
        # no data - set to zero
        out = numpy.empty(pxlCoords[0].shape, dtype=numpy.float64)
        out.fill(0)

    out = numpy.expand_dims(out, axis=0)
    data.imageOut.setData(out)

def getProjection(imageFile):
    """
    Returns the projection of an image as a WKT
    """
    ds = gdal.Open(imageFile)
    return ds.GetProjection()

def run(oldpath, newpath):
    """
    Runs the 6th basic test suite. Tests:

    Interpolation
    overlap
    setting pixel grid
    """
    input = os.path.join(oldpath, IN_FILE)
    interp = os.path.join(newpath, INTERP_FILE)
    origInterp = os.path.join(oldpath, INTERP_FILE)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile(input, lidarprocessor.READ)
    dataFiles.imageOut = lidarprocessor.ImageFile(interp, lidarprocessor.CREATE)

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setOverlap(2)

    # can't use origInterp as projection source as this might not 
    # be created yet (eg called from testing_cmds.sh)
    projectionSource = os.path.join(oldpath, PROJECTION_SOURCE)
    wkt = getProjection(projectionSource)
    pixGrid = pixelgrid.PixelGridDefn(xMin=509198.0, yMax=6944830, xMax=509856, 
                    yMin=6944130, xRes=2.0, yRes=2.0, projection=wkt)
    controls.setFootprint(lidarprocessor.BOUNDS_FROM_REFERENCE)
    controls.setReferencePixgrid(pixGrid)
    controls.setSpatialProcessing(True)

    lidarprocessor.doProcessing(interpGroundReturns, dataFiles, controls=controls)

    utils.compareImageFiles(origInterp, interp)
