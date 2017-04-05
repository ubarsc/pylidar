
"""
Simple testsuite that checks we update an spd file using info 
in a raster
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
import shutil
import numpy
from . import utils
from pylidar import lidarprocessor
from rios import cuiprogress

ORIG_FILE = 'testsuite1_idx.spd'
UPDATE_FILE = 'testsuite12.spd'
IMAGE_FILE = 'testsuite1.img'

def updatePointFunc(data):
    """
    Does the updating of an spd file with heights from an image
    """
    zVals = data.input.getPointsByBins(colNames='Z')
    (nPts, nRows, nCols) = zVals.shape

    if data.info.isFirstBlock():
        data.input.setScaling('HEIGHT', lidarprocessor.ARRAY_TYPE_POINTS, 
                    100.0, 0.0)

    if nPts > 0:
        # read in the DEM data
        dem = data.imageIn.getData()
        # make it match the size of the pts array
        # ie repeat it for the number of bins
        dem = numpy.repeat(dem, zVals.shape[0], axis=0)

        # calculate the height
        # ensure this is a masked array to match pts
        height = numpy.ma.array(zVals - dem, mask=zVals.mask)

        # update the lidar file
        data.input.setPoints(height, colName='HEIGHT')

def run(oldpath, newpath):
    """
    Runs the 12th basic test suite. Tests:

    Updating a spd file with information in a raster
    """
    updateFile = os.path.join(newpath, UPDATE_FILE)
    shutil.copyfile(os.path.join(oldpath, ORIG_FILE), updateFile)

    imageFile = os.path.join(oldpath, IMAGE_FILE)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile(updateFile, lidarprocessor.UPDATE)
    dataFiles.imageIn = lidarprocessor.ImageFile(imageFile, lidarprocessor.READ)

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(True)

    lidarprocessor.doProcessing(updatePointFunc, dataFiles, controls=controls)
    utils.compareLiDARFiles(os.path.join(oldpath, UPDATE_FILE), updateFile)
