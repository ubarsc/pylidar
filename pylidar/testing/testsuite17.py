
"""
Simple testsuite that checks we can update a lidar file in non
spatial mode using the spatial toolbox
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
from pylidar.toolbox import spatial
from pylidar.toolbox import arrayutils

INPUT_SPD = 'testsuite1.spd'
INPUT_DEM = 'testsuite16.img'
UPDATE_SPD = 'testsuite17.spd'

def processChunk(data, otherArgs):
    lidar = data.input1.getPoints(colNames=['X', 'Y', 'Z'])
    rows, cols = spatial.xyToRowCol(lidar['X'], lidar['Y'], 
                otherArgs.xMin, otherArgs.yMax, otherArgs.binSize)

    height = lidar['Z'] - otherArgs.inImage[rows, cols]
    lidar = arrayutils.addFieldToStructArray(lidar, 'HEIGHT', numpy.float, height)
    data.input1.setScaling('HEIGHT', lidarprocessor.ARRAY_TYPE_POINTS, 10, -10)
    data.input1.setPoints(lidar)

def run(oldpath, newpath):
    """
    Runs the 17th basic test suite. Tests:

    update a spd file using an image using the non spatial mode
    """
    inputSPD = os.path.join(oldpath, INPUT_SPD)
    updateSPD = os.path.join(newpath, UPDATE_SPD)
    shutil.copyfile(inputSPD, updateSPD)

    inputDEM = os.path.join(oldpath, INPUT_DEM)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = lidarprocessor.LidarFile(updateSPD, lidarprocessor.UPDATE)

    otherArgs = lidarprocessor.OtherArgs()
    (otherArgs.inImage, otherArgs.xMin, otherArgs.yMax, otherArgs.binSize) = spatial.readImageLayer(inputDEM)

    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)

    lidarprocessor.doProcessing(processChunk, dataFiles, otherArgs=otherArgs, controls=controls)

    origUpdate = os.path.join(oldpath, UPDATE_SPD)
    utils.compareLiDARFiles(origUpdate, updateSPD)
