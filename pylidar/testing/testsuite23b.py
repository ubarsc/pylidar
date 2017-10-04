
"""
Simple testsuite that checks the toolbox.interpolation.interpPoints function.
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
from . import utils
from pylidar import lidarprocessor
from pylidar.toolbox.interpolation import interpPoints

INPUT_SPD = 'testsuite1.spd'
OUTPUT_DATA = 'testsuite23b.npy'

REQUIRED_FORMATS = ["CGALINTERP"]

def processChunk(data, otherArgs):
    data = data.input1.getPoints(colNames=['X', 'Y', 'Z', 'CLASSIFICATION'])

    groundMask = data['CLASSIFICATION'] == lidarprocessor.CLASSIFICATION_GROUND
    groundPoints = data[groundMask]

    nonGroundMask = ~groundMask
    nonGroundPoints = numpy.empty((numpy.count_nonzero(nonGroundMask), 2), 
                            numpy.double)
    nonGroundPoints[..., 0] = data['X'][nonGroundMask]
    nonGroundPoints[..., 1] = data['Y'][nonGroundMask]

    interped = interpPoints(groundPoints['X'], groundPoints['Y'], 
            groundPoints['Z'], nonGroundPoints, method='cgalnn')

    if otherArgs.output is None:
        otherArgs.output = interped
    else:
        otherArgs.output = numpy.append(otherArgs.output, interped)

def run(oldpath, newpath):
    """
    Runs the 23nd basic test suite. Tests:

    toolbox.interpolation.interpPoints
    """
    inputSPD = os.path.join(oldpath, INPUT_SPD)
    outputDAT = os.path.join(newpath, OUTPUT_DATA)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = lidarprocessor.LidarFile(inputSPD, lidarprocessor.READ)

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.output = None

    lidarprocessor.doProcessing(processChunk, dataFiles, otherArgs=otherArgs)

    numpy.save(outputDAT, otherArgs.output)

    utils.compareNumpyFiles(os.path.join(oldpath, OUTPUT_DATA), outputDAT)
