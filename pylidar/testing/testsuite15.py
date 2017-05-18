
"""
Simple testsuite that checks we can process a SPD file into an image
using the spatial toolbox in non-spatial mode
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
from numba import jit
from . import utils
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.toolbox import spatial

INPUT_SPD = 'testsuite1.spd'
OUTPUT_DEM = 'testsuite15.img'

BINSIZE = 1.0

@jit
def findMinZs(data, outImage, xMin, yMax):
    for i in range(data.shape[0]):
        if data[i]['CLASSIFICATION'] == lidarprocessor.CLASSIFICATION_GROUND:
            row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                    xMin, yMax, BINSIZE)
            if outImage[row, col] != 0:
                if data[i]['Z'] < outImage[row, col]:
                    outImage[row, col] = data[i]['Z']
            else:
                outImage[row, col] = data[i]['Z']

def processChunk(data, otherArgs):
    lidar = data.input1.getPoints(colNames=['X', 'Y', 'Z', 'CLASSIFICATION'])
    findMinZs(lidar, otherArgs.outImage, otherArgs.xMin, otherArgs.yMax)

def run(oldpath, newpath):
    """
    Runs the 15th basic test suite. Tests:

    creating a raster using the 'new' non-spatial mode
    """
    inputSPD = os.path.join(oldpath, INPUT_SPD)
    outputDEM = os.path.join(newpath, OUTPUT_DEM)

    info = generic.getLidarFileInfo(inputSPD)
    header = info.header

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = lidarprocessor.LidarFile(inputSPD, lidarprocessor.READ)

    xMin, yMax, ncols, nrows = spatial.getGridInfoFromHeader(header, BINSIZE)

    outImage = numpy.zeros((nrows, ncols))

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.outImage = outImage
    otherArgs.xMin = xMin
    otherArgs.yMax = yMax

    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)

    lidarprocessor.doProcessing(processChunk, dataFiles, otherArgs=otherArgs, controls=controls)

    iw = spatial.ImageWriter(outputDEM, tlx=xMin, tly=yMax, binSize=BINSIZE)
    iw.setLayer(outImage)
    iw.close()

    utils.compareImageFiles(os.path.join(oldpath, OUTPUT_DEM), outputDEM)
    