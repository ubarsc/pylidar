
"""
Simple testsuite that checks we can do an interpolation in non
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
import numpy
from . import utils
from pylidar import lidarprocessor
from pylidar.toolbox import spatial
from pylidar.toolbox import interpolation

INPUT_SPD = 'testsuite1.spd'
OUTPUT_DEM = 'testsuite16.img'

BINSIZE = 1.0

REQUIRED_FORMATS = ["PYNNINTERP"]

def run(oldpath, newpath):
    """
    Runs the 16th basic test suite. Tests:

    creating a raster with interpolation using the 'new' non-spatial mode
    """
    inputSPD = os.path.join(oldpath, INPUT_SPD)
    outputDEM = os.path.join(newpath, OUTPUT_DEM)

    data = spatial.readLidarPoints(inputSPD, 
            classification=lidarprocessor.CLASSIFICATION_GROUND)

    (xMin, yMax, ncols, nrows) = spatial.getGridInfoFromData(data['X'], data['Y'],
                BINSIZE)

    pxlCoords = spatial.getBlockCoordArrays(xMin, yMax, ncols, nrows, BINSIZE)

    dem = interpolation.interpGrid(data['X'], data['Y'], data['Z'], pxlCoords, 
                    method='pynn') 

    iw = spatial.ImageWriter(outputDEM, tlx=xMin, tly=yMax, binSize=BINSIZE)
    iw.setLayer(dem)
    iw.close()

    utils.compareImageFiles(os.path.join(oldpath, OUTPUT_DEM), outputDEM)
