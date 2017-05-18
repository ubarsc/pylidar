
"""
Simple testsuite that checks we can import a LAS file and create a DEM
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
from . import utils
from pylidar.lidarformats import generic
from pylidar.toolbox.translate.las2spdv4 import translate
from pylidar.toolbox.indexing.gridindex import createGridSpatialIndex
from pylidar.toolbox.rasterization import rasterize

REQUIRED_FORMATS = ["LAS"]

INPUT_LAS = 'apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba.las'
IMPORTED_SPD = 'testsuite1.spd'
INDEXED_SPD = 'testsuite1_idx.spd'
OUTPUT_DEM = 'testsuite1.img'

def run(oldpath, newpath):
    """
    Runs the first basic test suite. Tests:

    Importing LAS
    Creating spatial index
    creating a raster
    """
    inputLas = os.path.join(oldpath, INPUT_LAS)
    info = generic.getLidarFileInfo(inputLas)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputLas, importedSPD, epsg=28356, 
            pulseIndex='FIRST_RETURN', buildPulses=True)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD)

    indexedSPD = os.path.join(newpath, INDEXED_SPD)
    createGridSpatialIndex(importedSPD, indexedSPD, binSize=2.0, 
            tempDir=newpath)
    utils.compareLiDARFiles(os.path.join(oldpath, INDEXED_SPD), indexedSPD)

    outputDEM = os.path.join(newpath, OUTPUT_DEM)
    rasterize([indexedSPD], outputDEM, ['Z'], function="numpy.ma.min", 
            atype='POINT')
    utils.compareImageFiles(os.path.join(oldpath, OUTPUT_DEM), outputDEM)
    