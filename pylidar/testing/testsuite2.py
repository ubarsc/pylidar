
"""
Simple testsuite that checks we can create a DEM from 2 files.
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
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.toolbox.translate.las2spdv4 import translate
from pylidar.toolbox.indexing.gridindex import createGridSpatialIndex
from pylidar.toolbox.rasterization import rasterize

REQUIRED_FORMATS = ["LAS"]

INPUT2_LAS = 'apl1dr_x510000ys6945000z56_2009_ba1m6_pbrisba.las'
IMPORTED_SPD = 'testsuite2.spd'
INDEXED_SPD_1 = 'testsuite1_idx.spd'
INDEXED_SPD_2 = 'testsuite2_idx.spd'
OUTPUT_DEM = 'testsuite2.img'

def run(oldpath, newpath):
    """
    Runs the first basic test suite. Tests:

    Importing
    Creating spatial index
    creating a raster from 2 files at a different resolution
    """
    inputLas = os.path.join(oldpath, INPUT2_LAS)
    info = generic.getLidarFileInfo(inputLas)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputLas, importedSPD, epsg=28356, 
            pulseIndex='FIRST_RETURN', buildPulses=True)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD)

    indexedSPD1 = os.path.join(oldpath, INDEXED_SPD_1)
    indexedSPD2 = os.path.join(newpath, INDEXED_SPD_2)
    createGridSpatialIndex(importedSPD, indexedSPD2, binSize=2.0, 
            tempDir=newpath)
    utils.compareLiDARFiles(os.path.join(oldpath, INDEXED_SPD_2), indexedSPD2)

    outputDEM = os.path.join(newpath, OUTPUT_DEM)
    rasterize([indexedSPD1, indexedSPD2], outputDEM, ['Z'], binSize=3.0,
            function="numpy.ma.min", atype='POINT', footprint=lidarprocessor.UNION)
    utils.compareImageFiles(os.path.join(oldpath, OUTPUT_DEM), outputDEM)
    