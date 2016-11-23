
"""
Simple testsuite that checks we can import a SPDV3 file and create a DEM
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
from pylidar.toolbox.translate.spdv32spdv4 import translate
from pylidar.toolbox.indexing.gridindex import createGridSpatialIndex
from pylidar.toolbox.rasterization import rasterize

INPUT_SPDV3 = 'gpv1wf_14501655e03676013s_20120504_aa2f0_r06cd_p300khz_x14_sub.spdv3'
IMPORTED_SPD = 'testsuite10.spd'
INDEXED_SPD = 'testsuite10_idx.spd'
OUTPUT_DEM = 'testsuite10.img'

# override the default scaling
# type, varname, gain, offset
SCALINGS = [('PULSE', 'X_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'Y_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'Z_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'X_IDX', 'DFLT', 1000, -1000),
('PULSE', 'Y_IDX', 'DFLT', 1000, -1000),
('POINT', 'WIDTH_RETURN', 'DFLT', 0.1, -1000.0),
('POINT', 'X', 'DFLT', 1000, -1000),
('POINT', 'Y', 'DFLT', 1000, -1000),
('POINT', 'Z', 'DFLT', 1000, -1000)]

# because the files have lots of points, use a smaller
# windowsize to prevent running out of memory
WINDOWSIZE = 50

def run(oldpath, newpath):
    """
    Runs the 10th basic test suite. Tests:

    Importing SPDV3
    Creating spatial index
    creating a raster
    """
    inputLas = os.path.join(oldpath, INPUT_SPDV3)
    info = generic.getLidarFileInfo(inputLas)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputLas, importedSPD, scaling=SCALINGS)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD,
            windowSize=WINDOWSIZE)

    indexedSPD = os.path.join(newpath, INDEXED_SPD)
    createGridSpatialIndex(importedSPD, indexedSPD, binSize=2.0, 
            tempDir=newpath)
    utils.compareLiDARFiles(os.path.join(oldpath, INDEXED_SPD), indexedSPD,
            windowSize=WINDOWSIZE)

    outputDEM = os.path.join(newpath, OUTPUT_DEM)
    rasterize([indexedSPD], outputDEM, ['Z'], function="numpy.ma.min", 
            atype='POINT', windowSize=WINDOWSIZE)
    utils.compareImageFiles(os.path.join(oldpath, OUTPUT_DEM), outputDEM)
    