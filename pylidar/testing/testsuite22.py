
"""
Simple testsuite that checks we can import a HDF5 LVIS file
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
from pylidar.toolbox.translate.pulsewaves2spdv4 import translate

INPUT_LVIS = 'testsuite22.h5'
IMPORTED_SPD = 'testsuite22.spd'

# Coords are lat/long, southern hemisphere
SCALING = [('POINT', 'Y', 'DFLT', 100.0, -100.0)]

def run(oldpath, newpath):
    """
    Runs the 22nd basic test suite. Tests:

    Importing LVIS HDF5
    """
    inputLVIS = os.path.join(oldpath, INPUT_LVIS)
    info = generic.getLidarFileInfo(inputLVIS)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputLVIS, importedSPD, scaling=SCALING)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD)
