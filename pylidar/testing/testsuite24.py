
"""
Simple testsuite that checks we can import a Riegl RDB file
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
import shutil
from . import utils
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.toolbox.translate.rieglrdb2spdv4 import translate
from rios import cuiprogress

REQUIRED_FORMATS = ["RIEGLRDB"]

INPUT_RIEGL = 'testsuite24.rdbx'
IMPORTED_SPD = 'testsuite24.spd'

# override the default scaling
# type, varname, gain, offset
SCALINGS = [('PULSE', 'X_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'Y_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'Z_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'X_IDX', 'DFLT', 1000, -1000),
('PULSE', 'Y_IDX', 'DFLT', 1000, -1000),
('POINT', 'X', 'DFLT', 1000, -1000),
('POINT', 'Y', 'DFLT', 1000, -1000),
('POINT', 'Z', 'DFLT', 1000, -1000),
('POINT', 'RHO_APP', 'DFLT', 1000, -1000)]

# because the files have lots of points, use a smaller
# windowsize to prevent running out of memory
WINDOWSIZE = 50

def run(oldpath, newpath):
    """
    Runs the 24th basic test suite. Tests:

    Importing Riegl RDB
    """
    inputRiegl = os.path.join(oldpath, INPUT_RIEGL)
    info = generic.getLidarFileInfo(inputRiegl)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputRiegl, importedSPD, scalings=SCALINGS)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD,
            windowSize=WINDOWSIZE)

