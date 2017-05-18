
"""
Simple testsuite that checks we can import a simple ascii file
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
from pylidar.toolbox.translate.ascii2spdv4 import translate

INPUT_ASCII = 'first_1402.grd.dat'
IMPORTED_SPD = 'testsuite14.spd'

COLTYPES = [('X', 'FLOAT64'), ('Y', 'FLOAT64'), ('Z', 'FLOAT64'),
    ('ORIG_RETURN_NUMBER', 'UINT64')]
CONST_COLS = [('POINT', 'CLASSIFICATION', 'UINT8', 
    generic.CLASSIFICATION_GROUND)]

def run(oldpath, newpath):
    """
    Runs the 14th basic test suite. Tests:

    Importing time sequential ascii
    """
    inputASCII = os.path.join(oldpath, INPUT_ASCII)
    info = generic.getLidarFileInfo(inputASCII)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputASCII, importedSPD, COLTYPES, constCols=CONST_COLS)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD)

