
"""
Simple testsuite that checks we can import a time sequential
ascii file
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

REQUIRED_FORMATS = ["ASCIIGZ"]

INPUT_ASCII = '0820Q5_142ew48c_sub.dat.gz'
IMPORTED_SPD = 'testsuite13.spd'

COLTYPES = [('GPS_TIME', 'FLOAT64'), ('X_IDX', 'FLOAT64'), ('Y_IDX', 'FLOAT64'),
('Z_IDX', 'FLOAT64'), ('X', 'FLOAT64'), ('Y', 'FLOAT64'), ('Z', 'FLOAT64'),
('CLASSIFICATION', 'UINT8'), ('ORIG_RETURN_NUMBER', 'UINT8'),
('ORIG_NUMBER_OF_RETURNS', 'UINT8'), ('AMPLITUDE', 'FLOAT64'), 
('FWHM', 'FLOAT64'), ('RANGE', 'FLOAT64')]
PULSE_COLS = ['GPS_TIME', 'X_IDX', 'Y_IDX', 'Z_IDX']

def run(oldpath, newpath):
    """
    Runs the 13th basic test suite. Tests:

    Importing time sequential ascii
    """
    inputASCII = os.path.join(oldpath, INPUT_ASCII)
    info = generic.getLidarFileInfo(inputASCII)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputASCII, importedSPD, COLTYPES, PULSE_COLS)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD)

