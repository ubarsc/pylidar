
"""
Simple testsuite that checks we can import a LVIS Binary file
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
from pylidar.toolbox.translate.lvisbin2spdv4 import translate

# The input files have version 1.02 of the structures the sizes are in bytes:
# lce=36 lge=52 lgw=492
# these sizes have been used to truncate the input files down to 
# a reasonable size
# also had to clobber the tlon and tlat values in the lce to make sensible...

INPUT_LVIS = 'testsuite21.lce'
IMPORTED_SPD = 'testsuite21.spd'

def run(oldpath, newpath):
    """
    Runs the 21st basic test suite. Tests:

    Importing LVIS Binary
    """
    inputLVIS = os.path.join(oldpath, INPUT_LVIS)
    info = generic.getLidarFileInfo(inputLVIS)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputLVIS, importedSPD)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD)
