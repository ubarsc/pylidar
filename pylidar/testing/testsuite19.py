
"""
Simple testsuite that checks we can export to a PulseWaves file
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
import subprocess
from . import utils
from pylidar.lidarformats import generic
from pylidar.toolbox.translate.spdv42pulsewaves import translate

REQUIRED_FORMATS = ["PULSEWAVES"]

IMPORTED_SPD = 'testsuite8.spd' # came from .rxp originally
EXPORTED_PULSEWAVES = 'testsuite19.pls'

def run(oldpath, newpath):
    """
    Runs the 19th basic test suite. Tests:

    Exporting to PulseWAVES
    """
    inputSPD = os.path.join(oldpath, IMPORTED_SPD)
    info = generic.getLidarFileInfo(inputSPD)

    exportedPW = os.path.join(newpath, EXPORTED_PULSEWAVES)
    translate(info, inputSPD, exportedPW)

    # no pulsediff that we can use to compare. Simply
    # see if file identifies as pulsewaves. Probably can do better.
    info = generic.getLidarFileInfo(exportedPW)
    if info.getDriverName() != 'PulseWaves':
        msg = 'output file was not correctly identified as PulseWaves'
        raise utils.TestingDataMismatch(msg)
