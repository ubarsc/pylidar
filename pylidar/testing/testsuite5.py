
"""
Simple testsuite that checks we can export to a LAS file
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
from pylidar.toolbox.translate.spdv42las import translate

REQUIRED_FORMATS = ["LAS"]

ORIG_LAS = 'apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba.las'
IMPORTED_SPD = 'testsuite1.spd'
EXPORTED_LAS = 'testsuite5.las'

def run(oldpath, newpath):
    """
    Runs the 5th basic test suite. Tests:

    Exporting to LAS
    """
    inputSPD = os.path.join(oldpath, IMPORTED_SPD)
    info = generic.getLidarFileInfo(inputSPD)

    exportedLAS = os.path.join(newpath, EXPORTED_LAS)
    translate(info, inputSPD, exportedLAS, spatial=False)

    # now run lasdiff
    # maybe need to do better with parsing output
    # not sure if this is valid comparisons with rounding errors etc
    # I've changed this from comparing against ORIG_LAS as there were
    # too many spurious errors. Compare to known good version instead. 
    # Hope this ok.
    origLAS = os.path.join(oldpath, EXPORTED_LAS)
    result = subprocess.check_output(['lasdiff', '-i', exportedLAS, origLAS], 
            stderr=subprocess.STDOUT)

    nLines = len(result.split(b'\n'))
    if nLines > 7:
        print(result)
        msg = ('Seems to be more output from lasdiff than expected, ' + 
                    'likely to be a difference in file')
        raise utils.TestingDataMismatch(msg)
