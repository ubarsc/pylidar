
"""
Simple testsuite that checks we can build a vertical profile
using the canopy toolbox.
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
from . import utils
from pylidar import lidarprocessor
from pylidar.toolbox.canopy.canopymetric import runCanopyMetric

# use the imported .rxp file from testsuite8 so we don't
# have to have the Riegl driver present for this test
IMPORTED_SPD = 'testsuite8.spd'
CANOPY_CSV = 'testsuite9.csv'

def run(oldpath, newpath):
    """
    Runs the 9th basic test suite. Tests:

    Creating a canopy profile
    """
    otherargs = lidarprocessor.OtherArgs()
    
    otherargs.weighted = False
    otherargs.heightcol = 'Z'
    otherargs.heightbinsize = 0.5
    otherargs.minheight = 0.0
    otherargs.maxheight = 50.0
    otherargs.zenithbinsize = 5.0
    otherargs.minazimuth = [0.0]
    otherargs.maxazimuth = [360.0]
    otherargs.minzenith = [35.0]
    otherargs.maxzenith = [70.0]
    otherargs.planecorrection = False
    otherargs.rptfile = None
    otherargs.gridsize = 20
    otherargs.gridbinsize = 5.0
    otherargs.excludedclasses = []
    otherargs.externaldem = None
    otherargs.totalpaimethod = "HINGE"

    inFile = os.path.join(oldpath, IMPORTED_SPD)
    outFile = os.path.join(newpath, CANOPY_CSV)

    runCanopyMetric([inFile], [outFile], "PAVD_CALDERS2014", otherargs)

    newData = numpy.genfromtxt(outFile, delimiter=',', names=True)
    oldData = numpy.genfromtxt(os.path.join(oldpath, CANOPY_CSV), 
                delimiter=',', names=True)

    if not (newData == oldData).all():
        msg = 'New canopy profile does not match old'
        raise utils.TestingDataMismatch(msg)
