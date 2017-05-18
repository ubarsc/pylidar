
"""
Simple testsuite that checks we can update a column and add
a column to a SPDV4 file.
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
import shutil
from . import utils
from pylidar import lidarprocessor
from rios import cuiprogress

ORIG_FILE = 'testsuite1_idx.spd'
UPDATE_FILE = 'testsuite4_idx.spd'

def updatePointFunc(data):

    zVals = data.input1.getPointsByBins(colNames='Z')
    zVals = zVals + 1

    zBy2 = zVals / 2.0

    # update Z
    data.input1.setPoints(zVals, colName='Z')
    # create new column ZBY2
    data.input1.setPoints(zBy2, colName='ZBY2')

def run(oldpath, newpath):
    """
    Runs the 4th basic test suite. Tests:

    Updating a column in a file
    Creating a new column
    """
    input = os.path.join(oldpath, ORIG_FILE)
    update = os.path.join(newpath, UPDATE_FILE)
    shutil.copyfile(input, update)

    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(update, lidarprocessor.UPDATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(True)
    
    lidarprocessor.doProcessing(updatePointFunc, dataFiles, controls=controls)

    origUpdate = os.path.join(oldpath, UPDATE_FILE)
    utils.compareLiDARFiles(origUpdate, update)
