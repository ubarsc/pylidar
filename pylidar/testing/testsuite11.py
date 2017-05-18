
"""
Simple testsuite that checks we can update a column in an SPDV3 file
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

ORIG_FILE = 'gpv1wf_14501655e03676013s_20120504_aa2f0_r06cd_p300khz_x14_sub.spdv3'
UPDATE_FILE = 'testsuite11.spd'

def updatePointFunc(data):

    zVals = data.input1.getPoints(colNames='Z')
    zVals = zVals + 1

    # update Z
    data.input1.setPoints(zVals, colName='Z')

def run(oldpath, newpath):
    """
    Runs the 11th basic test suite. Tests:

    Updating a column in an SPDV3 file
    """
    input = os.path.join(oldpath, ORIG_FILE)
    update = os.path.join(newpath, UPDATE_FILE)
    shutil.copyfile(input, update)

    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(update, lidarprocessor.UPDATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(False)
    
    lidarprocessor.doProcessing(updatePointFunc, dataFiles, controls=controls)

    origUpdate = os.path.join(oldpath, UPDATE_FILE)
    utils.compareLiDARFiles(origUpdate, update)
