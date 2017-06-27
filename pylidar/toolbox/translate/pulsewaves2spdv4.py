"""
Handles conversion between PulseWaves and SPDV4 formats
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

import numpy
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from pylidar.lidarformats import lvisbin
from rios import cuiprogress

from . import translatecommon

def transFunc(data, otherArgs):
    """
    Called from lidarprocessor. Does the actual conversion to SPD V4
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    waveformInfo = data.input1.getWaveformInfo()
    revc = data.input1.getReceived()
    trans = data.input1.getTransmitted()
    
    # set scaling and write header
    if data.info.isFirstBlock():

        translatecommon.setOutputScaling(otherArgs.scaling, data.output1)
        translatecommon.setOutputNull(otherArgs.nullVals, data.output1)

    # check the range
    translatecommon.checkRange(otherArgs.expectRange, points, pulses, 
            waveformInfo)
    # any constant columns
    points, pulses, waveformInfo = translatecommon.addConstCols(otherArgs.constCols,
            points, pulses, waveformInfo)

    data.output1.setPoints(points)
    data.output1.setPulses(pulses)
    if waveformInfo is not None and waveformInfo.size > 0:
        data.output1.setWaveformInfo(waveformInfo)
    if revc is not None and revc.size > 0:
        data.output1.setReceived(revc)
    if trans is not None and trans.size > 0:
        data.output1.setTransmitted(trans)

def translate(info, infile, outfile, expectRange=None,  
        scaling=None, nullVals=None, constCols=None):
    """
    Main function which does the work.

    * Info is a fileinfo object for the input file.
    * infile and outfile are paths to the input and output files respectively.
    * expectRange is a list of tuples with (type, varname, min, max).
    * scaling is a list of tuples with (type, varname, dtype, gain, offset).
    * nullVals is a list of tuples with (type, varname, value)
    * constCols is a list of tupes with (type, varname, dtype, value)
    
    """
    scalingsDict = translatecommon.overRideDefaultScalings(scaling)

    # set up the variables
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.scaling = scalingsDict
    otherArgs.expectRange = expectRange
    otherArgs.nullVals = nullVals
    otherArgs.constCols = constCols

    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')
    dataFiles.output1.setLiDARDriverOption('SCALING_BUT_NO_DATA_WARNING', False)

    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls, 
                    otherArgs=otherArgs)

