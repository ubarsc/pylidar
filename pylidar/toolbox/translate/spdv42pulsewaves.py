"""
Handles conversion between SPDV4 and PulseWaves formats
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
from rios import cuiprogress

def transFunc(data):
    """
    Called from pylidar. Does the actual conversion to pulsewaves
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    waveformInfo = data.input1.getWaveformInfo()
    revc = data.input1.getReceived()
    trans = data.input1.getTransmitted()
    
    data.output1.setPoints(points)
    data.output1.setPulses(pulses)
    if waveformInfo is not None:
        data.output1.setWaveformInfo(waveformInfo)
    if revc is not None:
        data.output1.setReceived(revc)
    if trans is not None:
        data.output1.setTransmitted(trans)

def translate(info, infile, outfile):
    """
    Does the translation between SPD V4 and PulseWaves format files.

    * Info is a fileinfo object for the input file.
    * infile and outfile are paths to the input and output files respectively.

    """
    # set up the variables
    dataFiles = lidarprocessor.DataFiles()
        
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)

    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('PulseWaves')

    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls)
