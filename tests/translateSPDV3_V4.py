#!/usr/bin/env python

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

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress

MAX_UINT16 = 2**16

def setOutputScaling(header, output):
    xOffset = header['X_MIN']
    xGain = MAX_UINT16 / (header['X_MAX'] - header['X_MIN'])
    yOffset = header['Y_MAX']
    yGain = MAX_UINT16 / (header['Y_MIN'] - header['Y_MAX'])
    zOffset = header['Z_MIN']
    zGain = MAX_UINT16 / (header['Z_MAX'] - header['Z_MIN'])
    rangeOffset = header['RANGE_MIN']
    rangeGain = MAX_UINT16 / (header['RANGE_MAX'] - header['RANGE_MIN'])
    
    output.setScaling('X_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, xGain, xOffset)
    output.setScaling('Y_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, yGain, yOffset)
    output.setScaling('Z_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, zGain, zOffset)
    output.setScaling('H_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, zGain, zOffset)
    output.setScaling('X_IDX', lidarprocessor.ARRAY_TYPE_PULSES, xGain, xOffset)
    output.setScaling('Y_IDX', lidarprocessor.ARRAY_TYPE_PULSES, yGain, yOffset)
    
    azOffset = header['AZIMUTH_MIN']
    azGain = MAX_UINT16 / (header['AZIMUTH_MAX'] - header['AZIMUTH_MIN'])
    output.setScaling('AZIMUTH', lidarprocessor.ARRAY_TYPE_PULSES, azGain, azOffset)
    
    zenOffset = header['ZENITH_MIN']
    zenGain = MAX_UINT16 / (header['ZENITH_MAX'] - header['ZENITH_MIN'])
    output.setScaling('ZENITH', lidarprocessor.ARRAY_TYPE_PULSES, zenGain, zenOffset)

    output.setScaling('X', lidarprocessor.ARRAY_TYPE_POINTS, xGain, xOffset)
    output.setScaling('Y', lidarprocessor.ARRAY_TYPE_POINTS, yGain, yOffset)
    output.setScaling('Z', lidarprocessor.ARRAY_TYPE_POINTS, zGain, zOffset)
    output.setScaling('HEIGHT', lidarprocessor.ARRAY_TYPE_POINTS, zGain, zOffset)
    
    output.setScaling('RANGE_TO_WAVEFORM_START', lidarprocessor.ARRAY_TYPE_WAVEFORMS, 
            rangeGain, rangeOffset)

def transFunc(data):
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    waveformInfo = data.input1.getWaveformInfo()
    revc = data.input1.getReceived()
    trans = data.input1.getTransmitted()
    
    data.output1.translateFieldNames(data.input1, points, 
            lidarprocessor.ARRAY_TYPE_POINTS)
    data.output1.translateFieldNames(data.input1, pulses, 
            lidarprocessor.ARRAY_TYPE_PULSES)
            
    # work out scaling 
    if data.info.isFirstBlock():
        header = data.input1.getHeader()
        setOutputScaling(header, data.output1)
    
    data.output1.setPoints(points)
    data.output1.setPulses(pulses)
    data.output1.setWaveformInfo(waveformInfo)
    data.output1.setReceived(revc)
    data.output1.setTransmitted(trans)
    
def testConvert(infile, outfile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testConvert(sys.argv[1], sys.argv[2])
    
