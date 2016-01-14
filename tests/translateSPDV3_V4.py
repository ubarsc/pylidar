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
import optparse
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from rios import cuiprogress

class CmdArgs(object):
    def __init__(self):
        p = optparse.OptionParser()
        p.add_option("--spatial", dest="spatial", 
            help="process the data spatially. Specify 'yes' or 'no'. " +
            "Default is spatial if a spatial index exists.")
        p.add_option("--v3", dest="v3",
            help="input v3 .spd file")
        p.add_option("--v4", dest="v4",
            help="output SPD V4 file name")
            
        (options, args) = p.parse_args()
        self.__dict__.update(options.__dict__)

        if self.v3 is None or self.v4 is None:
            p.print_help()
            sys.exit()

def setOutputScaling(header, output):
    xOffset = header['X_MIN']
    yOffset = header['Y_MAX']
    zOffset = header['Z_MIN']
    rangeOffset = header['RANGE_MIN']
    
    dtype = output.getNativeDataType('X_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES)
    xGain = numpy.iinfo(dtype).max / (header['X_MAX'] - header['X_MIN'])
    output.setScaling('X_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, xGain, xOffset)
    
    dtype = output.getNativeDataType('Y_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES)
    yGain = numpy.iinfo(dtype).max / (header['Y_MAX'] - header['Y_MIN'])
    output.setScaling('Y_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, yGain, yOffset)
    
    dtype = output.getNativeDataType('Z_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES)
    zGain = numpy.iinfo(dtype).max / (header['Z_MAX'] - header['Z_MIN'])
    output.setScaling('Z_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, zGain, zOffset)
    
    dtype = output.getNativeDataType('H_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES)
    zGain = numpy.iinfo(dtype).max / (header['Z_MAX'] - header['Z_MIN'])
    output.setScaling('H_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, zGain, zOffset)
    
    dtype = output.getNativeDataType('X_IDX', lidarprocessor.ARRAY_TYPE_PULSES)
    xGain = numpy.iinfo(dtype).max / (header['X_MAX'] - header['X_MIN'])
    output.setScaling('X_IDX', lidarprocessor.ARRAY_TYPE_PULSES, xGain, xOffset)
    
    dtype = output.getNativeDataType('Y_IDX', lidarprocessor.ARRAY_TYPE_PULSES)
    yGain = numpy.iinfo(dtype).max / (header['Y_MAX'] - header['Y_MIN'])
    output.setScaling('Y_IDX', lidarprocessor.ARRAY_TYPE_PULSES, yGain, yOffset)
    
    azOffset = header['AZIMUTH_MIN']
    dtype = output.getNativeDataType('AZIMUTH', lidarprocessor.ARRAY_TYPE_PULSES)
    azGain = numpy.iinfo(dtype).max / (header['AZIMUTH_MAX'] - header['AZIMUTH_MIN'])
    output.setScaling('AZIMUTH', lidarprocessor.ARRAY_TYPE_PULSES, azGain, azOffset)
    
    zenOffset = header['ZENITH_MIN']
    dtype = output.getNativeDataType('ZENITH', lidarprocessor.ARRAY_TYPE_PULSES)
    zenGain = numpy.iinfo(dtype).max / (header['ZENITH_MAX'] - header['ZENITH_MIN'])
    output.setScaling('ZENITH', lidarprocessor.ARRAY_TYPE_PULSES, zenGain, zenOffset)

    dtype = output.getNativeDataType('X', lidarprocessor.ARRAY_TYPE_POINTS)
    xGain = numpy.iinfo(dtype).max / (header['X_MAX'] - header['X_MIN'])
    output.setScaling('X', lidarprocessor.ARRAY_TYPE_POINTS, xGain, xOffset)
    
    dtype = output.getNativeDataType('Y', lidarprocessor.ARRAY_TYPE_POINTS)
    yGain = numpy.iinfo(dtype).max / (header['Y_MAX'] - header['Y_MIN'])
    output.setScaling('Y', lidarprocessor.ARRAY_TYPE_POINTS, yGain, yOffset)
    
    dtype = output.getNativeDataType('Z', lidarprocessor.ARRAY_TYPE_POINTS)
    zGain = numpy.iinfo(dtype).max / (header['Z_MAX'] - header['Z_MIN'])
    output.setScaling('Z', lidarprocessor.ARRAY_TYPE_POINTS, zGain, zOffset)
    
    dtype = output.getNativeDataType('HEIGHT', lidarprocessor.ARRAY_TYPE_POINTS)
    zGain = numpy.iinfo(dtype).max / (header['Z_MAX'] - header['Z_MIN'])
    output.setScaling('HEIGHT', lidarprocessor.ARRAY_TYPE_POINTS, zGain, zOffset)
    
    dtype = output.getNativeDataType('RANGE_TO_WAVEFORM_START', lidarprocessor.ARRAY_TYPE_WAVEFORMS)
    rangeGain = numpy.iinfo(dtype).max / (header['RANGE_MAX'] - header['RANGE_MIN'])
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
    
def convert(infile, outfile, spatial):

    # first we need to determine if the file is spatial or not
    info = generic.getLidarFileInfo(infile)
    if spatial is not None:
        if spatial and not info.hasSpatialIndex:
            raise SystemExit("Spatial processing requested but file does not have spatial index")
    else:
        spatial = info.hasSpatialIndex

    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(spatial)
    
    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    cmdargs = CmdArgs()
    spatial = None
    if cmdargs.spatial is not None:
        spatialStr = cmdargs.spatial.lower()
        if spatialStr != 'yes' and spatialStr != 'no':
            raise SystemExit("Must specify either 'yes' or 'no' for --spatial flag")
    
        spatial = (spatialStr == 'yes')

    convert(cmdargs.v3, cmdargs.v4, spatial)
    
