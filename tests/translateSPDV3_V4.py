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
from pylidar.lidarformats import spdv4
from rios import cuiprogress

PULSE_DEFAULT_GAINS = {'AZIMUTH':100.0, 'ZENITH':100.0, 'X_IDX':100.0, 'Y_IDX':100.0,
'X_ORIGIN':100.0, 'Y_ORIGIN':100.0, 'Z_ORIGIN':100.0, 'H_ORIGIN':100.0,
'AMPLITUDE_PULSE':100.0, 'WIDTH_PULSE':100.0}
PULSE_DEFAULT_OFFSETS = {'AZIMUTH':0.0, 'ZENITH':0.0, 'X_IDX':0.0, 'Y_IDX':0.0, 
'X_ORIGIN':0.0, 'Y_ORIGIN':0.0, 'Z_ORIGIN':0.0, 'H_ORIGIN':0.0,
'AMPLITUDE_PULSE':0.0, 'WIDTH_PULSE':0.0}

POINT_DEFAULT_GAINS = {'X':100.0, 'Y':100.0, 'Z':100.0, 'HEIGHT':100.0, 'INTENSITY':1.0, 'RANGE':100.0, 'AMPLITUDE_RETURN':1.0, 'WIDTH_RETURN':1.0}
POINT_DEFAULT_OFFSETS = {'X':0.0, 'Y':0.0, 'Z':0.0, 'HEIGHT':0.0, 'INTENSITY':0.0, 'RANGE':0.0, 'AMPLITUDE_RETURN':0.0, 'WIDTH_RETURN':0.0}

WAVEFORM_DEFAULT_GAINS = {'RANGE_TO_WAVEFORM_START':100.0}
WAVEFORM_DEFAULT_OFFSETS = {'RANGE_TO_WAVEFORM_START':0.0}


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

def getInfoFromHeader(colName, header):
    """
    Guesses at the range and offset of data given
    column name and header
    """
    if colName.startswith('X_') or colName == 'X':
        return (header['X_MAX'] - header['X_MIN']), header['X_MIN']
    elif colName.startswith('Y_') or colName == 'Y':
        return (header['Y_MAX'] - header['Y_MIN']), header['Y_MIN']
    elif colName.startswith('Z_') or colName.startswith('H_') or colName == 'Z' or colName == 'HEIGHT':
        return (header['Z_MAX'] - header['Z_MIN']), header['Z_MIN']
    elif colName == 'ZENITH':
        return (header['ZENITH_MAX'] - header['ZENITH_MIN']), header['ZENITH_MIN']
    elif colName == 'AZIMUTH':
        return (header['AZIMUTH_MAX'] - header['AZIMUTH_MIN']), header['AZIMUTH_MIN']
    elif colName.startswith('RANGE_'):
        return (header['RANGE_MAX'] - header['RANGE_MIN']), header['RANGE_MIN']
    else:
        return 100, 0

def setOutputScaling(header, output):
    xOffset = header['X_MIN']
    yOffset = header['Y_MAX']
    zOffset = header['Z_MIN']
    rangeOffset = header['RANGE_MIN']

    for arrayType in (lidarprocessor.ARRAY_TYPE_PULSES, 
            lidarprocessor.ARRAY_TYPE_POINTS, lidarprocessor.ARRAY_TYPE_WAVEFORMS):
        cols = output.getScalingColumns(arrayType)
        for col in cols:
            dtype = output.getNativeDataType(col, arrayType)
            if col in PULSE_DEFAULT_GAINS:
                output.setScaling(col, lidarprocessor.ARRAY_TYPE_PULSES, PULSE_DEFAULT_GAINS[col], PULSE_DEFAULT_OFFSETS[col])
            elif col in POINT_DEFAULT_GAINS:
                output.setScaling(col, lidarprocessor.ARRAY_TYPE_POINTS, POINT_DEFAULT_GAINS[col], POINT_DEFAULT_OFFSETS[col]) 
            elif col in WAVEFORM_DEFAULT_GAINS:
                output.setScaling(col, lidarprocessor.ARRAY_TYPE_WAVEFORMS, WAVEFORM_DEFAULT_GAINS[col], WAVEFORM_DEFAULT_OFFSETS[col])
            else:
                range, offset = getInfoFromHeader(col, header)
                gain = numpy.iinfo(dtype).max / range
                output.setScaling(col, arrayType, gain, offset)

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
        # write header while we are at it
        data.output1.setHeader(header)
    
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
    
