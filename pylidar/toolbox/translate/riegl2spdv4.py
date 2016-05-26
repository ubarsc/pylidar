"""
Handles conversion between Riegl and SPDV4 formats
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

import copy
import json
import numpy
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from rios import cuiprogress

from . import translatecommon

def setHeaderValues(h, rieglInfo, output):
    """
    Set the header values in the output SPD V4 file using info gathered
    by rangeFunc and the riegl driver. 
    For this test case we assume a RIEGL VZ400
    """  
    h["PULSE_ANGULAR_SPACING_SCANLINE"] = rieglInfo["PHI_INC"]
    h["PULSE_ANGULAR_SPACING_SCANLINE_IDX"] = rieglInfo["THETA_INC"]
    h["SENSOR_BEAM_EXIT_DIAMETER"] = rieglInfo["BEAM_EXIT_DIAMETER"]
    h["SENSOR_BEAM_DIVERGENCE"] = rieglInfo["BEAM_DIVERGENCE"]    
    meta = {'Transform': rieglInfo['ROTATION_MATRIX'].tolist(),
            'Longitude': rieglInfo['LONGITUDE'],
            'Latitude': rieglInfo['LATITUDE'],
            'Height': rieglInfo['HEIGHT'],
            'HMSL': rieglInfo['HMSL']}
    h['USER_META_DATA'] = json.dumps(meta)
    output.setHeader(h)

def transFunc(data, rangeDict):
    """
    Called from translate(). Does the actual conversion to SPD V4
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    waveformInfo = data.input1.getWaveformInfo()
    recv = data.input1.getReceived()
    
    if points is not None:
        data.output1.translateFieldNames(data.input1, points, 
            lidarprocessor.ARRAY_TYPE_POINTS)
    if pulses is not None:
        data.output1.translateFieldNames(data.input1, pulses, 
            lidarprocessor.ARRAY_TYPE_PULSES)
            
    # set scaling and write header
    if data.info.isFirstBlock():
        translatecommon.setOutputScaling(rangeDict, data.output1)
        rieglInfo = data.input1.getHeader()
        setHeaderValues(rangeDict['header'], rieglInfo, data.output1)

    data.output1.setPulses(pulses)
    if points is not None:
        data.output1.setPoints(points)
    if waveformInfo is not None:
        data.output1.setWaveformInfo(waveformInfo)
    if recv is not None:
        data.output1.setReceived(recv)

def translate(info, infile, outfile, scalings, internalrotation, 
        magneticdeclination):
    """
    Main function which does the work.

    * Info is a fileinfo object for the input file.
    * infile and outfile are paths to the input and output files respectively.
    * scaling is a list of tuples with (type, varname, gain, offset).
    * if internalrotation is True then the internal rotation will be applied
        to data
    * magneticdeclination. If not 0, then this will be applied to the data
    """
    scalingsDict = translatecommon.overRideDefaultScalings(scalings)

    # set up the variables
    dataFiles = lidarprocessor.DataFiles()
        
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(False)
    
    rangeDict = translatecommon.getRange(dataFiles.input1)

    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')
    # first get the rotation matrix out of the file if asked for
    if internalrotation:
        if "ROTATION_MATRIX" in info.header:
            dataFiles.output1.setLiDARDriverOption("ROTATION_MATRIX", 
                    info.header["ROTATION_MATRIX"])
        else:
            msg = "Internal Rotation requested but no information found in input file"
            raise generic.LiDARInvalidSetting(msg)
            
    # set the magnetic declination if not 0 (the default)
    if magneticdeclination != 0:
        dataFiles.output1.setLiDARDriverOption("MAGNETIC_DECLINATION", 
                magneticdeclination)

    # also need the default/overriden scaling
    rangeDict['scaling'] = scalingsDict

    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls, 
                    otherArgs=rangeDict)

    