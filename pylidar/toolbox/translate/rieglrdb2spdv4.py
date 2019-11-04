"""
Handles conversion between Riegl RDB and SPDV4 formats
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
from osgeo import osr
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from rios import cuiprogress

from . import translatecommon

def transFunc(data, otherArgs):
    """
    Called from translate(). Does the actual conversion to SPD V4
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    
    if points is not None:
        data.output1.translateFieldNames(data.input1, points, 
            lidarprocessor.ARRAY_TYPE_POINTS)
    if pulses is not None:
        data.output1.translateFieldNames(data.input1, pulses, 
            lidarprocessor.ARRAY_TYPE_PULSES)
            
    # set scaling and write header
    if data.info.isFirstBlock():
        translatecommon.setOutputScaling(otherArgs.scaling, data.output1)
        translatecommon.setOutputNull(otherArgs.nullVals, data.output1)
        rieglInfo = otherArgs.rieglInfo

        # TODO: is there likely to be only one gemetry?
        # We go for the first one
        beamGeom = list(rieglInfo["riegl.scan_pattern"].keys())[0]
        data.output1.setHeaderValue("PULSE_ANGULAR_SPACING_SCANLINE", 
                rieglInfo["riegl.scan_pattern"][beamGeom]['phi_increment'])
        data.output1.setHeaderValue("PULSE_ANGULAR_SPACING_SCANLINE_IDX",
                rieglInfo["riegl.scan_pattern"][beamGeom]['theta_increment'])
        data.output1.setHeaderValue("SENSOR_BEAM_EXIT_DIAMETER",
                rieglInfo["riegl.beam_geometry"]['beam_exit_diameter'])
        data.output1.setHeaderValue("SENSOR_BEAM_DIVERGENCE",
                rieglInfo["riegl.beam_geometry"]['beam_divergence'])
                
        data.output1.setHeaderValue("PULSE_INDEX_METHOD", 0) # first return

        if otherArgs.epsg is not None:
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(otherArgs.epsg)
            data.output1.setHeaderValue('SPATIAL_REFERENCE', sr.ExportToWkt())
        elif otherArgs.wkt is not None:
            data.output1.setHeaderValue('SPATIAL_REFERENCE', otherArgs.wkt)

    # check the range
    translatecommon.checkRange(otherArgs.expectRange, points, pulses)
    # any constant columns
    points, pulses, waveformInfo = translatecommon.addConstCols(otherArgs.constCols,
            points, pulses)

    data.output1.setPulses(pulses)
    if points is not None:
        data.output1.setPoints(points)

def translate(info, infile, outfile, expectRange=None, scalings=None, 
        nullVals=None, constCols=None, epsg=None, wkt=None):
    """
    Main function which does the work.

    * Info is a fileinfo object for the input file.
    * infile and outfile are paths to the input and output files respectively.
    * expectRange is a list of tuples with (type, varname, min, max).
    * scaling is a list of tuples with (type, varname, gain, offset).
    * nullVals is a list of tuples with (type, varname, value)
    * constCols is a list of tupes with (type, varname, dtype, value)
    """
    scalingsDict = translatecommon.overRideDefaultScalings(scalings)

    # set up the variables
    dataFiles = lidarprocessor.DataFiles()
        
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(False)

    otherArgs = lidarprocessor.OtherArgs()
    # and the header so we don't collect it again
    otherArgs.rieglInfo = info.header
    # also need the default/overriden scaling
    otherArgs.scaling = scalingsDict
    # expected range of the data
    otherArgs.expectRange = expectRange
    # null values
    otherArgs.nullVals = nullVals
    # constant columns
    otherArgs.constCols = constCols
    otherArgs.epsg = epsg
    otherArgs.wkt = wkt

    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')
    dataFiles.output1.setLiDARDriverOption('SCALING_BUT_NO_DATA_WARNING', False)
    
    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls, 
                    otherArgs=otherArgs)

    
