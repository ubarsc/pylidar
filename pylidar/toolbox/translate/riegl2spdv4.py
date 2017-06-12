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
        translatecommon.setOutputScaling(otherArgs.scaling, data.output1)
        translatecommon.setOutputNull(otherArgs.nullVals, data.output1)
        rieglInfo = otherArgs.rieglInfo

        data.output1.setHeaderValue("PULSE_ANGULAR_SPACING_SCANLINE", 
                rieglInfo["PHI_INC"])
        data.output1.setHeaderValue("PULSE_ANGULAR_SPACING_SCANLINE_IDX",
                rieglInfo["THETA_INC"])
        data.output1.setHeaderValue("SENSOR_BEAM_EXIT_DIAMETER",
                rieglInfo["BEAM_EXIT_DIAMETER"])
        data.output1.setHeaderValue("SENSOR_BEAM_DIVERGENCE",
                rieglInfo["BEAM_DIVERGENCE"])

        if otherArgs.epsg is not None:
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(otherArgs.epsg)
            data.output1.setHeaderValue('SPATIAL_REFERENCE', sr.ExportToWkt())
        elif otherArgs.wkt is not None:
            data.output1.setHeaderValue('SPATIAL_REFERENCE', otherArgs.wkt)

        rotationMatrixList = None
        if otherArgs.rotationMatrix is not None:
            rotationMatrixList = otherArgs.rotationMatrix.tolist()

        # Extra Info?? Not sure if this should be handled 
        # as separate fields in the header
        meta = {'Transform': rotationMatrixList,
            'Longitude': rieglInfo['LONGITUDE'],
            'Latitude': rieglInfo['LATITUDE'],
            'Height': rieglInfo['HEIGHT'],
            'HMSL': rieglInfo['HMSL']}
        data.output1.setHeaderValue('USER_META_DATA', json.dumps(meta))

    # check the range
    translatecommon.checkRange(otherArgs.expectRange, points, pulses, 
            waveformInfo)
    # any constant columns
    points, pulses, waveformInfo = translatecommon.addConstCols(otherArgs.constCols,
            points, pulses, waveformInfo)

    data.output1.setPulses(pulses)
    if points is not None:
        data.output1.setPoints(points)
    if waveformInfo is not None:
        data.output1.setWaveformInfo(waveformInfo)
    if recv is not None:
        data.output1.setReceived(recv)

def translate(info, infile, outfile, expectRange=None, scalings=None, 
        internalrotation=False, magneticdeclination=0.0, 
        externalrotationfn=None, nullVals=None, constCols=None, 
        epsg=None, wkt=None):
    """
    Main function which does the work.

    * Info is a fileinfo object for the input file.
    * infile and outfile are paths to the input and output files respectively.
    * expectRange is a list of tuples with (type, varname, min, max).
    * scaling is a list of tuples with (type, varname, gain, offset).
    * if internalrotation is True then the internal rotation will be applied
        to data. Overrides externalrotationfn
    * if externalrotationfn is not None then then the external rotation matrix
        will be read from this file and applied to the data
    * magneticdeclination. If not 0, then this will be applied to the data
    * nullVals is a list of tuples with (type, varname, value)
    * constCols is a list of tupes with (type, varname, dtype, value)
    """
    scalingsDict = translatecommon.overRideDefaultScalings(scalings)

    # set up the variables
    dataFiles = lidarprocessor.DataFiles()
        
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)

    # first set the rotation matrix if asked for
    if internalrotation and externalrotationfn:
        msg = "Can't use both internal and external rotation"
        raise generic.LiDARInvalidSetting(msg)

    rotationMatrix = None
    if internalrotation:
        if "ROTATION_MATRIX" in info.header:
            dataFiles.input1.setLiDARDriverOption("ROTATION_MATRIX", 
                    info.header["ROTATION_MATRIX"])
            rotationMatrix = info.header["ROTATION_MATRIX"]
        else:
            msg = "Internal Rotation requested but no information found in input file"
            raise generic.LiDARInvalidSetting(msg)
    elif externalrotationfn is not None:
        externalrotation = numpy.loadtxt(externalrotationfn, ndmin=2, 
                delimiter=" ", dtype=numpy.float32)            
        dataFiles.input1.setLiDARDriverOption("ROTATION_MATRIX", 
                externalrotation)
        rotationMatrix = externalrotation
            
    # set the magnetic declination if not 0 (the default)
    if magneticdeclination != 0:
        dataFiles.input1.setLiDARDriverOption("MAGNETIC_DECLINATION", 
                magneticdeclination)    

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(False)

    otherArgs = lidarprocessor.OtherArgs()
    # and the header so we don't collect it again
    otherArgs.rieglInfo = info.header
    # also need the default/overriden scaling
    otherArgs.scaling = scalingsDict
    # Add the rotation matrix to otherArgs 
    # for updating the header
    otherArgs.rotationMatrix = rotationMatrix
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

    
