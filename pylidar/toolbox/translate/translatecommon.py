"""
Common data and functions for translation
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
import numpy
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from pylidar.toolbox import arrayutils
from rios import cuiprogress

PULSE_DEFAULT_SCALING = {'X_ORIGIN':[100.0, 0.0], 'Y_ORIGIN':[100.0, 0.0],
    'Z_ORIGIN':[100.0, 0.0], 'H_ORIGIN':[100.0, 0.0], 'AZIMUTH':[100.0, 0.0], 
    'ZENITH':[100.0, 0.0], 'X_IDX':[100.0, 0.0], 'Y_IDX':[100.0, 0.0],
    'AMPLITUDE_PULSE':[100.0, 0.0], 'WIDTH_PULSE':[100.0, 0.0]}
"Default scaling for pulses"
# add in the spdv4 types
for key in PULSE_DEFAULT_SCALING.keys():
    dtype = spdv4.PULSE_FIELDS[key]
    PULSE_DEFAULT_SCALING[key].append(dtype)

POINT_DEFAULT_SCALING = {'X':[100.0, 0.0], 'Y':[100.0, 0.0], 
    'Z':[100.0, -100.0], 'HEIGHT':[100.0, -100.0], 'INTENSITY':[1.0, 0.0],
    'RANGE':[100.0, 0.0], 'AMPLITUDE_RETURN':[1.0, 0.0],
    'WIDTH_RETURN':[1.0, 0.0], 'RHO_APP':[10000.0, 0.0]}
"default scaling for points"
# add in the spdv4 types
for key in POINT_DEFAULT_SCALING.keys():
    dtype = spdv4.POINT_FIELDS[key]
    POINT_DEFAULT_SCALING[key].append(dtype)

WAVEFORM_DEFAULT_SCALING = {'RANGE_TO_WAVEFORM_START':[100.0, 0.0]}
"default scaling for waveforms"
# add in the spdv4 types
for key in WAVEFORM_DEFAULT_SCALING.keys():
    dtype = spdv4.WAVEFORM_FIELDS[key]
    WAVEFORM_DEFAULT_SCALING[key].append(dtype)

DEFAULT_SCALING = {lidarprocessor.ARRAY_TYPE_PULSES : PULSE_DEFAULT_SCALING, 
    lidarprocessor.ARRAY_TYPE_POINTS : POINT_DEFAULT_SCALING, 
    lidarprocessor.ARRAY_TYPE_WAVEFORMS : WAVEFORM_DEFAULT_SCALING}
"all the default scalings as a dictionary"

HEADER = 'header'

POINT = 'POINT'
PULSE = 'PULSE'
WAVEFORM = 'WAVEFORM'
NAME_TO_CODE_DICT = {POINT:lidarprocessor.ARRAY_TYPE_POINTS,
            PULSE:lidarprocessor.ARRAY_TYPE_PULSES,
            WAVEFORM:lidarprocessor.ARRAY_TYPE_WAVEFORMS}

STRING_TO_DTYPE = {'INT8':numpy.int8, 'UINT8':numpy.uint8, 'INT16':numpy.int16,
    'UINT16':numpy.uint16, 'INT32':numpy.int32, 'UINT32':numpy.uint32,
    'INT64':numpy.int64, 'UINT64':numpy.uint64, 'FLOAT32':numpy.float32,
    'FLOAT64':numpy.float64}
"String to numpy dtype dictionary"
DEFAULT_DTYPE_STR = 'DFLT'
"Code that means: use the default type"

def checkRange(expectRange, points, pulses, waveforms=None):
    """
    Checks the expected range against the data that has been
    passed. Raises an exception if data is outside of range

    * expectRange is a list of tuples with (type, varname, min, max).
    * points, pulses and waveforms are the arrays to check
    """
    if expectRange is None:
        return

    for dataType, varName, minVal, maxVal in expectRange:
        if dataType == POINT:
            arr = points
        elif dataType == PULSE:
            arr = pulses
        elif dataType == WAVEFORM:
            arr = waveforms
        else:
            msg = 'Unknown data type %s' % dataType
            raise generic.LiDARInvalidSetting(msg)

        if arr is None:
            continue

        if varName not in arr.dtype.fields:
            msg = 'Could not find field %s in input data' % varName
            raise generic.LiDARInvalidData(msg)

        dataMin = arr[varName].min()
        if dataMin < float(minVal):
            msg = 'Minimum value found (%f) less than range minimum (%s)'
            msg = msg % (dataMin, minVal)
            raise generic.LiDARInvalidData(msg)

        dataMax = arr[varName].max()
        if dataMax > float(maxVal):
            msg = 'Maximum value found (%f) greater than range minimum (%s)'
            msg = msg % (dataMax, maxVal)
            raise generic.LiDARInvalidData(msg)
        
def setOutputScaling(scalingDict, output):
    """
    Set the scaling on the output SPD V4 file.

    Designed to be called from inside a lidarprocessor function so 
    output should be an instance of :class:`pylidar.userclasses.LidarData`.

    scalingDict should be what was returned by  
    overRideDefaultScalings().
    """
    # pulses
    pulseScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_PULSES]
    for field in pulseScalingDict:
        gain, offset, dtype = pulseScalingDict[field]
        output.setScaling(field, lidarprocessor.ARRAY_TYPE_PULSES, 
                gain, offset)
        output.setNativeDataType(field, 
                lidarprocessor.ARRAY_TYPE_PULSES, dtype)

    # points
    pointScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_POINTS]
    for field in pointScalingDict:
        gain, offset, dtype = pointScalingDict[field]
        output.setScaling(field, lidarprocessor.ARRAY_TYPE_POINTS, 
                gain, offset)
        output.setNativeDataType(field, 
                lidarprocessor.ARRAY_TYPE_POINTS, dtype)

    # waveforms
    waveformScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_WAVEFORMS]
    for field in waveformScalingDict:
        gain, offset, dtype = waveformScalingDict[field]
        output.setScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS, 
                gain, offset)
        output.setNativeDataType(field, 
                lidarprocessor.ARRAY_TYPE_WAVEFORMS, dtype)

def setOutputNull(nullVals, output):
    """
    Set the null values.

    nullVals should be a list of (type, varname, value) tuples

    Designed to be called from inside a lidarprocessor function so 
    output should be an instance of :class:`pylidar.userclasses.LidarData`.
    """
    if nullVals is None:
        return

    for typeName, varName, value in nullVals:
        value = float(value)

        if typeName not in NAME_TO_CODE_DICT:
            msg = 'unrecognised type %s' % typeName
            raise generic.LiDARInvalidSetting(msg)
        code = NAME_TO_CODE_DICT[typeName]

        output.setNullValue(varName, code, value)

def addConstCols(constCols, points, pulses, waveforms=None):
    """
    Add constant columns to points, pulses or waveforms

    constCols is a list of tupes with (type, varname, dtype, value)
    """
    if constCols is not None:
        for typeName, varName, dtypeName, value in constCols:

            if dtypeName not in STRING_TO_DTYPE:
                msg = "unrecognised dtype string %s" % dtypeName
                raise generic.LiDARInvalidSetting(msg)

            dtypeCode = STRING_TO_DTYPE[dtypeName]
            # I *think* this is safe since we set the dtype explicitly
            value = float(value)

            # TODO: what to do if array is None??
            if typeName == PULSE:
                pulses = arrayutils.addFieldToStructArray(pulses, varName, 
                            dtypeCode, value)
            elif typeName == POINT:
                points = arrayutils.addFieldToStructArray(points, varName, 
                            dtypeCode, value)
            elif typeName == WAVEFORM:
                points = arrayutils.addFieldToStructArray(points, varName, 
                            dtypeCode, value)
            else:
                msg = "unrecognised type %s" % typeName
                raise generic.LiDARInvalidSetting(msg)

    return points, pulses, waveforms

def overRideDefaultScalings(scaling):
    """
    Any scalings given on the commandline should over-ride 
    the default behaviours. if scalings is not None then 
    it is assumed to be a list of tuples with (type, varname, type, gain, offset).

    Returns a dictionary keyed on lidarprocessor.ARRAY_TYPE_PULSES,
    lidarprocessor.ARRAY_TYPE_POINTS, or lidarprocessor.ARRAY_TYPE_WAVEFORMS.
    Each value in this dictionary is in turn a dictionary keyed on the 
    column name in which each value is a tuple with gain, offset and dtype.
    """
    scalingsDict = copy.deepcopy(DEFAULT_SCALING)

    if scaling is not None:
        for (typeName, varName, dtypeName, gainStr, offsetStr) in scaling:
            gain = float(gainStr)
            offset = float(offsetStr)
            typeName = typeName.upper()
            dtypeName = dtypeName.upper()

            if typeName not in NAME_TO_CODE_DICT:
                msg = 'unrecognised type %s' % typeName
                raise generic.LiDARInvalidSetting(msg)
            code = NAME_TO_CODE_DICT[typeName]

            if dtypeName == DEFAULT_DTYPE_STR:
                # look up spdv4 to see what it uses
                SPDV4fieldsdict = {lidarprocessor.ARRAY_TYPE_POINTS:spdv4.POINT_FIELDS,
                    lidarprocessor.ARRAY_TYPE_PULSES:spdv4.PULSE_FIELDS,
                    lidarprocessor.ARRAY_TYPE_WAVEFORMS:spdv4.WAVEFORM_FIELDS}
                        
                SPDV4fields = SPDV4fieldsdict[code]
                if varName not in SPDV4fields:
                    msg = '%s is not a SPDV4 standard column for %s'
                    msg = msg % (varName, typeName)
                    raise generic.LiDARInvalidSetting(msg)

                dtype = SPDV4fields[varName]

            else:
                if dtypeName not in STRING_TO_DTYPE:
                    msg = 'unrecognised data type %s' % dtypeName
                    raise generic.LiDARInvalidSetting(msg)

                dtype = STRING_TO_DTYPE[dtypeName]

            scalingsDict[code][varName] = (gain, offset, dtype)

    return scalingsDict
