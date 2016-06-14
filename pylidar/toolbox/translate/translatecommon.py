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
from rios import cuiprogress

PULSE_DEFAULT_SCALING = {'X_ORIGIN':[100.0, 0.0], 'Y_ORIGIN':[100.0, 0.0],
    'Z_ORIGIN':[100.0, 0.0], 'H_ORIGIN':[100.0, 0.0], 'AZIMUTH':[100.0, 0.0], 
    'ZENITH':[100.0, 0.0], 'X_IDX':[100.0, 0.0], 'Y_IDX':[100.0, 0.0],
    'AMPLITUDE_PULSE':[100.0, 0.0], 'WIDTH_PULSE':[100.0, 0.0]}
# add in the spdv4 types
for key in PULSE_DEFAULT_SCALING.keys():
    dtype = spdv4.PULSE_FIELDS[key]
    PULSE_DEFAULT_SCALING[key].append(dtype)

POINT_DEFAULT_SCALING = {'X':[100.0, 0.0], 'Y':[100.0, 0.0], 
    'Z':[100.0, -100.0], 'HEIGHT':[100.0, -100.0], 'INTENSITY':[1.0, 0.0],
    'RANGE':[100.0, 0.0], 'AMPLITUDE_RETURN':[1.0, 0.0],
    'WIDTH_RETURN':[1.0, 0.0], 'RHO_APP':[10000.0, 0.0]}
# add in the spdv4 types
for key in POINT_DEFAULT_SCALING.keys():
    dtype = spdv4.POINT_FIELDS[key]
    POINT_DEFAULT_SCALING[key].append(dtype)

WAVEFORM_DEFAULT_SCALING = {'RANGE_TO_WAVEFORM_START':[100.0, 0.0]}
# add in the spdv4 types
for key in WAVEFORM_DEFAULT_SCALING.keys():
    dtype = spdv4.WAVEFORM_FIELDS[key]
    WAVEFORM_DEFAULT_SCALING[key].append(dtype)

DEFAULT_SCALING = {lidarprocessor.ARRAY_TYPE_PULSES : PULSE_DEFAULT_SCALING, 
    lidarprocessor.ARRAY_TYPE_POINTS : POINT_DEFAULT_SCALING, 
    lidarprocessor.ARRAY_TYPE_WAVEFORMS : WAVEFORM_DEFAULT_SCALING}

POINT = 'POINT'
PULSE = 'PULSE'
WAVEFORM = 'WAVEFORM'
NAME_TO_CODE_DICT = {POINT:lidarprocessor.ARRAY_TYPE_POINTS,
            PULSE:lidarprocessor.ARRAY_TYPE_PULSES,
            WAVEFORM:lidarprocessor.ARRAY_TYPE_WAVEFORMS}

"String to numpy dtype dictionary"
STRING_TO_DTYPE = {'INT8':numpy.int8, 'UINT8':numpy.uint8, 'INT16':numpy.int16,
    'UINT16':numpy.uint16, 'INT32':numpy.int32, 'UINT32':numpy.uint32,
    'INT64':numpy.int64, 'UINT64':numpy.uint64, 'FLOAT32':numpy.float32,
    'FLOAT64':numpy.float64}
"Code that means: use the default type"
DEFAULT_DTYPE_STR = 'DFLT'

def rangeFunc(data, rangeDict):
    """
    Called by lidaprocessor() via getRange() and used to determine range for 
    fields required by SPD V4 scaling and header fields.
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPoints()
    waveformInfo = data.input1.getWaveformInfo()

    if pulses.size > 0:
        if 'NUMBER_OF_PULSES' not in rangeDict['header']:
            rangeDict['header']['NUMBER_OF_PULSES'] = pulses.size
        else:
            rangeDict['header']['NUMBER_OF_PULSES'] += pulses.size
        for field in pulses.dtype.names:
            minKey = field + '_MIN'
            maxKey = field + '_MAX'
            minVal = pulses[field].min()
            maxVal = pulses[field].max()
            if (minKey not in rangeDict[PULSE] or 
                    minVal < rangeDict[PULSE][minKey]):
                rangeDict[PULSE][minKey] = minVal
                if minKey in spdv4.HEADER_FIELDS:
                    # update the header while we can
                    rangeDict['header'][minKey] = minVal
            if (maxKey not in rangeDict[PULSE] or 
                    maxVal > rangeDict[PULSE][maxKey]):
                rangeDict[PULSE][maxKey] = maxVal
                if maxKey in spdv4.HEADER_FIELDS:
                    # update the header while we can
                    rangeDict['header'][maxKey] = maxVal
    
    if points.size > 0:
        if 'NUMBER_OF_POINTS' not in rangeDict['header']:
            rangeDict['header']['NUMBER_OF_POINTS'] = points.size
        else:
            rangeDict['header']['NUMBER_OF_POINTS'] += points.size
        for field in points.dtype.names:
            minKey = field + '_MIN'
            maxKey = field + '_MAX'
            minVal = points[field].min()
            maxVal = points[field].max()
            if (minKey not in rangeDict[POINT] or 
                    minVal < rangeDict[POINT][minKey]):
                rangeDict[POINT][minKey] = minVal
                if minKey in spdv4.HEADER_FIELDS:
                    # update the header while we can
                    rangeDict['header'][minKey] = minVal
            if (maxKey not in rangeDict[POINT] or 
                    maxVal > rangeDict[POINT][maxKey]):
                rangeDict[POINT][maxKey] = maxVal
                if maxKey in spdv4.HEADER_FIELDS:
                    # update the header while we can
                    rangeDict['header'][maxKey] = maxVal
    
    if waveformInfo is not None and waveformInfo.size > 0:      
        if 'NUMBER_OF_WAVEFORMS' not in rangeDict['header']:        
            rangeDict['header']['NUMBER_OF_WAVEFORMS'] = waveformInfo.size
        else:
            rangeDict['header']['NUMBER_OF_WAVEFORMS'] += waveformInfo.size        
        for field in waveformInfo.dtype.names:
            minKey = field + '_MIN'
            maxKey = field + '_MAX'
            minVal = waveformInfo[field].min()
            maxVal = waveformInfo[field].max()
            if (minKey not in rangeDict[WAVEFORM] or 
                    minVal < rangeDict[WAVEFORM][minKey]):
                rangeDict[WAVEFORM][minKey] = minVal
            if (maxKey not in rangeDict[WAVEFORM] or 
                    maxVal > rangeDict[WAVEFORM][maxKey]):
                rangeDict[WAVEFORM][maxKey] = maxVal

def getRange(infile, controls=None, expectRange=None):
    """
    infile should be lidarprocessor.LidarFile so the user
    can set any driver options etc

    Determines the range of input data. Returns a dictionary
    with 4 keys: PULSE, POINT, WAVEFORM and 'header'

    Each value is in turn a dictionary. 'header' contains the new
    SPDV4 header based on the range. The other dictionaries are keyed
    on the name of each column (with _MIN and _MAX appended) and
    contain either the minimum or maximum values.

    If controls is not None then is should be a reference to a 
    lidarprocessor.Controls object. This will ensure that the file
    is processed in the same way as the translation. In theory
    it shouldn't matter, but spatial processing may not read all the 
    points/pulses in the file because of the way the spatial index
    is calculated so we allow this to be set in the same way the
    file will be translated.

    If expectRange is not None it is interpreted as a list of tuples.
    Each tuple should be (type, varName, min, max). type should be 
    one of 'POINT', 'PULSE', 'WAVEFORM', varName will be the variable
    name and min and max are interpreted to me the minimum and maximum
    values to be expected in the data. If values are found outside of 
    this then an error is raised.
    """
    print('Determining input data range...')
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = infile

    rangeDict = {PULSE:{}, POINT:{}, WAVEFORM:{}, 'header':{}}
    lidarprocessor.doProcessing(rangeFunc, dataFiles, controls=controls, 
                    otherArgs=rangeDict)

    if expectRange is not None:
        for (typeName, varName, minStr, maxStr) in expectRange:
            minVal = float(minStr)
            maxVal = float(maxStr)
            typeName = typeName.upper()
            if typeName not in NAME_TO_CODE_DICT.keys():
                msg = 'unrecognised type %s' % typeName
                raise generic.LiDARInvalidSetting(msg)

            minKey = varName + '_MIN'
            maxKey = varName + '_MAX'
            typeRange = rangeDict[typeName]
            if minKey not in typeRange or maxKey not in typeRange:
                msg = 'variable %s not found in file' % varName
                raise generic.LiDARInvalidSetting(msg)

            if typeRange[minKey] < minVal or typeRange[maxKey] > maxVal:
                msg = 'data for %s outside of expected range' % varName
                raise generic.LiDARInvalidData(msg)

    return rangeDict

def setOutputScaling(rangeDict, output):
    """
    Set the scaling on the output SPD V4 file using info gathered
    by rangeFunc. Takes into account default/overridden scaling also.

    Designed to be called from inside a lidarprocessor function so 
    output should be an instance of :class:`pylidar.userclasses.LidarData`.

    rangeDict should be what was returned by getRange() with an extra
    key 'scaling' which should be what was returned from 
    overRideDefaultScalings().
    """
    scalingDict = rangeDict['scaling']

    # pulses
    for key in rangeDict[PULSE].keys():
        pulseScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_PULSES]
        if key.endswith('_MIN'):
            field = key[0:-4]
            if field in pulseScalingDict:
                gain, offset, dtype = pulseScalingDict[field]
                minVal = rangeDict[PULSE][key]
                maxVal = rangeDict[PULSE][key.replace('_MIN','_MAX')]
                name = key.replace('_MIN','')
                checkScaling(gain, offset, dtype, minVal, maxVal, name)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_PULSES, 
                        gain, offset)
                output.setNativeDataType(field, 
                        lidarprocessor.ARRAY_TYPE_PULSES, dtype)

    # points
    for key in rangeDict[POINT].keys():
        pointScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_POINTS]
        if key.endswith('_MIN'):
            field = key[0:-4]            
            if field in pointScalingDict:
                gain, offset, dtype = pointScalingDict[field]
                minVal = rangeDict[POINT][key]
                maxVal = rangeDict[POINT][key.replace('_MIN','_MAX')]
                name = key.replace('_MIN','')
                checkScaling(gain, offset, dtype, minVal, maxVal, name)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_POINTS, 
                        gain, offset)
                output.setNativeDataType(field, 
                        lidarprocessor.ARRAY_TYPE_POINTS, dtype)

    # waveforms
    for key in rangeDict[WAVEFORM].keys():
        waveformScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_WAVEFORMS]
        if key.endswith('_MIN'):
            field = key[0:-4]
            if field in waveformScalingDict:
                gain, offset, dtype = waveformScalingDict[field]
                minVal = rangeDict[WAVEFORM][key]
                maxVal = rangeDict[WAVEFORM][key.replace('_MIN','_MAX')]
                name = key.replace('_MIN','')
                checkScaling(gain, offset, dtype, minVal, maxVal, name)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS, 
                        gain, offset)
                output.setNativeDataType(field, 
                        lidarprocessor.ARRAY_TYPE_WAVEFORMS, dtype)

def checkScaling(gain, offset, dtype, minVal, maxVal, varName):
    """
    Check that the given gain and offset will not overflow or underflow
    the given datatype
    """
    if numpy.issubdtype(dtype, numpy.integer):
        info = numpy.iinfo(dtype)
        scaledMin = (minVal - offset) * gain
        scaledMax = (maxVal - offset) * gain

        if scaledMin < info.min:
            msg = ("Scaling for %s gives values less than %d which is the "+
                "minimum value allowed by the data type. Over-ride defaults " +
                "on the command line.") % (varName, info.min)
            raise generic.LiDARInvalidSetting(msg)
        if scaledMax > info.max:
            msg = ("Scaling for %s gives values greater than %d which is the "+
                "maximum value allowed by the data type. Over-ride defaults " +
                "on the command line.") % (varName, info.max)
            raise generic.LiDARInvalidSetting(msg)

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
    scalingsDict = copy.copy(DEFAULT_SCALING)

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
