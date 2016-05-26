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

PULSE_DEFAULT_SCALING = {'X_ORIGIN':(100.0, 0.0), 'Y_ORIGIN':(100.0, 0.0),
    'Z_ORIGIN':(100.0, 0.0), 'H_ORIGIN':(100.0, 0.0), 'AZIMUTH':(100.0, 0.0), 
    'ZENITH':(100.0, 0.0), 'X_IDX':(100.0, 0.0), 'Y_IDX':(100.0, 0.0),
    'AMPLITUDE_PULSE':(100.0, 0.0), 'WIDTH_PULSE':(100.0, 0.0)}

POINT_DEFAULT_SCALING = {'X':(100.0, 0.0), 'Y':(100.0, 0.0), 
    'Z':(100.0, -100.0), 'HEIGHT':(100.0, -100.0), 'INTENSITY':(1.0, 0.0),
    'RANGE':(100.0, 0.0), 'AMPLITUDE_RETURN':(1.0, 0.0),
    'WIDTH_RETURN':(1.0, 0.0)}

WAVEFORM_DEFAULT_SCALING = {'RANGE_TO_WAVEFORM_START':(100.0, 0.0)}

DEFAULT_SCALING = {lidarprocessor.ARRAY_TYPE_PULSES : PULSE_DEFAULT_SCALING, 
    lidarprocessor.ARRAY_TYPE_POINTS : POINT_DEFAULT_SCALING, 
    lidarprocessor.ARRAY_TYPE_WAVEFORMS : WAVEFORM_DEFAULT_SCALING}

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
        for field in spdv4.PULSE_SCALED_FIELDS:
            # this is a field we need to save the scaling for
            if field in pulses.dtype.names:
                minKey = field + '_MIN'
                maxKey = field + '_MAX'
                minVal = pulses[field].min()
                maxVal = pulses[field].max()
                if (minKey not in rangeDict['pulses'] or 
                        minVal < rangeDict['pulses'][minKey]):
                    rangeDict['pulses'][minKey] = minVal
                if (maxKey not in rangeDict['pulses'] or 
                        maxVal > rangeDict['pulses'][maxKey]):
                    rangeDict['pulses'][maxKey] = maxVal
    
    if points.size > 0:
        if 'NUMBER_OF_POINTS' not in rangeDict['header']:
            rangeDict['header']['NUMBER_OF_POINTS'] = points.size
        else:
            rangeDict['header']['NUMBER_OF_POINTS'] += points.size
        for field in spdv4.POINT_SCALED_FIELDS:
            # this is a field we need to save the scaling for
            if field in points.dtype.names:
                minKey = field + '_MIN'
                maxKey = field + '_MAX'
                minVal = points[field].min()
                maxVal = points[field].max()
                if (minKey not in rangeDict['points'] or 
                        minVal < rangeDict['points'][minKey]):
                    rangeDict['points'][minKey] = minVal
                    if minKey in spdv4.HEADER_FIELDS:
                        # update the header while we can
                        rangeDict['header'][minKey] = minVal
                if (maxKey not in rangeDict['points'] or 
                        maxVal > rangeDict['points'][maxKey]):
                    rangeDict['points'][maxKey] = maxVal
                    if maxKey in spdv4.HEADER_FIELDS:
                        # update the header while we can
                        rangeDict['header'][maxKey] = maxVal
    
    if waveformInfo is not None and waveformInfo.size > 0:      
        if 'NUMBER_OF_WAVEFORMS' not in rangeDict['header']:        
            rangeDict['header']['NUMBER_OF_WAVEFORMS'] = waveformInfo.size
        else:
            rangeDict['header']['NUMBER_OF_WAVEFORMS'] += waveformInfo.size        
        for field in spdv4.WAVEFORM_SCALED_FIELDS:
            # this is a field we need to save the scaling for
            if field in waveformInfo.dtype.names:
                minKey = field + '_MIN'
                maxKey = field + '_MAX'
                minVal = waveformInfo[field].min()
                maxVal = waveformInfo[field].max()
                if (minKey not in rangeDict['waveforms'] or 
                        minVal < rangeDict['waveforms'][minKey]):
                    rangeDict['waveforms'][minKey] = minVal
                if (maxKey not in rangeDict['waveforms'] or 
                        maxVal > rangeDict['waveforms'][maxKey]):
                    rangeDict['waveforms'][maxKey] = maxVal

def getRange(infile, spatial=False):
    """
    infile should be lidarprocessor.LidarFile so the user
    can set any driver options etc

    Determines the range of input data. Returns a dictionary
    with 4 keys: 'pulses', 'points', 'waveforms' and 'header'

    Each value is in turn a dictionary. 'header' contains the new
    header based on the range. The other dictionaries are keyed
    on the name of each column (with _MIN and _MAX appended) and
    contain either the minimum or maximum values.
    """
    print('Determining input data range...')
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = infile

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(spatial)

    rangeDict = {'pulses':{},'points':{},'waveforms':{},'header':{}}
    lidarprocessor.doProcessing(rangeFunc, dataFiles, controls=controls, 
                    otherArgs=rangeDict)

    return rangeDict

def setOutputScaling(rangeDict, output):
    """
    Set the scaling on the output SPD V4 file using info gathered
    by rangeFunc. Takes into account default/overridden scaling also.

    Designed to be called from inside a lidarprocessor function so 
    output should be an instance of :class:`pylidar.userclasses.LidarData`.

    rangeDict should be what was returned by getRange().
    """
    scalingDict = rangeDict['scaling']

    # pulses
    for key in rangeDict['pulses'].keys():
        pulseScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_PULSES]
        if key.endswith('_MIN'):
            field = key[0:-4]
            if field in pulseScalingDict:
                gain, offset = pulseScalingDict[field]
                # use the scaling we already have
                checkScalingPositive(gain, offset, 
                        rangeDict['pulses'][key], key.replace('_MIN',''))
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_PULSES, 
                        gain, offset)
            else:
                # extract the _MAX also and set the scaling
                minVal = rangeDict['pulses'][key]
                maxVal = rangeDict['pulses'][key.replace('_MIN','_MAX')]            
                gain = np.iinfo(spdv4.PULSE_FIELDS[field]).max / (maxVal - minVal)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_PULSES, 
                        gain, minVal)

    # points
    for key in rangeDict['points'].keys():
        pointScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_POINTS]
        if key.endswith('_MIN'):
            field = key[0:-4]            
            if field in pointScalingDict:
                gain, offset = pointScalingDict[field]
                # use the scaling we already have
                checkScalingPositive(gain, offset, 
                        rangeDict['points'][key], key.replace('_MIN',''))
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_POINTS, 
                        gain, offset)
            else:
                # extract the _MAX also and set the scaling
                minVal = rangeDict['points'][key]
                maxVal = rangeDict['points'][key.replace('_MIN','_MAX')]
                if maxVal == minVal:
                    gain = 1.0
                else:
                    gain = np.iinfo(spdv4.POINT_FIELDS[field]).max / (maxVal - minVal)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_POINTS, gain, minVal)

    # waveforms
    for key in rangeDict['waveforms'].keys():
        waveformScalingDict = scalingDict[lidarprocessor.ARRAY_TYPE_WAVEFORMS]
        if key.endswith('_MIN'):
            field = key[0:-4]
            if field in waveformScalingDict:
                gain, offset = waveformScalingDict[field]
                # use the scaling we already have
                checkScalingPositive(gain, offset,
                        rangeDict['waveforms'][key], key)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS, 
                        gain, offset)
            else:
                # extract the _MAX also and set the scaling
                minVal = rangeDict['waveforms'][key]
                maxVal = rangeDict['waveforms'][key.replace('_MIN','_MAX')]
                gain = np.iinfo(spdv4.WAVEFORM_FIELDS[field]).max / (maxVal - minVal)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS, 
                    gain, minVal)

def checkScalingPositive(gain, offset, minVal, varName):
    """
    Check that the given gain and offset will not give rise to
    negative values when applied to the minimum value. 
    """
    scaledVal = (minVal - offset) * gain
    if scaledVal < 0:
        msg = ("Scaling for %s gives negative values, " + 
                "which SPD4 will not cope with. Over-ride defaults on " +
                "commandline. ") % varName
        raise generic.LiDARInvalidSetting(msg)

def overRideDefaultScalings(scaling):
    """
    Any scalings given on the commandline should over-ride 
    the default behaviours. if scalings is not None then 
    it is assumed to be a list of tuples with (type, varname, gain, offset).

    Returns a dictionary keyed on lidarprocessor.ARRAY_TYPE_PULSES,
    lidarprocessor.ARRAY_TYPE_POINTS, or lidarprocessor.ARRAY_TYPE_WAVEFORMS.
    Each value in this dictionary is in turn a dictionary keyed on the 
    column name in which each value is a tuple with gain and offset.
    """
    scalingsDict = copy.copy(DEFAULT_SCALING)
    nameToCodeDict = {'POINT':lidarprocessor.ARRAY_TYPE_POINTS,
            'PULSE':lidarprocessor.ARRAY_TYPE_PULSES,
            'WAVEFORM':lidarprocessor.ARRAY_TYPE_WAVEFORMS}

    if scaling is not None:
        for (typeName, varName, gainStr, offsetStr) in scaling:
            gain = float(gainStr)
            offset = float(offsetStr)
            typeName = typeName.upper()
            if typeName not in ['POINT', 'PULSE', 'WAVEFORM']:
                msg = 'unrecognised type %s' % typeName
                raise generic.LiDARInvalidSetting(msg)

            code = nameToCodeDict[typeName]
            scalingsDict[code][varName] = (gain, offset)

    return scalingsDict
