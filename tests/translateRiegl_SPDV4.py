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
import optparse
import numpy
import json
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
        p.add_option("--internalrotation", dest="internalrotation", 
            default=False, action="store_true",
            help="Use information within input file to find instrument rotation information")
        p.add_option("--magneticdeclination", dest="magneticdeclination",
            default=0.0, type="float")
        p.add_option("--riegl", dest="riegl",
            help="input Riegl .rxp file")
        p.add_option("--spd", dest="spd",
            help="output SPD V4 file name")
            
        (options, args) = p.parse_args()
        self.__dict__.update(options.__dict__)

        if self.riegl is None or self.spd is None:
            p.print_help()
            sys.exit()


def rangeFunc(data, rangeDict):
    """
    Called by pylidar and used to determine range for fields
    required by SPD V4 scaling and header fields.
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPoints()
    waveformInfo = data.input1.getWaveformInfo()

    if pulses.size > 0:    
        if 'NUMBER_OF_PULSES' not in rangeDict['pulses']:
            rangeDict['header']['NUMBER_OF_PULSES'] = pulses.size
        else:
            rangeDict['header']['NUMBER_OF_PULSES'] += pulses.size
        for field in spdv4.PULSE_SCALED_FIELDS:
            if field in pulses.dtype.names:
                minKey = field + '_MIN'
                maxKey = field + '_MAX'
                minVal = pulses[field].min()
                maxVal = pulses[field].max()
                if minKey not in rangeDict['pulses'] or minVal < rangeDict['pulses'][minKey]:
                    rangeDict['pulses'][minKey] = minVal
                    if field in ('AZIMUTH', 'ZENITH'):
                        rangeDict['header'][minKey] = minVal
                if maxKey not in rangeDict['pulses'] or maxVal > rangeDict['pulses'][maxKey]:
                    rangeDict['pulses'][maxKey] = maxVal
                    if field in ('AZIMUTH', 'ZENITH'):
                        rangeDict['header'][maxKey] = maxVal
        
    if points.size > 0:    
        if 'NUMBER_OF_POINTS' not in rangeDict['points']:
            rangeDict['header']['NUMBER_OF_POINTS'] = points.size
        else:
            rangeDict['header']['NUMBER_OF_POINTS'] += points.size
        for field in spdv4.POINT_SCALED_FIELDS:
            if field in points.dtype.names:
                minKey = field + '_MIN'
                maxKey = field + '_MAX'
                minVal = points[field].min()
                maxVal = points[field].max()
                if minKey not in rangeDict['points'] or minVal < rangeDict['points'][minKey]:
                    rangeDict['points'][minKey] = minVal
                    rangeDict['header'][minKey] = minVal
                if maxKey not in rangeDict['points'] or maxVal > rangeDict['points'][maxKey]:
                    rangeDict['points'][maxKey] = maxVal
                    rangeDict['header'][maxKey] = maxVal
            
    if waveformInfo is not None:
        if 'NUMBER_OF_WAVEFORMS' not in rangeDict['waveforms']:        
            rangeDict['header']['NUMBER_OF_WAVEFORMS'] = waveformInfo.size
        else:
            rangeDict['header']['NUMBER_OF_WAVEFORMS'] += waveformInfo.size        
        for field in spdv4.WAVEFORM_SCALED_FIELDS:
            if field in waveformInfo.dtype.names:
                minKey = field + '_MIN'
                maxKey = field + '_MAX'
                minVal = waveformInfo[field].min()
                maxVal = waveformInfo[field].max()
                if minKey not in rangeDict['waveforms'] or minVal < rangeDict['waveforms'][minKey]:
                    rangeDict['waveforms'][minKey] = minVal
                if maxKey not in rangeDict['waveforms'] or maxVal > rangeDict['waveforms'][maxKey]:
                    rangeDict['waveforms'][maxKey] = maxVal
    
    

def setOutputScaling(rangeDict, output, usedefaults=False):
    """
    Set the scaling on the output SPD V4 file using info gathered
    by rangeFunc.
    """
    for key in rangeDict['pulses'].keys():
        if key.endswith('_MIN'):
            field = key[0:-4]
            if field in PULSE_DEFAULT_GAINS:
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_PULSES, PULSE_DEFAULT_GAINS[field], PULSE_DEFAULT_OFFSETS[field])
            else:
                minVal = rangeDict['pulses'][key]
                maxVal = rangeDict['pulses'][key.replace('_MIN','_MAX')]            
                gain = np.iinfo(spdv4.PULSE_FIELDS[field]).max / (maxVal - minVal)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_PULSES, gain, minVal)
    for key in rangeDict['points'].keys():
        if key.endswith('_MIN'):
            field = key[0:-4]            
            if field in POINT_DEFAULT_GAINS:
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_POINTS, POINT_DEFAULT_GAINS[field], POINT_DEFAULT_OFFSETS[field])            
            else:
                minVal = rangeDict['points'][key]
                maxVal = rangeDict['points'][key.replace('_MIN','_MAX')]
                gain = np.iinfo(spdv4.POINT_FIELDS[field]).max / (maxVal - minVal)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_POINTS, gain, minVal)
    for key in rangeDict['waveforms'].keys():
        if key.endswith('_MIN'):
            field = key[0:-4]
            if field in WAVEFORM_DEFAULT_GAINS:
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS, WAVEFORM_DEFAULT_GAINS[field], WAVEFORM_DEFAULT_OFFSETS[field])            
            else:
                minVal = rangeDict['waveforms'][key]
                maxVal = rangeDict['waveforms'][key.replace('_MIN','_MAX')]
                gain = np.iinfo(spdv4.WAVEFORM_FIELDS[field]).max / (maxVal - minVal)
                output.setScaling(field, lidarprocessor.ARRAY_TYPE_WAVEFORMS, gain, minVal)



def setHeaderValues(rangeDict, rieglInfo, output):
    """
    Set the header values in the output SPD V4 file using info gathered
    by rangeFunc and the riegl driver. 
    For this test case we assume a RIEGL VZ400
    """
    h = rangeDict['header']
    h["SCANLINE_IDX_MIN"] = rieglInfo["SCANLINE_IDX_MIN"]
    h["SCANLINE_IDX_MAX"] = rieglInfo["SCANLINE_IDX_MAX"]
    h["SCANLINE_MIN"] = rieglInfo["SCANLINE_MIN"]
    h["SCANLINE_MAX"] = rieglInfo["SCANLINE_MAX"]    
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
    Called from pylidar. Does the actual conversion to SPD V4
    """
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    waveformInfo = data.input1.getWaveformInfo()
    revc = data.input1.getReceived()
    
    if points is not None:
        data.output1.translateFieldNames(data.input1, points, 
            lidarprocessor.ARRAY_TYPE_POINTS)
    if pulses is not None:
        data.output1.translateFieldNames(data.input1, pulses, 
            lidarprocessor.ARRAY_TYPE_PULSES)
            
    # set scaling and write header
    if data.info.isFirstBlock():
        print("Setting output scaling and writing header") 
        setOutputScaling(rangeDict, data.output1)
        rieglInfo = data.input1.getHeader()
        setHeaderValues(rangeDict, rieglInfo, data.output1)

    data.output1.setPoints(points)
    data.output1.setPulses(pulses)
    data.output1.setWaveformInfo(waveformInfo)
    data.output1.setReceived(revc)
        
        
def doTranslation(internalrotation, magneticdeclination, riegl, spd):
    """
    Does the translation between Riegl .rxp and SPD v4 format files.
    """
    # set up the variables
    dataFiles = lidarprocessor.DataFiles()
        
    dataFiles.input1 = lidarprocessor.LidarFile(riegl, lidarprocessor.READ)

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setSpatialProcessing(False)
    
    # now read through the file and get the range of values for fields 
    # that need scaling.
    print('Determining range of input data...')
    
    rangeDict = {'pulses':{},'points':{},'waveforms':{},'header':{}}
    lidarprocessor.doProcessing(rangeFunc, dataFiles, controls=controls, 
                    otherArgs=rangeDict)

    print('Converting to SPD V4...')

    dataFiles.output1 = lidarprocessor.LidarFile(spd, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')
    # first get the rotation matrix out of the file if asked for
    if internalrotation:
        print('Obtaining Internal Rotation Matrix...')
        info = generic.getLidarFileInfo(riegl)
        if "ROTATION_MATRIX" in info.header:
            dataFiles.output1.setLiDARDriverOption("ROTATION_MATRIX", 
                    info.header["ROTATION_MATRIX"])
        else:
            msg = "Internal Rotation requested but no information found in input file"
            raise SystemError(msg)
            
    # set the magnetic declination if not 0 (the default)
    if magneticdeclination != 0:
        dataFiles.output1.setLiDARDriverOption("MAGNETIC_DECLINATION", 
                magneticdeclination)

    lidarprocessor.doProcessing(transFunc, dataFiles, controls=controls, 
                    otherArgs=rangeDict)

    
if __name__ == '__main__':

    cmdargs = CmdArgs()
    doTranslation(cmdargs.internalrotation, cmdargs.magneticdeclination,
            cmdargs.riegl, cmdargs.spd)
        
