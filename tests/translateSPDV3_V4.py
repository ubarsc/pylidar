#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress

MAX_UINT16 = 2**16

def setOutputScaling(header, output):
    xOffset = header['X_MIN']
    xGain = (header['X_MAX'] - header['X_MIN']) / MAX_UINT16
    yOffset = header['Y_MAX']
    yGain = (header['Y_MIN'] - header['Y_MAX']) / MAX_UINT16
    zOffset = header['Z_MIN']
    zGain = (header['Z_MAX'] - header['Z_MIN']) / MAX_UINT16
    
    output.setScaling('X_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, xGain, xOffset)
    output.setScaling('Y_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, yGain, yOffset)
    output.setScaling('Z_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, zGain, zOffset)
    output.setScaling('H_ORIGIN', lidarprocessor.ARRAY_TYPE_PULSES, zGain, zOffset)
    output.setScaling('X_IDX', lidarprocessor.ARRAY_TYPE_PULSES, xGain, xOffset)
    output.setScaling('Y_IDX', lidarprocessor.ARRAY_TYPE_PULSES, yGain, yOffset)
    
    azOffset = header['AZIMUTH_MIN']
    azGain = (header['AZIMUTH_MAX'] - header['AZIMUTH_MIN']) / MAX_UINT16
    output.setScaling('AZIMUTH', lidarprocessor.ARRAY_TYPE_PULSES, azGain, azOffset)
    
    zenOffset = header['ZENITH_MIN']
    zenGain = (header['ZENITH_MAX'] - header['ZENITH_MIN']) / MAX_UINT16
    output.setScaling('ZENITH', lidarprocessor.ARRAY_TYPE_PULSES, zenGain, zenOffset)

    output.setScaling('X', lidarprocessor.ARRAY_TYPE_POINTS, xGain, xOffset)
    output.setScaling('Y', lidarprocessor.ARRAY_TYPE_POINTS, yGain, yOffset)
    output.setScaling('Z', lidarprocessor.ARRAY_TYPE_POINTS, zGain, zOffset)
    output.setScaling('HEIGHT', lidarprocessor.ARRAY_TYPE_POINTS, zGain, zOffset)

def transFunc(data):
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    
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
    
