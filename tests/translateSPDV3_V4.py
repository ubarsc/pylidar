#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor

def transFunc(data):
    pulses = data.input1.getPulses()
    points = data.input1.getPointsByPulse()
    
    data.output1.translateFieldNames(data.input1, points, 
            lidarprocessor.ARRAY_TYPE_POINTS)
    data.output1.translateFieldNames(data.input1, pulses, 
            lidarprocessor.ARRAY_TYPE_PULSES)
    
    data.output1.setPoints(points)
    data.output1.setPulses(pulses)
    
def testConvert(infile, outfile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.output1 = lidarprocessor.LidarFile(outfile, lidarprocessor.CREATE)
    dataFiles.output1.setLiDARDriver('SPDV4')
    
    lidarprocessor.doProcessing(transFunc, dataFiles)
    
if __name__ == '__main__':
    testConvert(sys.argv[1], sys.argv[2])
    
