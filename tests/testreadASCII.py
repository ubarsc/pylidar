#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
from pylidar import lidarprocessor

def readFunc(data):
    pulses = data.input1.getPulses()
    print('pulses', len(pulses))
    points = data.input1.getPoints()
    print('points', len(points))
    
def testRead(infile):
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)

    colTypes = [('GPS_TIME', numpy.float64), ("X_IDX", numpy.float64),
        ("Y_IDX", numpy.float64), ("Z_IDX", numpy.float64),
        ("X", numpy.float64), ("Y", numpy.float64), ("Z", numpy.float64),
        ("CLASSIFICATION", numpy.uint8), ("ORIG_RETURN_NUMBER", numpy.uint8),
        ("ORIG_NUMBER_OF_RETURNS", numpy.uint8), ("AMPLITUDE", numpy.float64),
        ("FWHM", numpy.float64), ("RANGE", numpy.float64)]
    dataFiles.input1.setLiDARDriverOption('COL_TYPES', colTypes)

    pulseCols = ['GPS_TIME', "X_IDX", "Y_IDX", "Z_IDX"]
    dataFiles.input1.setLiDARDriverOption('PULSE_COLS', pulseCols)

    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)
    
    lidarprocessor.doProcessing(readFunc, dataFiles, controls=controls)
    
if __name__ == '__main__':
    testRead(sys.argv[1])
        
