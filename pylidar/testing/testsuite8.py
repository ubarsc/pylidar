
"""
Simple testsuite that checks we can import a Riegl file
and create a spatial index and image from it. 
Also tests updating resulting file.
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

import os
import numpy
import shutil
from . import utils
from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.toolbox.translate.riegl2spdv4 import translate
from pylidar.toolbox.indexing.gridindex import createGridSpatialIndex
from pylidar.toolbox.rasterization import rasterize
from pylidar.toolbox.arrayutils import convertArgResultToIndexTuple
from rios import cuiprogress

REQUIRED_FORMATS = ["RIEGL"]

INPUT_RIEGL = '161122_092408.rxp'
IMPORTED_SPD = 'testsuite8.spd'
INDEXED_SPD = 'testsuite8_idx.spd'
OUTPUT_RASTER = 'testsuite8.img'
UPDATED_SPD = 'testsuite8_update.spd'

# override the default scaling
# type, varname, gain, offset
SCALINGS = [('PULSE', 'X_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'Y_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'Z_ORIGIN', 'DFLT', 1000, -1000),
('PULSE', 'X_IDX', 'DFLT', 1000, -1000),
('PULSE', 'Y_IDX', 'DFLT', 1000, -1000),
('POINT', 'X', 'DFLT', 1000, -1000),
('POINT', 'Y', 'DFLT', 1000, -1000),
('POINT', 'Z', 'DFLT', 1000, -1000)]

# because the files have lots of points, use a smaller
# windowsize to prevent running out of memory
WINDOWSIZE = 50

def updatePointFunc(data):
    """
    Function called from the processor that updates the points.
    I have taken the opportunity to test the convertArgResultToIndexTuple
    function which seems the best way to update with the results of
    argmin.
    """
    pts = data.input1.getPointsByBins(colNames=['Z', 'CLASSIFICATION'])
    zVals = pts['Z']
    classif = pts['CLASSIFICATION']
    if pts.shape[0] > 0:
        idx = numpy.argmin(zVals, axis=0)
        idxmask = numpy.ma.all(zVals, axis=0)
        z, y, x = convertArgResultToIndexTuple(idx, idxmask.mask)

        # set lowest points to class 2        
        classif[z, y, x] = 2
    
        data.input1.setPoints(classif, colName='CLASSIFICATION')

def run(oldpath, newpath):
    """
    Runs the 8th basic test suite. Tests:

    Importing Riegl
    Creating spatial index
    Create an image file
    updating resulting file
    """
    inputRiegl = os.path.join(oldpath, INPUT_RIEGL)
    info = generic.getLidarFileInfo(inputRiegl)

    importedSPD = os.path.join(newpath, IMPORTED_SPD)
    translate(info, inputRiegl, importedSPD, scalings=SCALINGS, 
            internalrotation=True)
    utils.compareLiDARFiles(os.path.join(oldpath, IMPORTED_SPD), importedSPD,
            windowSize=WINDOWSIZE)

    indexedSPD = os.path.join(newpath, INDEXED_SPD)
    createGridSpatialIndex(importedSPD, indexedSPD, binSize=1.0,
            tempDir=newpath)
    utils.compareLiDARFiles(os.path.join(oldpath, INDEXED_SPD), indexedSPD,
            windowSize=WINDOWSIZE)

    outputRaster = os.path.join(newpath, OUTPUT_RASTER)
    rasterize([indexedSPD], outputRaster, ['Z'], function="numpy.ma.min", 
            atype='POINT', windowSize=WINDOWSIZE)
    utils.compareImageFiles(os.path.join(oldpath, OUTPUT_RASTER), outputRaster)

    outputUpdate = os.path.join(newpath, UPDATED_SPD)
    shutil.copyfile(indexedSPD, outputUpdate)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = lidarprocessor.LidarFile(outputUpdate, 
                    lidarprocessor.UPDATE)
    
    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setWindowSize(WINDOWSIZE)
    controls.setSpatialProcessing(True)
    
    lidarprocessor.doProcessing(updatePointFunc, dataFiles, controls=controls)

    utils.compareLiDARFiles(os.path.join(oldpath, UPDATED_SPD), outputUpdate,
            windowSize=WINDOWSIZE)
