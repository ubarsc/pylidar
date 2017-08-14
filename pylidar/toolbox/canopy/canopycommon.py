"""
Common functions and classes for the canopy module
There is some temporary duplication of functions from the spatial branch
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

import numpy
from osgeo import osr
from osgeo import gdal
from rios import imageio
from rios import calcstats
from rios import pixelgrid
from pylidar.lidarformats import generic
from pylidar import lidarprocessor


class CanopyMetricError(Exception):
    "Exception type for canopymetric errors"


def prepareInputFiles(infiles, otherargs, index=None):
    """
    Prepare input files for calculation of canopy metrics
    """    
    dataFiles = lidarprocessor.DataFiles()
    if index is not None:
        dataFiles.inFiles = [lidarprocessor.LidarFile(infiles[index], lidarprocessor.READ)]
    else:
        dataFiles.inFiles = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) for fname in infiles]
    
    otherargs.lidardriver = []
    otherargs.proj = []
    
    nFiles = len(dataFiles.inFiles)
    for i in range(nFiles):        
        info = generic.getLidarFileInfo(dataFiles.inFiles[i].fname)
        if info.getDriverName() == 'riegl':
            if otherargs.externaltransformfn is not None:
                if index is not None:
                    externaltransform = numpy.loadtxt(otherargs.externaltransformfn[index], ndmin=2, delimiter=" ", dtype=numpy.float32) 
                else:
                    externaltransform = numpy.loadtxt(otherargs.externaltransformfn[i], ndmin=2, delimiter=" ", dtype=numpy.float32)
                dataFiles.inFiles[i].setLiDARDriverOption("ROTATION_MATRIX", externaltransform)
            elif "ROTATION_MATRIX" in info.header:
                dataFiles.inFiles[i].setLiDARDriverOption("ROTATION_MATRIX", info.header["ROTATION_MATRIX"])
            else:
                msg = 'Input file %s has no valid pitch/roll/yaw data' % dataFiles.inFiles[i].fname
                raise generic.LiDARInvalidData(msg)

        otherargs.lidardriver.append( info.getDriverName() )
        
        if "SPATIAL_REFERENCE" in info.header.keys():
            if len(info.header["SPATIAL_REFERENCE"]) > 0:
                otherargs.proj.append(info.header["SPATIAL_REFERENCE"])
            else:
                otherargs.proj.append(None)
        else:
            otherargs.proj.append(None)
    
    return dataFiles
