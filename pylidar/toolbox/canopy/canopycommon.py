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


def prepareInputFiles(infiles, index=None):
    """
    Prepare input files for calculation of canopy metrics
    """    
    dataFiles = lidarprocessor.DataFiles()
    if index is not None:
        dataFiles.inFiles = [lidarprocessor.LidarFile(infiles[index], lidarprocessor.READ)]
    else:
        dataFiles.inFiles = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) for fname in infiles]
    
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
    
    return dataFiles

    
def prepareOtherArgs(infiles, otherargs, index=None):
    """
    Extract file information commonly required for output products and make
    available in otherargs
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

        otherargs.lidardriver.append( info.getDriverName() )
        
        if "SPATIAL_REFERENCE" in info.header.keys():
            if len(info.header["SPATIAL_REFERENCE"]) > 0:
                otherargs.proj.append(info.header["SPATIAL_REFERENCE"])
            else:
                otherargs.proj.append(None)
        else:
            otherargs.proj.append(None)
    
    
def readAllPoints(fns, boundingbox=None, colnames=['X','Y','Z','CLASSIFICATION']):
    """
    Read the requested columns for the points in the given files, in a memory-efficient manner.
    Uses pylidar to read only a block of points at a time, and select out just the
    desired columns. This saves quite a lot of memory, in comparison to reading in all points 
    at once, since all columns for all points have to be read in at the same time.
    
    If boundingbox is given, it is a tuple of
        (xmin, xmax, ymin, ymax)
    and only points within this box are included.

    Return a single recarray with only the selected columns, and only the selected points.
    """
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.inFiles = [lidarprocessor.LidarFile(fn, lidarprocessor.READ) for fn in fns]
    
    otherargs = lidarprocessor.OtherArgs()
    otherargs.colNames = colnames
    otherargs.boundingBox = boundingbox
    otherargs.dataList = []
    
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)
    controls.setWindowSize(2048)
    
    # Read the input files
    lidarprocessor.doProcessing(selectColumns, dataFiles, otherArgs=otherargs, controls=controls)

    # Put all the separate rec-arrays together
    nPoints = sum([a.shape[0] for a in otherargs.dataList])
    if nPoints > 0:
        dataArray = numpy.empty(nPoints, dtype=otherargs.dataList[0].dtype)
        i = 0
        for tmp in otherargs.dataList:
            dataArray[i:i+tmp.shape[0]] = tmp
            i += tmp.shape[0]
    else:
        msg = 'Input files have no valid data'
        raise generic.LiDARInvalidData(msg)
        
    return dataArray


def selectColumns(data, otherargs):
    """
    Read the next block of lidar points, select out the requested columns. If requested,
    filter to ground only. If requested, restrict to the given bounding box.
    """
    pointsList = [inFile.getPoints(colNames=otherargs.colNames) for inFile in data.inFiles]
    nPoints = sum([p.shape[0] for p in pointsList])
    if nPoints > 0:
        points = numpy.empty(nPoints, dtype=pointsList[0].dtype)
        i = 0
        for tmp in pointsList:
            points[i:i+tmp.shape[0]] = tmp
            i += tmp.shape[0]    
    
    if otherargs.boundingBox is not None:
        mask = ((points['X'] >= otherargs.boundingBox[0]) & (points['X'] <= otherargs.boundingBox[1]) & 
                (points['Y'] >= otherargs.boundingBox[2]) & (points['Y'] <= otherargs.boundingBox[3]))
        points = points[mask]

    if points.shape[0] > 0:
        otherargs.dataList.append(points)
