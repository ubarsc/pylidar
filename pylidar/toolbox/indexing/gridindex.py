
"""
Deals with creating a grid spatial index
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

import os
import copy
import tempfile
from pylidar import lidarprocessor
from pylidar.lidarformats import spdv4
from pylidar.lidarformats import generic
from pylidar.basedriver import Extent

def createGridSpatialIndex(infile, outfile, binSize=1.0, blockSize=200, 
        tempDir='.', extent=None, indexMethod=spdv4.SPDV4_INDEX_CARTESIAN):
    """
    Creates a grid spatially indexed file from a non spatial input file.
    Currently only supports creation of a SPD V4 file.
    
    Creates a tempfile for every block and them merges them into the output.
    
    """
    if extent is None:
        # work out from header
        info = generic.getLidarFileInfo(infile)
        try:
            if indexMethod == spdv4.SPDV4_INDEX_CARTESIAN:
                xMax = info.header['X_MAX']
                xMin = info.header['Y_MIN']
                yMax = info.header['Y_MAX']
                yMin = info.header['Y_MIN']
            elif indexMethod == spdv4.SPDV4_INDEX_SPHERICAL:
                xMax = info.header['AZIMUTH_MAX']
                xMin = info.header['AZIMUTH_MIN']
                yMax = info.header['ZENITH_MAX']
                yMin = info.header['ZENITH_MIN']
            elif indexMethod == spdv4.SPDV4_INDEX_SCAN:
                xMax = info.header['SCANLINE_MAX']
                xMin = info.header['SCANLINE_MIN']
                yMax = info.header['SCANLINE_IDX_MAX']
                yMin = info.header['SCANLINE_IDX_MIN']
            else:
                msg = 'unsupported indexing method'
                raise generic.LiDARSpatialIndexNotAvailable(msg)
        except KeyError:
            msg = 'info for creating bounding box not available'
            raise generic.LiDARFunctionUnsupported(msg)
            
        extent = Extent(xMin, xMax, yMin, yMax, binSize)
    else:
        # ensure that our binSize comes from their exent
        binSize = extent.binSize
    
    extentList = []
    subExtent = Extent(xMin, xMin + blockSize, yMax - blockSize, yMax, binSize)
    controls = lidarprocessor.Controls()
        
    bMoreToDo = True
    while bMoreToDo:
        fd, fname = tempfile.mkstemp(suffix='spdv4', dir=tempDir)
        os.close(fd)
        
        userClass = lidarprocessor.LidarFile(fname, generic.CREATE)
        driver = spdv4.SPDV4File(fname, generic.CREATE, controls, userClass)
        
        data = (copy.copy(subExtent), driver)
        extentList.append(data)

        # move it along
        subExtent.xMin += blockSize
        subExtent.xMax += blockSize

        if subExtent.xMin >= extent.xMax:
            # next line down
            subExtent.xMin = extent.xMin
            subExtent.xMax = extent.xMin + blockSize
            subExtent.yMax -= blockSize
            subExtent.yMin -= blockSize
            
        # done?
        bMoreToDo = subExtent.yMax > extent.yMin
        