"""
Module for calculating canopy metrics from lidar data.
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
from pylidar.lidarformats import generic
from pylidar import lidarprocessor
from rios import cuiprogress

from pylidar.toolbox.canopy import pavd_calders2014
from pylidar.toolbox.canopy import voxel_hancock2016
from pylidar.toolbox.canopy import pgap_armston2013

DEFAULT_CANOPY_METRIC = "PAVD_CALDERS2014"


class CanopyMetricError(Exception):
    "Exception type for canopymetric errors"



def runCanopyMetric(infiles, outfiles, metric, otherargs):
    """
    Apply canopy metric
    Metric name should be of the form <metric>_<source>
    """
    
    dataFiles = lidarprocessor.DataFiles()
    
    controls = lidarprocessor.Controls()
    
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)    
    
    if metric == "PAVD_CALDERS2014":
        
        dataFiles.inList = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) for fname in infiles]
        otherargs.returnnumcol = []
        otherargs.radians = []
        for i in range( len(dataFiles.inList) ):        
            info = generic.getLidarFileInfo(dataFiles.inList[i].fname)
            if info.getDriverName() == 'riegl':
                if otherargs.externaltransformfn is not None:
                    externaltransform = numpy.loadtxt(otherargs.externaltransformfn[i], ndmin=2, delimiter=" ", dtype=numpy.float32) 
                    dataFiles.inList[i].setLiDARDriverOption("ROTATION_MATRIX", externaltransform)
                elif "ROTATION_MATRIX" in info.header:
                    dataFiles.inList[i].setLiDARDriverOption("ROTATION_MATRIX", info.header["ROTATION_MATRIX"])
                else:
                    msg = 'Input file %s has no valid pitch/roll/yaw data' % dataFiles.inList[i].fname
                    raise generic.LiDARInvalidData(msg)
            if info.getDriverName() == 'SPDV3':
                otherargs.returnnumcol.append('RETURN_ID')
                otherargs.radians.append(True)
            else:
                otherargs.returnnumcol.append('RETURN_NUMBER')
                otherargs.radians.append(False)
        controls.setSpatialProcessing(False)
        controls.setWindowSize(512)
        pavd_calders2014.run_pavd_calders2014(dataFiles, controls, otherargs, outfiles[0])     
    
    elif metric == "VOXEL_HANCOCK2016":              

        for i in range( len(dataFiles.inList) ):        
            info = generic.getLidarFileInfo(dataFiles.inList[i].fname)
            if info.getDriverName() == 'riegl':
                if otherargs.externaltransformfn is not None:
                    externaltransform = numpy.loadtxt(otherargs.externaltransformfn[i], ndmin=2, delimiter=" ", dtype=numpy.float32) 
                    dataFiles.inList[i].setLiDARDriverOption("ROTATION_MATRIX", externaltransform)
                elif "ROTATION_MATRIX" in info.header:
                    dataFiles.inList[i].setLiDARDriverOption("ROTATION_MATRIX", info.header["ROTATION_MATRIX"])
                else:
                    msg = 'Input file %s has no valid pitch/roll/yaw data' % dataFiles.inList[i].fname
                    raise generic.LiDARInvalidData(msg)
            if i == 0:
                if len(info.header["SPATIAL_REFERENCE"]) > 0:
                    otherargs.proj = info.header["SPATIAL_REFERENCE"]
                else:
                    otherargs.proj = None
        
        controls.setSpatialProcessing(False)
        controls.setWindowSize(512)
        voxel_hancock2016.run_voxel_hancock2016(dataFiles, controls, otherargs, outfiles)     
    
    elif metric == "PGAP_ARMSTON2013":
    
        controls.setSpatialProcessing(True)
        controls.setWindowSize(64)
        pgap_armston2013.run_pgap_armston2013(dataFiles, controls, otherargs, outfiles)     
                                                  
    else:
        
        msg = 'Unsupported metric %s' % metric
        raise CanopyMetricError(msg)
        

            
