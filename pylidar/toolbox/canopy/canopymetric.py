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

from pylidar.toolbox import spatial
from pylidar.toolbox.canopy import canopycommon
from pylidar.toolbox.canopy import pavd_calders2014
from pylidar.toolbox.canopy import voxel_hancock2016
from pylidar.toolbox.canopy import pgap_armston2013


def runCanopyMetric(infiles, outfiles, metric, otherargs):
    """
    Apply canopy metric
    Metric name should be of the form <metric>_<source>
    """    
    controls = lidarprocessor.Controls()
    
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)    
    
    if metric == "PAVD_CALDERS2014":
        
        dataFiles = canopycommon.prepareInputFiles(infiles, otherargs)
        if otherargs.externaldem is not None:
            otherargs.dataDem, otherargs.xMinDem, otherargs.yMaxDem, otherargs.binSizeDem = \
                spatial.readImageLayer(otherargs.externaldem)
        controls.setSpatialProcessing(False)
        controls.setWindowSize(512)
        pavd_calders2014.run_pavd_calders2014(dataFiles, controls, otherargs, outfiles[0])     
    
    elif metric == "VOXEL_HANCOCK2016":              
        
        if otherargs.externaldem is not None:
            otherargs.dataDem, otherargs.xMinDem, otherargs.yMaxDem, otherargs.binSizeDem = \
                spatial.readImageLayer(otherargs.externaldem)              
        controls.setSpatialProcessing(False)
        controls.setWindowSize(512)
        voxel_hancock2016.run_voxel_hancock2016(infiles, controls, otherargs, outfiles)     
    
    elif metric == "PGAP_ARMSTON2013":
    
        controls.setSpatialProcessing(True)
        controls.setWindowSize(64)
        pgap_armston2013.run_pgap_armston2013(dataFiles, controls, otherargs, outfiles)     
                                                  
    else:
        
        msg = 'Unsupported metric %s' % metric
        raise canopycommon.CanopyMetricError(msg)
        

            
