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
import importlib
from pylidar import lidarprocessor
from pylidar.toolbox.canopy import canopymetric
from rios import cuiprogress

DEFAULT_METRIC = "pavd_jupp2009"




class CanopyMetricError(Exception):
    "Exception type for canopymetric errors"


def pavd_jupp2009(data, otherargs):
	"""
	Calculate PAVD using Jupp et al. (2009)
	"""
    
    ptValList = [indata.getPoints(colNames=['HEIGHT','CLASSIFICATION','RETURN_NUMBER']) for indata in data.inList]
    ptValStack = numpy.hstack(ptValList)
    
    binCoords = data.info.getBlockCoordArrays()
    
    print(binCoords)
    sys.exit()
	
	
def runCanopyMetric(infiles, outfile, binsize=None, metric=DEFAULT_METRIC, quiet=False):
    """
    Apply canopy metric
    Metric name should be of the form <metric>_<source>
    """
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.inList = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) 
                        for fname in infiles]
    
	otherArgs = lidarprocessor.OtherArgs()
	
	if metric == "pavd_jupp2009":
		
        otherArgs.heightbinsize = 0.5
        
				
	else:
		msg = 'Unsupported metric %s' % metric
        raise CanopyMetricError(msg)	
	
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(True)
    if not quiet:
        progress = cuiprogress.GDALProgressBar()
        controls.setProgress(progress)

    if binsize is not None:
        controls.setReferenceResolution(binsize)
	
	metricFunc = eval(metric)
	
    lidarprocessor.doProcessing(metricFunc, dataFiles, controls=controls, 
            otherArgs=otherArgs)
