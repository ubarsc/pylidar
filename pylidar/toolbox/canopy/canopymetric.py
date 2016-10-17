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

import sys
import numpy
import importlib
from pylidar.lidarformats import generic
from pylidar import lidarprocessor
from rios import cuiprogress

from pylidar.toolbox.canopy import pavd_calders2014

DEFAULT_CANOPY_METRIC = "PAVD_CALDERS2014"


class CanopyMetricError(Exception):
    "Exception type for canopymetric errors"


def pavd_calders2014(data, otherargs):
    """
    Calculate PAVD following Calders et al. (2014)
    """
    pointcolnames = ['Z','CLASSIFICATION','RETURN_NUMBER']
    pulsecolnames = ['NUMBER_OF_RETURNS','ZENITH']   
    
    pointsByBin = [indata.getPointsByBins(indexByPulse=True, returnPulseIndex=True, colNames=pointcolnames) for indata in data.inList]
    pulsesByBin = [indata.getPulses(colNames=pulsecolnames, pulseIndex=pointsByBin[i][1]) for i,indata in enumerate(data.inList)]
    
    pointsByBin = numpy.ma.vstack([p[0] for p in pointsByBin])
    pulsesByBin = numpy.ma.vstack(pulsesByBin)
    
    (maxPts, nRows, nCols) = pointsByBin.shape  
    binCoords = data.info.getBlockCoordArrays()
    
    
    sys.exit()


def runCanopyMetric(infiles, outfile, binsize=None, metric=DEFAULT_CANOPY_METRIC, quiet=False):
    """
    Apply canopy metric
    Metric name should be of the form <metric>_<source>
    """
    
    # Set all the input files
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.inList = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) 
                        for fname in infiles]
        
    #info = generic.getLidarFileInfo(infiles[0])
    #print(info.header.keys())
    
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(True)
    if not quiet:
        progress = cuiprogress.GDALProgressBar()
        controls.setProgress(progress)

    if binsize is not None:
        controls.setReferenceResolution(binsize)   
    
    otherArgs = lidarprocessor.OtherArgs()
    
    if metric == "PAVD_CALDERS2014":
        
        heightbinsize = 0.5
        zenithbinsize = 5.0
        
        maxheight = 50.0
        minzenith = 30.0
        maxzenith = 70.0
        
        otherargs.zenith = numpy.arange(minzenith+zenithbinsize, maxzenith, zenithbinsize)
        otherargs.height = numpy.arange(0, maxheight, heightbinsize)
        otherargs.pgap = numpy.ones([otherargs.zenith.size,otherargs.height.size])
        
        controls.setWindowSize(64)
        
      
    else:
        msg = 'Unsupported metric %s' % metric
        raise CanopyMetricError(msg)
    
    metricFunc = eval(metric.lower())
    
    lidarprocessor.doProcessing(metricFunc, dataFiles, controls=controls, 
            otherArgs=otherArgs)
            
