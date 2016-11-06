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


def runPavdCalders2014(data, otherargs):
    """
    Calculate PAVD following Calders et al. (2014)
    """
    pointcolnames = [otherargs.heightcol,'CLASSIFICATION','RETURN_NUMBER']
    pulsecolnames = ['NUMBER_OF_RETURNS','ZENITH']
    
    for i,indata in enumerate(data.inList):
        
        points = indata.getPoints(colNames=pointcolnames)
        pulses = indata.getPulses(colNames=pulsecolnames)
        pulsesByPoint = numpy.ma.repeat(pulses, pulses['NUMBER_OF_RETURNS'])
        
        pavd_calders2014.stratifyPointsByZenithHeight(otherargs.zenith,otherargs.minimum_zenith[i],otherargs.maximum_zenith[i],
            otherargs.zenithbinsize,pulses['ZENITH'],pulsesByPoint['ZENITH'],points['RETURN_NUMBER'],pulsesByPoint['NUMBER_OF_RETURNS'],
            points[otherargs.heightcol],otherargs.height,otherargs.heightbinsize,otherargs.counts,otherargs.pulses,otherargs.weighted)
    

def runCanopyMetric(infiles, outfile, metric=DEFAULT_CANOPY_METRIC):
    """
    Apply canopy metric
    Metric name should be of the form <metric>_<source>
    """
    
    # Set all the input files
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.inList = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) 
                        for fname in infiles]
    
    controls = lidarprocessor.Controls()
    otherargs = lidarprocessor.OtherArgs()
    
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)  
    
    if metric == "PAVD_CALDERS2014":
        
        controls.setSpatialProcessing(False)
        controls.setWindowSize(512)
        
        otherargs.weighted = True
                
        otherargs.heightcol = 'Z'
        otherargs.heightbinsize = 0.5
        minheight = -2.0
        maxheight = 50.0               

        otherargs.zenithbinsize = 5.0
        otherargs.minimum_zenith = [35.0,0.0]
        otherargs.maximum_zenith = [70.0,35.0]
        
        minzenith = min(otherargs.minimum_zenith)
        maxzenith = max(otherargs.maximum_zenith)
        
        otherargs.zenith = numpy.arange(minzenith+otherargs.zenithbinsize, maxzenith, otherargs.zenithbinsize) \
                                        + otherargs.zenithbinsize / 2
        otherargs.height = numpy.arange(minheight, maxheight, otherargs.heightbinsize)
        otherargs.counts = numpy.zeros([otherargs.zenith.shape[0],otherargs.height.shape[0]])
        otherargs.pulses = numpy.zeros([otherargs.zenith.shape[0],1])     
        
        lidarprocessor.doProcessing(runPavdCalders2014, dataFiles, controls=controls, 
            otherArgs=otherargs)
        
        pgapz = 1 - numpy.cumsum(otherargs.counts, axis=0) / otherargs.pulses
        zenithRadians = numpy.radians(otherargs.zenith)
        #lpp_pai,lpp_pavd,lpp_mla = pavd_calders2014.calcLinearPlantProfiles(otherargs.height, zenithRadians, pgapz)
        #sapp_pai,sapp_pavd = pavd_calders2014.calcSolidAnglePlantProfiles(otherargs.height, zenithRadians, pgapz, 
        #    otherargs.zenithbinsize)
        
        pavd_calders2014.writePgapProfiles(outfile, otherargs.zenith, otherargs.height, pgapz)
              
    else:
        msg = 'Unsupported metric %s' % metric
        raise CanopyMetricError(msg)
        

            
