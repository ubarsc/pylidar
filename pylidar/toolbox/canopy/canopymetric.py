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
from pylidar.lidarformats import generic
from pylidar import lidarprocessor
from rios import cuiprogress

from pylidar.toolbox.canopy import pavd_calders2014

DEFAULT_CANOPY_METRIC = "PAVD_CALDERS2014"


class CanopyMetricError(Exception):
    "Exception type for canopymetric errors"



def runCanopyMetric(infiles, outfile, metric, otherargs):
    """
    Apply canopy metric
    Metric name should be of the form <metric>_<source>
    """
    
    # Set all the input files
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.inList = [lidarprocessor.LidarFile(fname, lidarprocessor.READ) for fname in infiles]
    
    controls = lidarprocessor.Controls()
    
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)  
    
    if metric == "PAVD_CALDERS2014":
        
        otherargs.returnnumcol = []
        otherargs.radians = []
        for i in range( len(dataFiles.inList) ):        
            info = generic.getLidarFileInfo(infiles[i])
            if info.getDriverName() == 'riegl':
                if "ROTATION_MATRIX" in info.header:
                    dataFiles.inList[i].setLiDARDriverOption("ROTATION_MATRIX", info.header["ROTATION_MATRIX"])
                else:
                    msg = 'Input file %s has no valid pitch/roll/yaw data' % infiles[i]
                    raise CanopyMetricError(msg)
            if info.getDriverName() == 'SPDV3':
                otherargs.returnnumcol.append('RETURN_ID')
                otherargs.radians.append(True)
            else:
                otherargs.returnnumcol.append('RETURN_NUMBER')
                otherargs.radians.append(False)
                
        controls.setSpatialProcessing(False)
        controls.setWindowSize(512)
        
        if otherargs.planecorrection:  
            print("Applying plane correction to point heights...")
            
            otherargs.xgrid = numpy.zeros(otherargs.gridsize**2, dtype=numpy.float64)
            otherargs.ygrid = numpy.zeros(otherargs.gridsize**2, dtype=numpy.float64)
            otherargs.zgrid = numpy.zeros(otherargs.gridsize**2, dtype=numpy.float64)
            otherargs.gridmask = numpy.ones(otherargs.gridsize**2, dtype=numpy.bool)
                       
            lidarprocessor.doProcessing(pavd_calders2014.runXYMinGridding, dataFiles, controls=controls, otherArgs=otherargs)
            
            otherargs.planefit = pavd_calders2014.planeFitHubers(otherargs.xgrid[~otherargs.gridmask], otherargs.ygrid[~otherargs.gridmask], 
                otherargs.zgrid[~otherargs.gridmask], reportfile=otherargs.rptfile)
        
        minZenithAll = min(otherargs.minzenith)
        maxZenithAll = max(otherargs.maxzenith)
        minHeightBin = min(0.0, otherargs.minheight)  
        
        otherargs.zenith = numpy.arange(minZenithAll+otherargs.zenithbinsize/2, maxZenithAll, otherargs.zenithbinsize)
        otherargs.height = numpy.arange(minHeightBin, otherargs.maxheight, otherargs.heightbinsize)
        otherargs.counts = numpy.zeros([otherargs.zenith.shape[0],otherargs.height.shape[0]])
        otherargs.pulses = numpy.zeros([otherargs.zenith.shape[0],1])     
        
        print("Calculating vertical plant profiles...")
        lidarprocessor.doProcessing(pavd_calders2014.runZenithHeightStratification, dataFiles, controls=controls, otherArgs=otherargs)
        
        pgapz = numpy.where(otherargs.pulses > 0, 1 - numpy.cumsum(otherargs.counts, axis=1) / otherargs.pulses, numpy.nan)
        zenithRadians = numpy.radians(otherargs.zenith)
        zenithBinSizeRadians = numpy.radians(otherargs.zenithbinsize)
        
        lpp_pai,lpp_pavd,lpp_mla = pavd_calders2014.calcLinearPlantProfiles(otherargs.height, otherargs.heightbinsize, 
            zenithRadians, pgapz)
        sapp_pai,sapp_pavd = pavd_calders2014.calcSolidAnglePlantProfiles(zenithRadians, pgapz, otherargs.heightbinsize,
            zenithBinSizeRadians)
        
        pavd_calders2014.writeProfiles(outfile, otherargs.zenith, otherargs.height, pgapz, 
                                       lpp_pai, lpp_pavd, lpp_mla, sapp_pai, sapp_pavd)
           
                                                  
    else:
        
        msg = 'Unsupported metric %s' % metric
        raise CanopyMetricError(msg)
        

            
