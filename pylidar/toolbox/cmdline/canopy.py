"""
Provides support for calling the canopy function from the 
command line.
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
import argparse
import numpy

from pylidar import lidarprocessor
from pylidar.toolbox.canopy import canopymetric

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infiles", nargs="+", help="Input lidar files (required)")
    p.add_argument("-o", "--output", nargs="+", help="Output file/s (required)")
    p.add_argument("-m", "--metric", default="PAVD_CALDERS2014", choices=["PAVD_CALDERS2014","VOXEL_HANCOCK2016","PGAP_ARMSTON2013"], 
        help="Canopy metric to calculate. default=%(default)s")
    p.add_argument("-w","--weighted", default=False, action="store_true", help="Calculate Pgap(z) using weighted interception (Armston et al., 2013)")         
    p.add_argument("-p","--planecorrection", default=False, action="store_true", help="Apply plane correction to point heights (PAVD_CALDERS2014 metric only)")
    p.add_argument("-r","--reportfile", help="Output file report file for point height plane correction (PAVD_CALDERS2014 metric only)")
    p.add_argument("-v","--voxelsize", default=1.0, type=float, help="Voxel spatial resolution (VOXEL_HANCOCK2016 metric only)")
    p.add_argument("-b","--bounds", nargs=6, type=float, default=[-50.0,-50.0,0.0,50.0,50.0,50.0], 
        help="Voxel bounds [minX,minY,minZ,maxX,maxY,maxZ] (VOXEL_HANCOCK2016 metric only)")
    p.add_argument("--heightcol", default='Z', help="Point column name to use for vertical profile heights (default: %(default)s).")
    p.add_argument("--heightbinsize", default=0.5, type=float, help="Vertical bin size (default: %(default)f m)")
    p.add_argument("--minheight", default=0.0, type=float, help="Minimum point height to include in vertical profile (default: %(default)f m)")
    p.add_argument("--maxheight", default=50.0, type=float, help="Maximum point height to include in vertical profile (default: %(default)f m)")
    p.add_argument("--zenithbinsize", default=5.0, type=float, help="View zenith bin size (default: %(default)f deg)")
    p.add_argument("--minzenith", nargs="+", default=[35.0], type=float, help="Minimum view zenith angle to use for each input file")
    p.add_argument("--maxzenith", nargs="+", default=[70.0], type=float, help="Maximum view zenith angle to use for each input file")
    p.add_argument("--minazimuth", nargs="+", default=[0.0], type=float, help="Minimum view azimuth angle to use for each input file")
    p.add_argument("--maxazimuth", nargs="+", default=[360.0], type=float, help="Maximum view azimuth angle to use for each input file")
    p.add_argument("--totalpaimethod", choices=['HINGE','LINEAR','EXTERNAL'], default="HINGE",  
        help="Total PAI method for solid angle weighted plant profiles (default: %(default)s; PAVD_CALDERS2014 metric only)")
    p.add_argument("--totalpai", default=1.0, type=float, help="Total PAI to use if --totalpaimethod=EXTERNAL is selected (default: %(default)f m2/m2; PAVD_CALDERS2014 metric only)")
    p.add_argument("--gridsize", default=20, type=int, help="Grid dimension for the point height plane correction (default: %(default)i; PAVD_CALDERS2014 metric only)")
    p.add_argument("--gridbinsize", default=5.0, type=float, help="Grid resolution for the point height plane correction (default: %(default)f m; PAVD_CALDERS2014 metric only)")
    p.add_argument("--excludedclasses", nargs="+", default=[], type=int, help="Point CLASSIFICATION values to exclude from the metric calculation (default is all points)")
    p.add_argument("--rasterdriver", default="HFA", help="GDAL format for output raster (default is %(default)s)")
    p.add_argument("--externaltransformfn", nargs="+", default=[], help="External transform filenames (for RIEGL RXP input files)")
    p.add_argument("--externaldem", help="External single layer DEM image to use for calculation of point heights (PAVD_CALDERS2014) or lower boundary of voxel traversal (VOXEL_HANCOCK2016)")
       
    cmdargs = p.parse_args()
    if (cmdargs.infiles is None) or (cmdargs.metric is None):
        print("Must specify metric and input file names") 
        p.print_help()
        sys.exit()
    
    nOutFiles = len(cmdargs.output)
    if (nOutFiles != 1) and (cmdargs.metric == "PAVD_CALDERS2014"):
        print("Must specify output CSV file name") 
        p.print_help()
        sys.exit()
    elif (nOutFiles != 3) and (cmdargs.metric == "VOXEL_HANCOCK2016"):
        print("Must specify three output GDAL image file names (NSCANS,COVER,CLASS). More outputs will be added as implementation progresses.") 
        p.print_help()
        sys.exit()
    
    nInfiles = len(cmdargs.infiles)
    if ( (len(cmdargs.minzenith) != nInfiles) or (len(cmdargs.maxzenith) != nInfiles) ) and (cmdargs.metric == "PAVD_CALDERS2014"):
        print("--minzenith and --maxzenith must have the same length as --infiles")
        p.print_help()
        sys.exit()
    if ( (len(cmdargs.minazimuth) != nInfiles) or (len(cmdargs.maxazimuth) != nInfiles) ) and (cmdargs.metric == "PAVD_CALDERS2014"):
        print("--minazimuth and --maxazimuth must have the same length as --infiles")
        p.print_help()
        sys.exit()
    
    nExternalTransformFiles = len(cmdargs.externaltransformfn)
    if nExternalTransformFiles != nInfiles:
        if nExternalTransformFiles == 0:
            cmdargs.externaltransformfn = None
        else:
            print("If present --externaltransformfn must have the same length as --infiles")
            p.print_help()
            sys.exit()        
    
    if cmdargs.planecorrection and (cmdargs.externaldem is not None) and (cmdargs.metric == "PAVD_CALDERS2014"):
        print("Cannot specify both --planecorrection and --externalDEM. Choose one.") 
        p.print_help()
        sys.exit()
    
    return cmdargs


def run():
    """
    Main function. Checks the command line parameters and calls
    the canopymetrics routine.
    """
    cmdargs = getCmdargs()
    otherargs = lidarprocessor.OtherArgs()

    if cmdargs.metric == "PAVD_CALDERS2014":    
        
        otherargs.weighted = cmdargs.weighted
        otherargs.heightcol = cmdargs.heightcol
        otherargs.heightbinsize = cmdargs.heightbinsize
        otherargs.minheight = cmdargs.minheight
        otherargs.maxheight = cmdargs.maxheight
        otherargs.zenithbinsize = cmdargs.zenithbinsize
        otherargs.minzenith = cmdargs.minzenith
        otherargs.maxzenith = cmdargs.maxzenith       
        otherargs.minazimuth = cmdargs.minazimuth
        otherargs.maxazimuth = cmdargs.maxazimuth      
        otherargs.planecorrection = cmdargs.planecorrection
        otherargs.rptfile = cmdargs.reportfile
        otherargs.gridsize = cmdargs.gridsize
        otherargs.gridbinsize = cmdargs.gridbinsize
        otherargs.excludedclasses = cmdargs.excludedclasses
        otherargs.externaltransformfn = cmdargs.externaltransformfn
        otherargs.totalpaimethod = cmdargs.totalpaimethod
        otherargs.totalpai = cmdargs.totalpai
        otherargs.externaldem = cmdargs.externaldem

    elif cmdargs.metric == "VOXEL_HANCOCK2016":    
        
        otherargs.voxelsize = numpy.repeat(cmdargs.voxelsize, 3)
        otherargs.bounds = numpy.array(cmdargs.bounds, dtype=numpy.float32)
        otherargs.rasterdriver = cmdargs.rasterdriver
        otherargs.externaltransformfn = cmdargs.externaltransformfn
        otherargs.externaldem = cmdargs.externaldem

    elif cmdargs.metric == "PGAP_ARMSTON2013":    
        
        pass
    
    else:
        msg = 'Unsupported metric %s' % cmdargs.metric
        raise CanopyMetricError(msg)
        
    canopymetric.runCanopyMetric(cmdargs.infiles, cmdargs.output, cmdargs.metric, otherargs)

