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

from pylidar import lidarprocessor
from pylidar.toolbox.canopy import canopymetric

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infiles", nargs="+", help="Input lidar files (required)")
    p.add_argument("-o", "--output", help="Output file (required)")
    p.add_argument("-m", "--metric", default=canopymetric.DEFAULT_CANOPY_METRIC, help="Canopy metric to calculate. default=%(default)s")
    p.add_argument("-w","--weighted", default=False, action="store_true", help="Calculate Pgap(z) using weighted interception (Armston et al., 2013)")         
    p.add_argument("-p","--planecorrection", default=False, action="store_true", help="Apply plane correction to point heights (PAVD_CALDERS2014 metric only)")
    p.add_argument("-r","--reportfile", help="Output file report file for point height plane correction (PAVD_CALDERS2014 metric only)")
    p.add_argument("--heightcol", default='Z', help="Point column name to use for vertical profile heights (default: %(default)s).")
    p.add_argument("--heightbinsize", default=0.5, type=float, help="Vertical bin size (default: %(default)f m)")
    p.add_argument("--minheight", default=0.0, type=float, help="Minimum vertical profile height (default: %(default)f m)")
    p.add_argument("--maxheight", default=50.0, type=float, help="Maximum vertical profile height (default: %(default)f m)")
    p.add_argument("--zenithbinsize", default=5.0, type=float, help="View zenith bin size (default: %(default)f deg)")
    p.add_argument("--minzenith", nargs="+", default=[35.0,5.0], type=float, help="Minimum view zenith angle to use for each input file (PAVD_CALDERS2014 metric only)")
    p.add_argument("--maxzenith", nargs="+", default=[70.0,35.0], type=float, help="Maximum view zenith angle to use for each input file (PAVD_CALDERS2014 metric only)")
    p.add_argument("--gridsize", default=100, type=int, help="Grid dimension for the point height plane correction (default: %(default)i; PAVD_CALDERS2014 metric only)")
    p.add_argument("--gridbinsize", default=5.0, type=float, help="Grid resolution for the point height plane correction (default: %(default)f m; PAVD_CALDERS2014 metric only)")    
       
    cmdargs = p.parse_args()
    if cmdargs.infiles is None:
        print("Must specify input file names") 
        p.print_help()
        sys.exit()

    if cmdargs.output is None:
        print("Must specify output file name") 
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
        otherargs.planecorrection = cmdargs.planecorrection
        otherargs.rptfile = cmdargs.reportfile
        otherargs.gridsize = cmdargs.gridsize
        otherargs.gridbinsize = cmdargs.gridbinsize
    
    else:
        msg = 'Unsupported metric %s' % cmdargs.metric
        raise CanopyMetricError(msg)
        
    canopymetric.runCanopyMetric(cmdargs.infiles, cmdargs.output, cmdargs.metric, otherargs)

