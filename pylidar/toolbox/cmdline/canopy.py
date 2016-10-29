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
    p.add_argument("-i", "--infiles", nargs="+", 
        help="Input lidar files (required)")
    p.add_argument("-o", "--output", help="output file (required)")
    p.add_argument("-m", "--metric", default=canopymetric.DEFAULT_CANOPY_METRIC, 
        help="Canopy metric to calculate. default=%(default)s")
    p.add_argument("-b", "--binsize", type=float, help="Binsize to use for reading input files")
    p.add_argument("-q", "--quiet", default=False, action='store_true', 
        help="Don't show progress etc")

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

    canopymetric.runCanopyMetric(cmdargs.infiles, cmdargs.output, 
        cmdargs.binsize, cmdargs.metric, cmdargs.quiet)
