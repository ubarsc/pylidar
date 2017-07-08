"""
Provides support for calling the rasterise function from the 
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
from pylidar.toolbox import rasterization

DEFAULT_FUNCTION = rasterization.DEFAULT_FUNCTION
DEFAULT_ATTRIBUTE = rasterization.DEFAULT_ATTRIBUTE

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infiles", nargs="+", 
        help="Input lidar files (required)")
    p.add_argument("-o", "--output", help="output file (required)")
    p.add_argument("-a", "--attributes", nargs="+", 
        help="Pulse, point or waveform attributes to rasterize. Can list " +
        "multiple attributes for one --type to create an image stack.")
    p.add_argument("-f", "--function", default=DEFAULT_FUNCTION,
        help="function to apply to data. Must contain module name " + 
        "and be able to operate on masked arrays and take the 'axis' " +
        "parameter. default=%(default)s")
    p.add_argument("-t", "--type", default=DEFAULT_ATTRIBUTE,
        choices=['POINT', 'PULSE'],
        help="Type of data to operate on. default=%(default)s")
    p.add_argument("-b","--background", type=float, default=0, 
        help="Output raster background value. default=%(default)s.")
    p.add_argument("--binsize", type=float, 
        help="resolution to do processing at")
    p.add_argument("--footprint", choices=['INTERSECTION', 'UNION', 'BOUNDS_FROM_REFERENCE'],
        help="Multiple input files will be combined in this way")
    p.add_argument("--windowsize", type=float, 
        help="Window size to use for each processing block")
    p.add_argument("--drivername", 
        help="GDAL driver to use for the output image file")
    p.add_argument("--driveroptions", nargs="+",
        help="Creation options to pass to the output format driver, e.g. COMPRESS=LZW")
    p.add_argument("-m", "--module", help="Extra modules that need to be " + 
        "imported for use by --function")
    p.add_argument("-q", "--quiet", default=False, action='store_true', 
        help="Don't show progress etc")

    cmdargs = p.parse_args()
    if cmdargs.attributes is None:
        print("Must specify attributes to use") 
        p.print_help()
        sys.exit()

    if cmdargs.output is None:
        print("Must specify output file name") 
        p.print_help()
        sys.exit()

    if cmdargs.footprint is not None:
        # Evaluate the footprint string as a named constant
        cmdargs.footprint = eval("lidarprocessor.{}".format(cmdargs.footprint))

    return cmdargs

def run():
    """
    Main function. Checks the command line parameters and calls
    the rasterisation routine.
    """
    cmdargs = getCmdargs()

    rasterization.rasterize(cmdargs.infiles, cmdargs.output, 
        cmdargs.attributes, cmdargs.function,
        cmdargs.type, cmdargs.background, cmdargs.binsize, cmdargs.module,
        cmdargs.quiet, footprint=cmdargs.footprint, 
        windowSize=cmdargs.windowsize, driverName=cmdargs.drivername,
        driverOptions=cmdargs.driveroptions)

