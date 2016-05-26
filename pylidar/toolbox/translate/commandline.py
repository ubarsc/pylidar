"""
Provides support for calling the translate functions from the 
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
from pylidar.lidarformats import generic
# need to import lidarprocessor also so we have
# all the formats imported
from pylidar import lidarprocessor

# conversion modules
from . import las2spdv4
from . import spdv32spdv4

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="Input file name")
    p.add_argument("--output", help="Output file name")
    p.add_argument("--format", help="Output format. One of [SPDV4, LAS]")
    p.add_argument("--spatial", default=False, action="store_true", 
        help="Process the data spatially. Default is False and if True " + 
            "requires a spatial index in the input.")
    p.add_argument("--epsg", type=int,
        help="Set to the EPSG (if not in supplied LAS file). i.e. " + 
            "GDA / MGA Zone 56 is 28356")
    p.add_argument("--scaling", nargs=4, metavar=('type', 'varName', 'gain', 
            'offset'), action='append', 
            help="Set gain and offset scaling for named variable." +
            " Can be given multiple times for different variables." +
            " type should be one of [POINT|PULSE|WAVEFORM]")
    p.add_argument("--binsize", "-b", type=float,
        help="Bin size to use when processing spatially (only for LAS inputs)")
    p.add_argument("--buildpulses", default=False, action="store_true",
        help="Build pulse data structure. Default is False (only for LAS inputs)")
    p.add_argument("--pulseindex", default="FIRST_RETURN",
        help="Pulse index method. Set to FIRST_RETURN or LAST_RETURN. " + 
            "Default is FIRST_RETURN.")

    cmdargs = p.parse_args()

    if cmdargs.input is None or cmdargs.output is None or cmdargs.format is None:
        p.print_help()
        sys.exit()

    cmdargs.pulseindex = cmdargs.pulseindex.upper()
    if (cmdargs.pulseindex != "FIRST_RETURN" and 
        cmdargs.pulseindex != "LAST_RETURN"):
        print("--pulseindex must be either FIRST_RETURN or LAST_RETURN")
        p.print_help()
        sys.exit()

    cmdargs.format = cmdargs.format.upper()

    return cmdargs

def run():
    """
    Main function. Looks at the command line arguments
    and calls the appropiate translation function.
    """
    cmdargs = getCmdargs()

    # first determine the format of the input file
    # I'm not sure this is possible in all situations, but assume it is for now
    info = generic.getLidarFileInfo(cmdargs.input)
    inFormat = info.getDriverName()

    if inFormat == 'LAS' and cmdargs.format == 'SPDV4':
        las2spdv4.translate(info, cmdargs.input, cmdargs.output, 
                cmdargs.spatial, cmdargs.scaling, cmdargs.epsg, 
                cmdargs.binsize, cmdargs.buildpulses, cmdargs.pulseindex)

    if inFormat == 'SPDV3' and cmdargs.format == 'SPDV4':
        spdv32spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.spatial, cmdargs.scaling)

    else:
        msg = 'Cannot convert between formats %s and %s' 
        msg = msg % (inFormat, cmdargs.format)
        raise generic.LiDARFunctionUnsupported(msg)


