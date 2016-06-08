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
from pylidar.toolbox.translate import las2spdv4
from pylidar.toolbox.translate import spdv32spdv4
from pylidar.toolbox.translate import riegl2spdv4
from pylidar.toolbox.translate import spdv42las
from pylidar.toolbox.translate import ascii2spdv4

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", help="Input file name")
    p.add_argument("-o", "--output", help="Output file name")
    p.add_argument("-f", "--format", help="Output format. One of [SPDV4, LAS]")
    p.add_argument("--spatial", default=False, action="store_true", 
        help="Process the data spatially. Default is False and if True " + 
            "requires a spatial index in the input.")
    p.add_argument("--epsg", type=int,
        help="Set to the EPSG (if not in supplied LAS file). i.e. " + 
            "GDA / MGA Zone 56 is 28356 (only for LAS inputs)")
    p.add_argument("--scaling", nargs=5, metavar=('type', 'varName', 'dtype',
            'gain', 'offset'), action='append', 
            help="Set gain and offset scaling for named variable." +
            " Can be given multiple times for different variables." +
            " type should be one of [POINT|PULSE|WAVEFORM] dtype should be "+
            "UINT16 format or DFLT for standard fields. " +
            "(only for SPDV4 outputs)")
    p.add_argument("--range", nargs=4, metavar=('type', 'varName', 'min', 
            'max'), action='append', help="expected range for variables. " +
            "Will fail with an error if variables outside specified limit "+
            " or do not exist. (only for SPDV4 outputs)")
    p.add_argument("--binsize", "-b", type=float,
        help="Bin size to use when processing spatially (only for LAS inputs)")
    p.add_argument("--buildpulses", default=False, action="store_true",
        help="Build pulse data structure. Default is False (only for LAS inputs)")
    p.add_argument("--pulseindex", default="FIRST_RETURN",
        help="Pulse index method. Set to FIRST_RETURN or LAST_RETURN. " + 
            "Default is FIRST_RETURN. (only for LAS inputs)")
    p.add_argument("--internalrotation", dest="internalrotation",
        default=False, action="store_true", help="Use internal rotation data" +
            " when processing .rxp files. Overrides --externalrotationfn (only for RIEGL inputs)")
    p.add_argument("--externalrotationfn", dest="externalrotationfn",
        help="Input space delimited text file with external 4x4 rotation matrix" +
            " (only for RIEGL inputs)")
    p.add_argument("--magneticdeclination", dest="magneticdeclination",
        default=0.0, type=float, help="Use given magnetic declination when" +
            " processing .rxp files (only for RIEGL inputs)")
    p.add_argument("--coltype", nargs=2, metavar=('varName', 'dtype'),
            action='append', help="Set column name and types. Can be given"+
            " multiple times. dtype should be UINT16 format. " +
            "(only for ASCII inputs)")
    p.add_argument("--pulsecols", help="Comma seperated list of column names "+
            " that are the pulse columns. Names must be definied by --coltype"+
            " (only for ASCII inputs)")

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
                cmdargs.range, cmdargs.spatial, cmdargs.scaling, cmdargs.epsg, 
                cmdargs.binsize, cmdargs.buildpulses, cmdargs.pulseindex)

    elif inFormat == 'SPDV3' and cmdargs.format == 'SPDV4':
        spdv32spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.range, cmdargs.spatial, cmdargs.scaling)

    elif inFormat == 'riegl' and cmdargs.format == 'SPDV4':
        riegl2spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.range, cmdargs.scaling, cmdargs.internalrotation, 
                cmdargs.magneticdeclination, cmdargs.externalrotationfn)

    elif inFormat == 'SPDV4' and cmdargs.format == 'LAS':
        spdv42las.translate(info, cmdargs.input, cmdargs.output, 
                cmdargs.spatial)

    elif inFormat == 'ASCII TS' and cmdargs.format == 'SPDV4':

        if cmdargs.coltype is None:
            msg = "must pass --coltypes parameter"
            raise generic.LiDARInvalidSetting(msg)
        if cmdargs.pulsecols is None:
            msg = "must pass --pulsecols parameter"
            raise generic.LiDARInvalidSetting(msg)

        pulsecols = cmdargs.pulsecols.split(',')

        ascii2spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.range, cmdargs.scaling, cmdargs.coltype, pulsecols)

    else:
        msg = 'Cannot convert between formats %s and %s' 
        msg = msg % (inFormat, cmdargs.format)
        raise generic.LiDARFunctionUnsupported(msg)

