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
from pylidar.toolbox.translate import lvisbin2spdv4
from pylidar.toolbox.translate import lvishdf52spdv4
from pylidar.toolbox.translate import pulsewaves2spdv4
from pylidar.toolbox.translate import spdv42pulsewaves

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", help="Input file name")
    p.add_argument("-o", "--output", help="Output file name")
    p.add_argument("-f", "--format", help="Output format", 
        choices=['SPDV4', 'LAS', 'PULSEWAVES'])
    p.add_argument("--spatial", default=False, action="store_true", 
        help="Process the data spatially. Default is False and if True " + 
            "requires a spatial index in the input.")
    p.add_argument("--extent", nargs=4, metavar=('xmin', 'ymin', 'xmax', 'ymax'),
        help="Only process the given spatial extent. Only valid with --spatial"+
            " option")
    p.add_argument("--epsg", type=int,
        help="Set to the EPSG (if not in supplied LAS file). e.g. " + 
            "GDA / MGA Zone 56 is 28356 (not yet implemented for all input formats, but it should be)")
    p.add_argument("--wktfile", help=("Name of text file with WKT string of projection of input "+
        "file. Not yet implemented for all input formats, but it should be. "))
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
    p.add_argument("--null", nargs=3, metavar=('type', 'varName', 'value'),
            action='append', help="Set null value for named variable." +
            " Can be given multiple times for different variables." +
            " type should be one of [POINT|PULSE|WAVEFORM]. value is scaled."+
            " (only for SPDV4 outputs)")
    p.add_argument("--binsize", "-b", type=float,
            help="Bin size to use when processing spatially (only for LAS inputs)")
    p.add_argument("--buildpulses", default=False, action="store_true",
            help="Build pulse data structure. Default is False (only for LAS inputs)")
    p.add_argument("--pulseindex", default="FIRST_RETURN",
            choices=['FIRST_RETURN', 'LAST_RETURN'],
            help="Pulse index method. (default: %(default)s). (only for LAS inputs)")
    p.add_argument("--internalrotation", dest="internalrotation",
            default=False, action="store_true", help="Use internal rotation data" +
            " when processing .rxp files. Overrides --externalrotationfn " + 
            "(only for RIEGL inputs)")
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
            " that are the pulse columns. Names must be defined by --coltype"+
            " (only for ASCII inputs)")
    p.add_argument("--classtrans", nargs=2, metavar=('internal', 'class'),
            action='append', help="Set translation between internal codes and" +
            " standard codes for CLASSIFICATION column. internal should be " +
            "an integer and class should be one of [CREATED,UNCLASSIFIED," +
            "GROUND,LOWVEGE,MEDVEGE,HIGHVEGE,BUILDING,LOWPOINT,HIGHPOINT," +
            "WATER,RAIL,ROAD,BRIDGE,WIREGUARD,WIRECOND,TRANSTOWER,INSULATOR," +
            "TRUNK,FOLIAGE,BRANCH] (only for ASCII inputs)")
    p.add_argument("--constcol", nargs=4, metavar=('type', 'varName', 'dtype', 
            'value'), action='append', help="Create a constant column in the " +
            "output file with the given type, name and value. type should be " +
            "one of [POINT|PULSE|WAVEFORM] dtype should be UINT16 format. " +
            "(only for SPDV4 outputs)")
    p.add_argument("--lasscalings", default=False, action="store_true",
            help="Use the scalings and types in the LAS file in the output. " + 
            "Overrides --scaling. (only for LAS inputs)")

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

    if cmdargs.extent is not None and not cmdargs.spatial:
        print("--extent can only be specified with --spatial")
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
    wktStr = None
    if cmdargs.wktfile is not None:
        wktStr = open(cmdargs.wktfile).read()

    # first determine the format of the input file
    # I'm not sure this is possible in all situations, but assume it is for now
    info = generic.getLidarFileInfo(cmdargs.input)
    inFormat = info.getDriverName()

    if inFormat == 'LAS' and cmdargs.format == 'SPDV4':
        las2spdv4.translate(info, cmdargs.input, cmdargs.output, 
                cmdargs.range, cmdargs.spatial, cmdargs.extent, cmdargs.scaling, 
                cmdargs.epsg, cmdargs.binsize, cmdargs.buildpulses, 
                cmdargs.pulseindex, cmdargs.null, cmdargs.constcol,
                cmdargs.lasscalings)

    elif inFormat == 'SPDV3' and cmdargs.format == 'SPDV4':
        spdv32spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.range, cmdargs.spatial, cmdargs.extent, cmdargs.scaling,
                cmdargs.null, cmdargs.constcol)

    elif inFormat == 'riegl' and cmdargs.format == 'SPDV4':
        riegl2spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.range, cmdargs.scaling, cmdargs.internalrotation, 
                cmdargs.magneticdeclination, cmdargs.externalrotationfn,
                cmdargs.null, cmdargs.constcol, epsg=cmdargs.epsg, wkt=wktStr)

    elif inFormat == 'SPDV4' and cmdargs.format == 'LAS':
        spdv42las.translate(info, cmdargs.input, cmdargs.output, 
                cmdargs.spatial, cmdargs.extent)

    elif inFormat == 'ASCII' and cmdargs.format == 'SPDV4':

        if cmdargs.coltype is None:
            msg = "must pass --coltypes parameter"
            raise generic.LiDARInvalidSetting(msg)

        if cmdargs.pulsecols is not None:
            pulsecols = cmdargs.pulsecols.split(',')
        else:
            pulsecols = None

        classtrans = None
        if cmdargs.classtrans is not None:
            # translate strings to codes
            for internalCode, strLasCode in cmdargs.classtrans:
                internalCode = int(internalCode)
                strLasCodeFull = "CLASSIFICATION_%s" % strLasCode
                try:
                    lasCode = getattr(lidarprocessor, strLasCodeFull)
                except AttributeError:
                    msg = 'class %s not understood' % strLasCode
                    raise generic.LiDARInvalidSetting(msg)

                if classtrans is None:
                    classtrans = []
                classtrans.append((internalCode, lasCode))

        ascii2spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.coltype, pulsecols, cmdargs.range, cmdargs.scaling, 
                classtrans, cmdargs.null, cmdargs.constcol)

    elif inFormat == 'LVIS Binary' and cmdargs.format == 'SPDV4':

        lvisbin2spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.range, cmdargs.scaling, 
                cmdargs.null, cmdargs.constcol)

    elif inFormat == 'LVIS HDF5' and cmdargs.format == 'SPDV4':

        lvishdf52spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.range, cmdargs.scaling, 
                cmdargs.null, cmdargs.constcol)

    elif inFormat == 'PulseWaves' and cmdargs.format == 'SPDV4':

        pulsewaves2spdv4.translate(info, cmdargs.input, cmdargs.output,
                cmdargs.range, cmdargs.scaling, 
                cmdargs.null, cmdargs.constcol)

    elif inFormat == 'SPDV4' and cmdargs.format == 'PULSEWAVES':
        spdv42pulsewaves.translate(info, cmdargs.input, cmdargs.output)

    else:
        msg = 'Cannot convert between formats %s and %s' 
        msg = msg % (inFormat, cmdargs.format)
        raise generic.LiDARFunctionUnsupported(msg)

