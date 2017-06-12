"""
Does the creation of a spatial index
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
import os
import argparse

from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from pylidar.toolbox.indexing import gridindex
from pylidar.basedriver import Extent

DEFAULT_BLOCKSIZE = 1000.0
DEFAULT_INDEXTYPE = "CARTESIAN"
DEFAULT_PULSEINDEXMETHOD = "FIRST_RETURN"

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser(description="""
        Split one or more input lidar files into separate tiles, on a regular grid. 
    """)
    p.add_argument("input", nargs='+', 
        help=("Input lidar file name. Can be specified multiple times, or using wild cards, for "
            +"multiple inputs."))
    p.add_argument("-r","--resolution", 
        type=float, help="Output resolution to use when choosing corners " + 
            "of the tiles (default: same value as --blocksize)")
    p.add_argument("-b","--blocksize", type=float, default=DEFAULT_BLOCKSIZE, 
        help=("The size (in world coordinates, i.e. metres) of the tiles into which the "+
            "data will be divided (default: %(default)s) "))
    p.add_argument("--indextype", default=DEFAULT_INDEXTYPE,
        choices=['CARTESIAN', 'SPHERICAL', 'SCAN'],
        help="Spatial index type. (default: %(default)s)")
    p.add_argument("--pulseindexmethod", default=DEFAULT_PULSEINDEXMETHOD,
        choices=['FIRST_RETURN', 'LAST_RETURN'],
        help="Pulse index method. (default: %(default)s)")
    p.add_argument("--extent", nargs=4, metavar=('xmin', 'ymin', 'xmax', 'ymax'),
        help="Set extent of the input file to use")
    p.add_argument("--outdir", default='.', help="Output directory to use "+
            " (default: %(default)s)")
    p.add_argument("-q", "--quiet", default=False, action='store_true',
        help="Suppress the printing of the tile filenames")
    p.add_argument("-f", "--footprint", choices=['union', 'intersection'],
        default='union', help='how to combine multiple inputs')
    p.add_argument("--format", choices=['SPDV4', 'LAS'], default='SPDV4',
        help=("Format to output the tiles as (default: %(default)s). This is not as orthogonal "+
            "as it should be, i.e. not all combinations of input and output format actually work. "))
    p.add_argument("--keepemptytiles", default=False, action="store_true",
        help="Do not delete the tiles which turn out to be empty. Default will remove them")
    p.add_argument("--buildpulses", default=False, action="store_true",
            help="Build pulse data structure. Default is False (only for LAS inputs)")

    cmdargs = p.parse_args()
    
    if cmdargs.resolution is None:
        cmdargs.resolution = cmdargs.blocksize

    return cmdargs

def run():
    """
    Main function. Looks at the command line arguments and calls
    the tiling code.
    """
    cmdargs = getCmdargs()

    if cmdargs.extent is not None:
        extent = Extent(float(cmdargs.extent[0]), float(cmdargs.extent[2]), 
            float(cmdargs.extent[1]), float(cmdargs.extent[3]), 
            cmdargs.resolution)
    else:
        extent = None

    try:
        indexType = getattr(spdv4, "SPDV4_INDEX_%s" % cmdargs.indextype)
    except AttributeError:
        msg = 'Unsupported index type %s' % cmdargs.indextype
        raise generic.LiDARPulseIndexUnsupported(msg)            

    try:
        pulseindexmethod = getattr(spdv4, 
                    "SPDV4_PULSE_INDEX_%s" % cmdargs.pulseindexmethod)
    except AttributeError:
        msg = 'Unsupported pulse indexing method %s' % cmdargs.pulseindexmethod
        raise generic.LiDARPulseIndexUnsupported(msg)        

    if cmdargs.footprint == 'union':
        footprint = lidarprocessor.UNION
    elif cmdargs.footprint == 'intersection':    
        footprint = lidarprocessor.INTERSECTION

    # returns header and extent that we don't use so we ignore.
    # those are useful in the spatial indexing situation
    header, extent, fnameList = gridindex.splitFileIntoTiles(cmdargs.input, 
                                binSize=cmdargs.resolution,
                                blockSize=cmdargs.blocksize,
                                tempDir=cmdargs.outdir, 
                                extent=extent, indexType=indexType,
                                pulseIndexMethod=pulseindexmethod,
                                footprint=footprint, 
                                outputFormat=cmdargs.format,
                                buildPulses=cmdargs.buildpulses)
    
    # Delete the empty ones
    if not cmdargs.keepemptytiles:
        fnameList = deleteEmptyTiles(fnameList)

    # now print the names of the tiles to the screen
    if not cmdargs.quiet:
        for fname, subExtent in fnameList:
            print(fname)


def deleteEmptyTiles(fnameList):
    """
    After creating all the tiles, some of them will have ended up being empty. Delete them. 
    """
    newFileList = []
    for (fname, subExtent) in fnameList:
        info = generic.getLidarFileInfo(fname)
        header = info.header
        translator = info.getHeaderTranslationDict()
        numPointsField = translator[generic.HEADER_NUMBER_OF_POINTS]
        if header[numPointsField] > 0:
            newFileList.append((fname, subExtent))
        else:
            if os.path.exists(fname):
                os.remove(fname)

    return newFileList
