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
import argparse

from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from pylidar.toolbox.indexing import gridindex
from pylidar.basedriver import Extent

DEFAULT_RESOLUTION = 1.0
DEFAULT_INDEXTYPE = "CARTESIAN"
DEFAULT_PULSEINDEXMETHOD = "FIRST_RETURN"

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", action='append', 
        help="Input SPD file name. Can be specified multiple times for "
            +"multiple inputs.", required=True)
    p.add_argument("-r","--resolution", default=DEFAULT_RESOLUTION, 
        type=float, help="Output resolution to use when choosing corners " + 
            "of the tiles (default: %(default)s)")
    p.add_argument("-b","--blocksize", type=float,
        help="Override the default blocksize")
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

    cmdargs = p.parse_args()

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
                                footprint=footprint)

    # now print the names of the tiles to the screen
    if not cmdargs.quiet:
        for fname, subExtent in fnameList:
            print(fname)
