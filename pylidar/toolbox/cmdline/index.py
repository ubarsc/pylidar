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
    p.add_argument("-i", "--input", help="Input SPD file name")
    p.add_argument("-o", "--output", help="Output SPD file with spatial index")
    p.add_argument("-r","--resolution", default=DEFAULT_RESOLUTION, 
        type=float, help="Output SPD file grid resolution " + 
            "(default: %(default)s)")
    p.add_argument("-b","--blocksize", type=float,
        help="Override the default blocksize")
    p.add_argument("--indextype", default=DEFAULT_INDEXTYPE,
        choices=['CARTESIAN', 'SPHERICAL', 'SCAN'],
        help="Spatial index type. (default: %(default)s)")
    p.add_argument("--pulseindexmethod", default=DEFAULT_PULSEINDEXMETHOD,
        choices=['FIRST_RETURN', 'LAST_RETURN'],
        help="Pulse index method. (default: %(default)s)")
    p.add_argument("--extent", nargs=4, metavar=('xmin', 'ymin', 'xmax', 'ymax'),
        help="Set extent of the output file")
    p.add_argument("--tempdir", help="Temporary directory to use. "+
            " If not specified, a temporary directory will be created and " +
            "removed at the end of processing.")
    p.add_argument("--wkt", help="projection to use for output in WKT format")

    cmdargs = p.parse_args()

    if cmdargs.input is None or cmdargs.output is None:
        p.print_help()
        sys.exit()

    return cmdargs

def run():
    """
    Main function. Looks at the command line arguments and calls
    the indexing code.
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

    gridindex.createGridSpatialIndex(cmdargs.input, cmdargs.output, 
                                extent=extent, tempDir=cmdargs.tempdir,
                                indexType=indexType,
                                pulseIndexMethod=pulseindexmethod,
                                binSize=cmdargs.resolution,
                                blockSize=cmdargs.blocksize,
                                wkt=cmdargs.wkt) 

