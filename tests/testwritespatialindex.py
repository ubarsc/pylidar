#!/usr/bin/env python

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

import os
import sys
import tempfile
import optparse
import numpy as np
from pylidar.lidarformats import generic
from pylidar.lidarformats import spdv4
from pylidar.toolbox.indexing import gridindex
from pylidar.basedriver import Extent

INDEX_CARTESIAN = spdv4.SPDV4_INDEX_CARTESIAN
INDEX_SPHERICAL = spdv4.SPDV4_INDEX_SPHERICAL
INDEX_SCAN = spdv4.SPDV4_INDEX_SCAN


class CmdArgs(object):
    def __init__(self):
        p = optparse.OptionParser()
        p.add_option("-i","--infile", dest="infile",
            help="Input SPD file")
        p.add_option("-o","--outfile", dest="outfile",
            help="Output SPD file with spatial index")
        p.add_option("-r","--resolution", dest="resolution",
            default=1.0, type="float",
            help="Output SPD file grid resolution")
        p.add_option("--indextype", dest="indextype",
            default="CARTESIAN",
            help="Spatial index type")
        p.add_option("--pulseindexmethod", dest="pulseindexmethod",
            default="FIRST_RETURN",
            help="Pulse index method")            
        (options, args) = p.parse_args()
        self.__dict__.update(options.__dict__)

        if self.infile is None or self.outfile is None:
            p.print_help()
            sys.exit()


def doIndexing(cmdargs,dirname):
    """
    Does the spatial indexing
    """
        
    info = generic.getLidarFileInfo(cmdargs.infile)
    h = info.header

    if cmdargs.indextype is "CARTESIAN":        
        ulx = np.floor(h["X_MIN"])
        uly = np.ceil(h["Y_MAX"])
        lrx = np.ceil(h["X_MAX"])
        lry = np.floor(h["Y_MIN"])        
        extent = Extent(ulx, lrx, lry, uly, cmdargs.resolution)        
        try:
            pulseindexmethod = getattr(spdv4,"SPDV4_PULSE_INDEX_%s"%cmdargs.pulseindexmethod)
        except AttributeError:
            msg = 'Unsupported pulse indexing method'
            raise generic.LiDARPulseIndexUnsupported(msg)            
        gridindex.createGridSpatialIndex(cmdargs.infile, cmdargs.outfile, 
                                         extent=extent, tempDir=dirname,
                                         indexMethod=INDEX_CARTESIAN,
                                         pulseIndexMethod=pulseindexmethod)

    elif cmdargs.indextype is "SPHERICAL":        
        ulx = 0.0
        uly = 180.0
        lrx = 360.0
        lry = 0.0        
        extent = Extent(ulx, lrx, lry, uly, cmdargs.resolution)            
        gridindex.createGridSpatialIndex(cmdargs.infile, cmdargs.outfile, 
                                         extent=extent, tempDir=dirname,
                                         indexMethod=INDEX_SPHERICAL)
        
    elif cmdargs.indextype is "SCAN":        
       gridindex.createGridSpatialIndex(cmdargs.infile, cmdargs.outfile, tempDir=dirname,
                                         indexMethod=INDEX_SCAN)

    else:
        msg = 'Unsupported spatial indexing method'
        raise generic.LiDARSpatialIndexNotAvailable(msg)
    

if __name__ == '__main__':

    dirname = tempfile.mkdtemp()
    cmdargs = CmdArgs()
    doIndexing(cmdargs,dirname)
    os.removedirs(dirname)         
