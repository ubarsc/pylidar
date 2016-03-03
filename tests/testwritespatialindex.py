#!/usr/bin/env python

import os
import sys
import numpy as np
from pylidar.lidarformats import generic

from pylidar.toolbox.indexing import gridindex
from pylidar.basedriver import Extent

infile = sys.argv[1]
outfile = sys.argv[2]
resolution = float(sys.argv[3])

info = generic.getLidarFileInfo(infile)
h = info.header

ulx = np.floor(h["X_MIN"])
uly = np.ceil(h["Y_MAX"])
lrx = np.ceil(h["X_MAX"])
lry = np.floor(h["Y_MIN"])

extent = Extent(ulx, lrx, lry, uly, resolution)

dirname = 'tmp'
if not os.path.exists(dirname):
    os.mkdir(dirname)
gridindex.createGridSpatialIndex(infile, outfile, extent=extent, tempDir=dirname)
