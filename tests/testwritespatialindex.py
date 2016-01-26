#!/usr/bin/env python

import sys
from pylidar.toolbox.indexing import gridindex
from pylidar.basedriver import Extent

infile = sys.argv[1]
outfile = sys.argv[2]
extent = Extent(471698.99841, 471800.99, 5228601.0, 5228702.98222, 1.0)

gridindex.createGridSpatialIndex(infile, outfile, extent=extent, tempDir='tmp')


