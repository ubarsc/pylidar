#!/usr/bin/env python

from __future__ import print_function, division

import sys
from pylidar.lidarformats import generic
# need to import lidarprocessor also so we have
# all the formats imported
# TODO: sort this out
from pylidar import lidarprocessor

info = generic.getLidarFileInfo(sys.argv[1])
for key in info.__dict__.keys():
    print(key, getattr(info, key))
    

