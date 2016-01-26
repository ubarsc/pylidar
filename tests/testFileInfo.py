#!/usr/bin/env python

from __future__ import print_function, division

import sys
from pylidar.lidarformats import generic
# need to import lidarprocessor also so we have
# all the formats imported
# TODO: sort this out
from pylidar import lidarprocessor

info = generic.getLidarFileInfo(sys.argv[1])
for key,val in sorted(info.__dict__.items()):
    if isinstance(val, dict):
        print(key.upper())
        for hkey,hval in sorted(val.items()):
            print(" ", hkey.upper(), hval)
    else:
        print(key.upper(), val)
    

