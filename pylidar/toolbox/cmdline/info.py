"""
Does the printing out to terminal of file info
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

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", help="Input file name")
    p.add_argument("-v", "--verbose", default=False, action="store_true",
            help="print more verbose output")

    cmdargs = p.parse_args()

    if cmdargs.input is None:
        p.print_help()
        sys.exit()

    return cmdargs

def run():
    """
    Main function. Looks at the command line arguments
    and prints the info
    """
    cmdargs = getCmdargs()

    info = generic.getLidarFileInfo(cmdargs.input, cmdargs.verbose)
    print('Driver Name:', info.getDriverName())
    print('')

    for key,val in sorted(info.__dict__.items()):
        if isinstance(val, dict):
            print(key.upper()) # TODO: we need to be uppercase??
            for hkey,hval in sorted(val.items()):
                print(" ", hkey.upper(), hval)
        else:
            print(key.upper(), val)
