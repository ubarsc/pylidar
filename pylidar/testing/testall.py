
"""
Runs all the available test suites
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

from . import utils
from . import testsuite1
from . import testsuite2
from . import testsuite3
from . import testsuite4
from . import testsuite5
from pylidar import lidarprocessor

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", help="Input tar file name")
    p.add_argument("-p", "--path", default='.', 
            help="Path to use. (default: %(default)s)")

    cmdargs = p.parse_args()

    if cmdargs.input is None:
        p.print_help()
        sys.exit()

    return cmdargs

def run():
    cmdargs = getCmdargs()

    oldpath, newpath = utils.extractTarFile(cmdargs.input, cmdargs.path)

    testsRun = 0
    testsIgnoredNoDriver = 0

    if lidarprocessor.HAVE_FMT_LAS:
        testsuite1.run(oldpath, newpath)
        testsRun += 1
    else:
        testsIgnoredNoDriver += 1

    if lidarprocessor.HAVE_FMT_LAS:
        testsuite2.run(oldpath, newpath)
        testsRun += 1
    else:
        testsIgnoredNoDriver += 1

    if lidarprocessor.HAVE_FMT_LAS:
        testsuite3.run(oldpath, newpath)
        testsRun += 1
    else:
        testsIgnoredNoDriver += 1

    testsuite4.run(oldpath, newpath)
    testsRun += 1

    if lidarprocessor.HAVE_FMT_LAS:
        testsuite5.run(oldpath, newpath)
        testsRun += 1
    else:
        testsIgnoredNoDriver += 1

    print(testsRun, 'tests run successfully')
    print(testsIgnoredNoDriver, 'tests skipped because of missing format drivers')
