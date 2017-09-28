
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
import shutil
import argparse
import importlib

from . import utils
# testsuite1 etc loaded dynamically below
from pylidar import lidarprocessor
from pylidar.toolbox import interpolation # so we can check we have pynninterp etc

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", help="Input tar file name")
    p.add_argument("-p", "--path", default='.', 
            help="Path to use. (default: %(default)s)")
    p.add_argument("-l", "--list", action="store_true", default=False,
            help="List tests in input, then exit")
    p.add_argument("-n", "--noremove", action="store_true", default=False,
            help="Do not clean up files on exit")
    p.add_argument("-t", "--test", action="append",
            help="Just run specified test. Can be given multiple times")
    p.add_argument("--ignore", action="append",
            help="Ignore a test. Can be given multiple times")
    p.add_argument("--noversioncheck", action="store_true", default=False,
            help="Don't do version check on supplied tar file")
    p.add_argument("--ignorefailures", action="store_true", default=False,
            help="Ignore failed tests and continue. Use with caution.")

    cmdargs = p.parse_args()

    if cmdargs.input is None:
        p.print_help()
        sys.exit()

    return cmdargs

def run():
    cmdargs = getCmdargs()

    oldpath, newpath, tests = utils.extractTarFile(cmdargs.input, cmdargs.path,
                                    not cmdargs.noversioncheck)

    if cmdargs.list:
        for name in tests:
            print(name)
        if not cmdargs.noremove:
            shutil.rmtree(oldpath)
            shutil.rmtree(newpath)
        sys.exit()

    testsRun = 0
    testsIgnored = 0
    testsIgnoredNoDriver = 0
    testsFailed = 0

    # get current package name (needed for module importing below)
    # should be pylidar.testing (remove .testall)
    arr = __name__.split('.')
    package = '.'.join(arr[:-1])
    
    for name in tests:

        if cmdargs.test is not None and name not in cmdargs.test:
            continue
        if cmdargs.ignore is not None and name in cmdargs.ignore:
            testsIgnored += 1
            continue

        # import module - should we do something better if there
        # is an error? ie we don't have the specific test asked for?
        mod = importlib.import_module('.' + name, package=package)

        # Check we can actually run this test
        doTest = True
        if hasattr(mod, 'REQUIRED_FORMATS'):
            fmts = getattr(mod, 'REQUIRED_FORMATS')
            for fmt in fmts:
                if fmt == "LAS":
                    if not lidarprocessor.HAVE_FMT_LAS:
                        print('Skipping', name, 'due to missing format driver', fmt)
                        doTest = False
                        break
                elif fmt == "RIEGL":
                    if not lidarprocessor.HAVE_FMT_RIEGL:
                        print('Skipping', name, 'due to missing format driver', fmt)
                        doTest = False
                        break
                elif fmt == "ASCIIGZ":
                    if not lidarprocessor.HAVE_FMT_ASCII_ZLIB:
                        print('Skipping', name, 'due to missing format driver', fmt)
                        doTest = False
                        break
                elif fmt == "PULSEWAVES":
                    if not lidarprocessor.HAVE_FMT_PULSEWAVES:
                        print('Skipping', name, 'due to missing format driver', fmt)
                        doTest = False
                        break
                elif fmt == "PYNNINTERP":
                    if not interpolation.havePyNNInterp:
                        print('Skipping', name, 'due to missing feature', fmt)
                        doTest = False
                        break
                elif fmt == "CGALINTERP":
                    if not interpolation.haveCGALInterpPy:
                        print('Skipping', name, 'due to missing feature', fmt)
                        doTest = False
                        break
                else:
                    msg = 'Unknown required format/feature %s' % fmt
                    raise ValueError(msg)

        if doTest:
            print('Running', name)
            if cmdargs.ignorefailures:
                try:
                    mod.run(oldpath, newpath)
                except Exception:
                    testsFailed += 1

            else:
                # just run it and let exception exit 
                mod.run(oldpath, newpath)
            testsRun += 1
        else:
            testsIgnoredNoDriver += 1

    print(testsRun, 'tests run successfully')
    if testsIgnored != 0:
        print(testsIgnored, 'test(s) ignored as directed by user')
    if testsIgnoredNoDriver != 0:
        print(testsIgnoredNoDriver, 
            'test(s) skipped because of missing format drivers or features')

    if not cmdargs.noremove:
        shutil.rmtree(oldpath)
        shutil.rmtree(newpath)

    # return error code 
    if testsFailed == 0:
        return 0
    else:
        return 1
