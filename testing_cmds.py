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

# update the TESTSUITE_VERSION in pylidar/testing/utils.py AND INSTALL
# before running this

from __future__ import print_function, division

import os
import sys
import glob
import json
import shlex
import tarfile
import subprocess

VERSION_TXT = 'version.txt'

def runCmd(cmd):
    """
    Print a command then run it via subprocess
    """
    print(cmd)
    subprocess.call(shlex.split(cmd))

# remove the location of this script from sys.path so we get the 
# actual installed pylidar rather than the files in the hg sandbox
del sys.path[0]

from pylidar.testing.utils import TESTSUITE_VERSION, TESTDATA_DIR

if os.path.split(os.getcwd())[1] != TESTDATA_DIR:
    msg = ("This script should be run in a directory called %s that contains " +
            "the test files") % TESTDATA_DIR
    raise SystemExit(msg)

TARFILE = 'testdata_%d.tar.bz2' % TESTSUITE_VERSION

for wildcard in ('*.spd', '*.img', '*.xml'):
    for fname in glob.glob(wildcard):
        os.remove(fname)

# output of testsuite5 that checks we can export to LAS
# clobber so we can start from scratch (can't add *.las to above
# since some of the inputs are .las)
if os.path.exists('testsuite5.las'):
    os.remove('testsuite5.las')

# same for testsuite19 and export to PulseWaves
if os.path.exists('testsuite19.pls'):
    os.remove('testsuite19.pls')
if os.path.exists('testsuite19.wvs'):
    os.remove('testsuite19.wvs')

# create version.txt info
versionInfo = {'version' : TESTSUITE_VERSION}

# list of names that we save in version.txt
nameList = []

# testsuite1
cmd = ('pylidar_translate -i apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba.las ' +
    '-o testsuite1.spd --epsg 28356 -f SPDV4 --pulseindex FIRST_RETURN --buildpulses')
runCmd(cmd)
cmd = 'pylidar_index -i testsuite1.spd -o testsuite1_idx.spd -r 2.0'
runCmd(cmd)
cmd = 'pylidar_rasterize -i testsuite1_idx.spd -o testsuite1.img -f numpy.ma.min -a Z'
runCmd(cmd)
nameList.append('testsuite1')

# testsuite2 
cmd = ('pylidar_translate -i apl1dr_x510000ys6945000z56_2009_ba1m6_pbrisba.las ' +
    '-o testsuite2.spd --epsg 28356 -f SPDV4 --pulseindex FIRST_RETURN --buildpulses')
runCmd(cmd)
cmd = 'pylidar_index -i testsuite2.spd -o testsuite2_idx.spd -r 2.0'
runCmd(cmd)
cmd = ('pylidar_rasterize -i testsuite1_idx.spd testsuite2_idx.spd ' +
    '-o testsuite2.img -f numpy.ma.min -a Z --binsize 3.0 --footprint UNION')
runCmd(cmd)
nameList.append('testsuite2')

# testsuite3
cmd = ('laszip -i apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba.las ' +
    '-o apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba_zip.laz')
runCmd(cmd)
cmd = 'lasindex -i apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba_zip.laz'
runCmd(cmd)
cmd = ('pylidar_translate -i apl1dr_x509000ys6945000z56_2009_ba1m6_pbrisba_zip.laz ' +
    '-o testsuite3.spd --epsg 28356 -f SPDV4 --pulseindex FIRST_RETURN ' +
    '--spatial --binsize=2.0 --buildpulses')
runCmd(cmd)
cmd = 'pylidar_rasterize -i testsuite3.spd -o testsuite3.img -f numpy.ma.min -a Z'
runCmd(cmd)
nameList.append('testsuite3')

# testsuite4
from pylidar.testing import testsuite4
testsuite4.run('.', '.')
nameList.append('testsuite4')

# testsuite5
cmd = 'pylidar_translate -i testsuite1.spd -o testsuite5.las -f LAS'
runCmd(cmd)
nameList.append('testsuite5')

# testsuite6
from pylidar.testing import testsuite6
testsuite6.run('.', '.')
nameList.append('testsuite6')

# testsuite6b
from pylidar.testing import testsuite6b
testsuite6b.run('.', '.')
nameList.append('testsuite6b')

# testsuite7
from pylidar.testing import testsuite7 
testsuite7.run('.', '.')
nameList.append('testsuite7')

# testsuite8
from pylidar.testing import testsuite8
testsuite8.run('.', '.')
nameList.append('testsuite8')

# testsuite9
from pylidar.testing import testsuite9
testsuite9.run('.', '.')
nameList.append('testsuite9')

# testsuite10
from pylidar.testing import testsuite10
testsuite10.run('.', '.')
nameList.append('testsuite10')

# testsuite11
from pylidar.testing import testsuite11
testsuite11.run('.', '.')
nameList.append('testsuite11')

# testsuite12
from pylidar.testing import testsuite12
testsuite12.run('.', '.')
nameList.append('testsuite12')

# testsuite13
from pylidar.testing import testsuite13
testsuite13.run('.', '.')
nameList.append('testsuite13')

# testsuite14
from pylidar.testing import testsuite14
testsuite14.run('.', '.')
nameList.append('testsuite14')

# testsuite15
from pylidar.testing import testsuite15
testsuite15.run('.', '.')
nameList.append('testsuite15')

# testsuite16
from pylidar.testing import testsuite16
testsuite16.run('.', '.')
nameList.append('testsuite16')

# testsuite16b
from pylidar.testing import testsuite16b
testsuite16b.run('.', '.')
nameList.append('testsuite16b')

# testsuite17
from pylidar.testing import testsuite17
testsuite17.run('.', '.')
nameList.append('testsuite17')

# testsuite18
from pylidar.testing import testsuite18
testsuite18.run('.', '.')
nameList.append('testsuite18')

# testsuite19
from pylidar.testing import testsuite19
testsuite19.run('.', '.')
nameList.append('testsuite19')

# testsuite20
from pylidar.testing import testsuite20
testsuite20.run('.', '.')
nameList.append('testsuite20')

# testsuite21
from pylidar.testing import testsuite21
testsuite21.run('.', '.')
nameList.append('testsuite21')

# testsuite22
from pylidar.testing import testsuite22
testsuite22.run('.', '.')
nameList.append('testsuite22')

# testsuite23
from pylidar.testing import testsuite23
testsuite23.run('.', '.')
nameList.append('testsuite23')

# testsuite23b
from pylidar.testing import testsuite23b
testsuite23b.run('.', '.')
nameList.append('testsuite23b')

# add our list of tests
versionInfo['tests'] = nameList

# create version.txt
fh = open(VERSION_TXT, 'w')
fh.write(json.dumps(versionInfo))
fh.close()

# PAM files created by GDAL but not needed by us
for fname in glob.glob('*.xml'):
    os.remove(fname)

# change to one level up and create the tar file
os.chdir('..')
fh = tarfile.open(TARFILE, 'w:bz2')
fh.add(TESTDATA_DIR, recursive=True)
fh.close()
