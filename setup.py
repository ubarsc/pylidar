#!/usr/bin/env python

"""
Install script for PyLidar
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
from __future__ import print_function

import os
from numpy.distutils.core import setup, Extension

import pylidar

def getExtraCXXFlags():
    """
    Looks at the $PYLIDAR_CXX_FLAGS environment variable.
    If it exists, this function returns a list of flags
    to be passed as the extra_compile_args argument to
    the Extension constructor.
    Otherwise None.
    """
    if 'PYLIDAR_CXX_FLAGS' in os.environ:
        return os.environ['PYLIDAR_CXX_FLAGS'].split()
    else:
        return None

def addRieglDriver(extModules, cxxFlags):
    """
    Decides if the Riegl driver is to be built. If so 
    adds the Extension class to extModules.
    """
    if 'RIVLIB_ROOT' in os.environ and 'RIWAVELIB_ROOT' in os.environ:
        print('Building Riegl Extension...')
        rivlibRoot = os.environ['RIVLIB_ROOT']
        riwavelibRoot = os.environ['RIWAVELIB_ROOT']
        rieglModule = Extension(name='pylidar.lidarformats._riegl', 
                sources=['src/riegl.cpp', 'src/pylidar.c'],
                include_dirs=[os.path.join(rivlibRoot, 'include'),
                                os.path.join(riwavelibRoot, 'include')],
                extra_compile_args=cxxFlags,
                libraries=['scanlib-mt', 'riboost_chrono-mt', 
                     'riboost_date_time-mt', 'riboost_filesystem-mt', 
                     'riboost_regex-mt', 'riboost_system-mt', 
                     'riboost_thread-mt', 'wfmifc-mt'],
                library_dirs=[os.path.join(rivlibRoot, 'lib'),
                                os.path.join(riwavelibRoot, 'lib')])
                 
        extModules.append(rieglModule)
    else:
        print('Riegl Libraries not found.')
        print('If installed set $RIVLIB_ROOT to the install location of RiVLib')
        print('and $RIWAVELIB_ROOT to the install location of the waveform extraction library (riwavelib)')

# get any C++ flags
cxxFlags = getExtraCXXFlags()
# work out if we need to build any of the C/C++ extension
# modules
externalModules = []
addRieglDriver(externalModules, cxxFlags)

setup(name='pylidar',
      version=pylidar.PYLIDAR_VERSION,
      ext_modules=externalModules,
      description='Tools for simplifying LiDAR data I/O and tools for processing.',
      packages=['pylidar', 'pylidar/lidarformats', 'pylidar/toolbox', 'pylidar/toolbox/grdfilters'],
      license='LICENSE.txt', 
      url='http://pylidar.org/')
      