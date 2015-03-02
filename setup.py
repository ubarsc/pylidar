#!/usr/bin/env python

"""
Install script for PyLidar
"""
# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Neil Flood and Sam Gillingham
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

from distutils.core import setup

import pylidar

setup(name='rios',
      version=pylidar.PYLIDAR_VERSION,
      description='Raster Input/Output Simplification',
      packages=['pylidar', 'pylidar/lidarformats'],
      license='LICENSE.txt', 
      url='https://bitbucket.org/chchrsc/pylidar/overview')
      