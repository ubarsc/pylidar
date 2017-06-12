"""
Functions for calculating Pgap from ALS waveform data (Armston et al., 2013)
Assumes standard point fields required to calculate return energy integral are available
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

import numpy
from numba import jit

from pylidar import lidarprocessor


def run_pgap_armston2013(dataFiles, controls, otherargs, outfile):
    """
    Main function for PGAP_ARMSTON2013
    """
    pass
