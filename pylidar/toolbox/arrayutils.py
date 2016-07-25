"""
Utility functions for use with pylidar.
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

def addFieldToStructArray(oldArray, newName, newType, newData=0):
    """
    Creates a new array with all the data from oldArray, but with a new
    field as specified by newName and newType. newType should be one 
    of the numpy types (numpy.uint16 etc). 
    newData should either be a constant value, or an array of the correct
    shape and the new field will be initialised to this value.

    """
    # first get all the fields
    newdtype = []
    for key in oldArray.dtype.fields.keys():
        atype = oldArray.dtype.fields[key][0]
        newdtype.append((key, atype))

    # add our new type
    newdtype.append((newName, newType))

    # new array
    newArray = numpy.empty(oldArray.shape, dtype=newdtype)

    # copy old data over
    for key in oldArray.dtype.fields.keys():
        newArray[key] = oldArray[key]

    # new field
    newArray[newName] = newData

    return newArray

