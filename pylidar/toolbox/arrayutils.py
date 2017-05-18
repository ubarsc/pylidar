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
from numba import jit

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

    # if oldArray was masked, make newArray masked also
    if isinstance(oldArray, numpy.ma.MaskedArray):
        # get first field so we can get the 'one mask value per element'
        # kind of mask instead of the 'mask value per field' since this 
        # would have changed
        firstField = oldArray.dtype.names[0]
        mask = oldArray[firstField].mask
        newArray = numpy.ma.MaskedArray(newArray, mask=mask)

    return newArray

@jit
def convertArgResultToIndexTuple(input, mask):
    """
    Converts the result of the numpy.ma.arg* set of functions into
    a tuple of arrays that can be used to index the original array.
    'mask' should be the mask of the result of numpy.ma.all for the same axis.
    Below is an example::

        zVals = pts['Z']
        classif = pts['CLASSIFICATION']

        idx = numpy.argmin(zVals, axis=0)
        idxmask = numpy.ma.all(zVals, axis=0)
        z, y, x = convertArgResultToIndexTuple(idx, idxmask.mask)
        
        classif[z, y, x] = 2

    """
    nrows, ncols = input.shape

    nIndices = (nrows * ncols) - mask.sum()

    zIdxs = numpy.zeros(nIndices, dtype=numpy.uint64)
    yIdxs = numpy.zeros(nIndices, dtype=numpy.uint64)
    xIdxs = numpy.zeros(nIndices, dtype=numpy.uint64)

    count = 0
    for y in range(nrows):
        for x in range(ncols):
            if not mask[y, x]:
                zIdxs[count] = input[y, x]
                yIdxs[count] = y
                xIdxs[count] = x
                count += 1

    return zIdxs, yIdxs, xIdxs
