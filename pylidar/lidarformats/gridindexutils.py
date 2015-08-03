"""
Common utility functions for dealing with grid spatial indices
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

import numpy
from numba import jit

@jit
def updateBoolArray(boolArray, mask):
    """
    Used by readPointsForExtentByBins and writeData to update the mask of 
    elements that are to be written. Often elements need to be dropped
    since they are outside the window etc.
    """
    nBool = boolArray.shape[0]
    maskIdx = 0
    for n in range(nBool):
        if boolArray[n]:
            boolArray[n] = mask[maskIdx]
            maskIdx += 1

@jit
def flattenMaskedStructuredArray(inArray, inArrayMask, outArray):
    """
    using compressed() on a masked structured array does not
    work. Here is a workaround.
    
    inArray and inArrayMask should be 2d. outArray is 1d.
    """
    nX = inArray.shape[1]
    nY = inArray.shape[0]
    outIdx = 0
    for x in range(nX):
        for y in range(nY):
            if not inArrayMask[y, x]:
                outArray[outIdx] = inArray[y, x]
                outIdx += 1
    
    

