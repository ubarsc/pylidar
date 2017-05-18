"""
A utilities to extend h5py's reading and writing of ranges of data,
specifically the ability to quickly deal with multiple ranges.
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
import numpy
import h5py
from numba import jit
import ctypes

# Need to give ourselves access to H5Sselect_hyperslab()
# within the HDF5 library so we can call it from numba
# Sadly, we also need to cope with not accessing this, in order that ReadTheDocs will still 
# be able to build to documentation. Hence the elaborate try/except madness. 
try:
    if sys.platform == 'win32':
        HDF5_DLL = ctypes.CDLL('hdf5.dll')
    elif sys.platform == 'darwin':
        HDF5_DLL = ctypes.CDLL('libhdf5.dylib')
    else:
        HDF5_DLL = ctypes.CDLL('libhdf5.so')
    H5Sselect_hyperslab = HDF5_DLL.H5Sselect_hyperslab
    # checked on 64 and 32 bits
    H5Sselect_hyperslab.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, 
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    H5Sselect_hyperslab.restype = ctypes.c_int32
except Exception:
    H5Sselect_hyperslab = None

@jit
def convertBoolToHDF5Space(boolArray, boolStart, spaceid, start, count, 
        select_hyperslab, selectSet, selectOr):
    """
    Convert boolArray and boolStart into a newly created h5.h5s.SpaceID object.

    * spaceid should be the .id of the h5.h5s.SpaceID object.
    * start and count should be empty length 1 numpy.uint64 arrays
    * select_hyperslab should be the global H5Sselect_hyperslab.
    * selectSet and selectOr shold be the value of h5py.h5s.SELECT_SET and h5py.h5s.SELECT_OR respectively.

    """
    nVals = boolArray.shape[0]
    op = selectSet
    start[0] = boolStart

    startRange = 0
    state = boolArray[0]
    for n in range(1, nVals):
        if state != boolArray[n]:
            if not boolArray[n]:
                # end of a range
                count[0] = n - startRange
                select_hyperslab(spaceid, op, start.ctypes.data, 
                            0, count.ctypes.data, 0)
                op = selectOr
                state = boolArray[n]
            else:
                # start of a range
                start[0] = boolStart + n
                startRange = n
                state = boolArray[n]
                
    if boolArray[nVals - 1]:
        # ended on a true - need to end range
        count[0] = nVals - startRange
        if count[0] > 0:
            select_hyperslab(spaceid, op, start.ctypes.data, 
                                    0, count.ctypes.data, 0)


@jit
def updateFromBool(spaceid, boolStart, boolArray, mask, start, count, 
                select_hyperslab, selectNotb):
    """
    Given a Space and the original boolArray with a mask that applies
    when boolArray == True update both the mask and the Space.
    
    selectNotb should be h5py.h5s.SELECT_NOTB
    """
    nVals = boolArray.shape[0]
    state = boolArray[0]
    state = False
    startRange = 0
    nM = 0 # mask
    lastN = 0 # for mask
    for n in range(nVals):
        # only consider when True in the original mask
        if boolArray[n]:
            # First True element - init state
            if nM == 0:
                state = mask[nM]
                startRange = n
                start[0] = boolStart + n
            # change of state
            elif state != mask[nM]:
                if mask[nM]:
                    # end of a range of Falses
                    count[0] = n - startRange
                    select_hyperslab(spaceid, selectNotb, start.ctypes.data, 
                                0, count.ctypes.data, 0)
                    state = mask[nM]
                else:
                    # end of a range of Trues
                    start[0] = boolStart + n
                    startRange = n
                    state = mask[nM]
            lastN = n
            nM += 1
            boolArray[n] = state
            
    if nM > 0 and not mask[nM - 1]:
        # ended on a False - must close the range
        count[0] = (lastN+1) - startRange
        select_hyperslab(spaceid, selectNotb, start.ctypes.data,
            0, count.ctypes.data, 0)

def createSpaceFromRange(start, end, size):
    """
    Creates a H5Space object given the start and end of a range
    """
    boolArray = numpy.ones((end - start), dtype=numpy.bool)
    space = H5Space(size, boolArray, start)
    return space

class H5Space(object):
    """
    Object that wraps a h5py.h5s.SpaceID object and allows 
    conversion quickly from boolean arrays used elsewhere.
    Also contains methods for reading and writing to/from
    h5py datasets.
    
    """
    def __init__(self, size, boolArray=None, boolStart=None, indices=None):
        """
        size is the size of the dataset this object will be used with
        boolArray is a boolean array that is True for elements to be read.
        boolStart is the index into the start of the dataset where boolArray 
            begins.

        indices is an array containing the indices that need to be selected.

        Pass either boolArray and boolStart or indices but not all 3.
        """
        # create the space object
        self.space = h5py.h5s.create_simple((size,), (size,))
        # default is all selected - reset to none in case boolArray all False
        self.space.select_none()

        if boolArray is not None and boolStart is not None:
            # convert the bool array into it
            start = numpy.empty(1, dtype=numpy.uint64)
            count = numpy.empty(1, dtype=numpy.uint64)
            if boolArray.size > 0:
                convertBoolToHDF5Space(boolArray, boolStart, self.space.id, 
                    start, count, H5Sselect_hyperslab, h5py.h5s.SELECT_SET, 
                    h5py.h5s.SELECT_OR)
        
            # grab these for updateBoolArray()    
            self.boolArray = boolArray
            self.boolStart = boolStart
            self.indices = None

        elif indices is not None:
            if len(indices) > 0:
                # this step is necessary otherwise you get the error:
                # Coordinate array must have shape (<npoints>, 1)
                indices = numpy.expand_dims(indices, axis=1)
                self.space.select_elements(indices)
            self.indices = indices
            self.boolArray = None
            self.boolStart = None

        else:
            msg = 'Need to specify either boolArray and boolStart or indices'
            raise ValueError(msg)

    def read(self, dataSet):
        """
        Given a h5py dataset read the data ranges selected and return a
        numpy array.
        """
        # create an empty array of the right size
        npoints = self.space.get_select_npoints()
        data = numpy.empty(npoints, dtype=dataSet.dtype)

        if npoints > 0:        
            # create a 'mspace' which is the size of the array
            mspace = h5py.h5s.create_simple(data.shape, data.shape)
        
            # read it
            dataSet.id.read(mspace, self.space, data)
        return data
        
    def write(self, dataSet, data):
        """
        Given a h5py dataset and a numpy array write the array into the
        ranges selected.
        """
        if data.size > 0:
            # create the mpspace
            mspace = h5py.h5s.create_simple(data.shape, data.shape)
        
            # write it
            dataSet.id.write(mspace, self.space, data)
        
    def updateBoolArray(self, mask):
        """
        Update the h5py.h5s.SpaceID object (and cached boolArray)
        with the mask which applies to the current selection.
        """
        #if mask.size != self.space.get_select_npoints():
        #    raise ValueError('mask is wrong size')
            
        if self.boolStart is not None and self.boolArray is not None:
            start = numpy.empty(1, dtype=numpy.uint64)
            count = numpy.empty(1, dtype=numpy.uint64)
            updateFromBool(self.space.id, self.boolStart, self.boolArray, mask, 
                start, count, H5Sselect_hyperslab, h5py.h5s.SELECT_NOTB)
        else:
            # indices
            self.indices = self.indices[mask]
            self.space.select_none()
            self.space.select_elements(self.indices)

    def getSelectionSize(self):
        """
        Return the number of elements that are cuurently selected
        """
        return self.space.get_select_npoints()
        
    def getSelectedIndices(self):
        """
        Return the selected indices, mainly for used my advanced spatial 
        indices. Returns self.indices if set, otherwise works it out form 
        boolArray etc.
        """
        if self.indices is not None:
            return self.indices
        else:
            indices = numpy.arange(self.boolStart, 
                            self.boolStart + len(self.boolArray), 
                            dtype=numpy.uint64)

            return indices[self.boolArray]
