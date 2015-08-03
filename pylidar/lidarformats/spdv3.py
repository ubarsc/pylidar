
"""
SPD V3 format driver and support functions
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

import sys
import copy
import numpy
import h5py
from numba import jit
from rios import pixelgrid
from . import generic
from . import gridindexutils

# so we can check the user has passed in expected array type
PULSE_DTYPE = numpy.dtype([('GPS_TIME', '<u8'), ('PULSE_ID', '<u8'), 
('X_ORIGIN', '<f8'), ('Y_ORIGIN', '<f8'), ('Z_ORIGIN', '<f4'), 
('H_ORIGIN', '<f4'), ('X_IDX', '<f8'), ('Y_IDX', '<f8'), ('AZIMUTH', '<f4'), 
('ZENITH', '<f4'), ('NUMBER_OF_RETURNS', 'u1'), 
('NUMBER_OF_WAVEFORM_TRANSMITTED_BINS', '<u2'), 
('NUMBER_OF_WAVEFORM_RECEIVED_BINS', '<u2'), ('RANGE_TO_WAVEFORM_START', '<f4'),
('AMPLITUDE_PULSE', '<f4'), ('WIDTH_PULSE', '<f4'), ('USER_FIELD', '<u4'), 
('SOURCE_ID', '<u2'), ('SCANLINE', '<u4'), ('SCANLINE_IDX', '<u2'), 
('RECEIVE_WAVE_NOISE_THRES', '<f4'), ('TRANS_WAVE_NOISE_THRES', '<f4'), 
('WAVELENGTH', '<f4'), ('RECEIVE_WAVE_GAIN', '<f4'), 
('RECEIVE_WAVE_OFFSET', '<f4'), ('TRANS_WAVE_GAIN', '<f4'), 
('TRANS_WAVE_OFFSET', '<f4'), ('PTS_START_IDX', '<u8'), 
('TRANSMITTED_START_IDX', '<u8'), ('RECEIVED_START_IDX', '<u8')])

POINT_DTYPE = numpy.dtype([('RETURN_ID', 'u1'), ('GPS_TIME', '<f8'), 
('X', '<f8'), ('Y', '<f8'), ('Z', '<f4'), ('HEIGHT', '<f4'), ('RANGE', '<f4'), 
('AMPLITUDE_RETURN', '<f4'), ('WIDTH_RETURN', '<f4'), ('RED', '<u2'), 
('GREEN', '<u2'), ('BLUE', '<u2'), ('CLASSIFICATION', 'u1'), 
('USER_FIELD', '<u4'), ('IGNORE', 'u1'), ('WAVE_PACKET_DESC_IDX', '<i2'), 
('WAVEFORM_OFFSET', '<u4')])

HEADER_FIELDS = {'AZIMUTH_MAX' : numpy.float64, 'AZIMUTH_MIN' : numpy.float64,
'BANDWIDTHS' : numpy.float32, 'BIN_SIZE' : numpy.float32,
'BLOCK_SIZE_POINT' : numpy.uint16, 'BLOCK_SIZE_PULSE' : numpy.uint16,
'BLOCK_SIZE_RECEIVED' : numpy.uint16, 'BLOCK_SIZE_TRANSMITTED' : numpy.uint16,
'CAPTURE_DAY_OF' : numpy.uint16, 'CAPTURE_HOUR_OF' : numpy.uint16,
'CAPTURE_MINUTE_OF' : numpy.uint16, 'CAPTURE_MONTH_OF' : numpy.uint16,
'CAPTURE_SECOND_OF' : numpy.uint16, 'CAPTURE_YEAR_OF' : numpy.uint16,
'CREATION_DAY_OF' : numpy.uint16, 'CREATION_HOUR_OF' : numpy.uint16,
'CREATION_MINUTE_OF' : numpy.uint16, 'CREATION_MONTH_OF' : numpy.uint16,
'CREATION_SECOND_OF' : numpy.uint16, 'CREATION_YEAR_OF' : numpy.uint16,
'DEFINED_DECOMPOSED_PT' : numpy.int16, 'DEFINED_DISCRETE_PT' : numpy.int16,
'DEFINED_HEIGHT' : numpy.int16, 'DEFINED_ORIGIN' : numpy.int16,
'DEFINED_RECEIVE_WAVEFORM' : numpy.int16, 'DEFINED_RGB' : numpy.int16,
'DEFINED_TRANS_WAVEFORM' : numpy.int16, 'FIELD_OF_VIEW' : numpy.float32,
'FILE_SIGNATURE' : bytes, 'FILE_TYPE' : numpy.uint16,
'GENERATING_SOFTWARE' : bytes, 'INDEX_TYPE' : numpy.uint16,
'NUMBER_BINS_X' : numpy.uint32, 'NUMBER_BINS_Y' : numpy.uint32,
'NUMBER_OF_POINTS' : numpy.uint64, 'NUMBER_OF_PULSES' : numpy.uint64,
'NUM_OF_WAVELENGTHS' : numpy.uint16, 'POINT_DENSITY' : numpy.float32,
'PULSE_ALONG_TRACK_SPACING' : numpy.float32, 
'PULSE_ANGULAR_SPACING_AZIMUTH' : numpy.float32,
'PULSE_ANGULAR_SPACING_ZENITH' : numpy.float32,
'PULSE_CROSS_TRACK_SPACING' : numpy.float32, 'PULSE_DENSITY' : numpy.float32,
'PULSE_ENERGY' : numpy.float32, 'PULSE_FOOTPRINT' : numpy.float32,
'PULSE_INDEX_METHOD' : numpy.uint16, 'RANGE_MAX' : numpy.float64,
'RANGE_MIN' : numpy.float64, 'RETURN_NUMBERS_SYN_GEN' : numpy.int16,
'SCANLINE_IDX_MAX' : numpy.float64, 'SCANLINE_IDX_MIN' : numpy.float64,
'SCANLINE_MAX' : numpy.float64, 'SCANLINE_MIN' : numpy.float64,
'SENSOR_APERTURE_SIZE' : numpy.float32, 'SENSOR_BEAM_DIVERGENCE' : numpy.float32,
'SENSOR_HEIGHT' : numpy.float64, 'SENSOR_MAX_SCAN_ANGLE' : numpy.float32,
'SENSOR_PULSE_REPETITION_FREQ' : numpy.float32,
'SENSOR_SCAN_RATE' : numpy.float32, 'SENSOR_SPEED' : numpy.float32,
'SENSOR_TEMPORAL_BIN_SPACING' : numpy.float64, 'SPATIAL_REFERENCE' : bytes,
'SYSTEM_IDENTIFIER' : bytes, 'USER_META_DATA' : bytes,
'VERSION_MAJOR_SPD' : numpy.uint16, 'VERSION_MINOR_SPD' : numpy.uint16,
'VERSION_POINT' : numpy.uint16, 'VERSION_PULSE' : numpy.uint16,
'WAVEFORM_BIT_RES' : numpy.uint16, 'WAVELENGTHS' : numpy.float32,
'X_MAX' : numpy.float64, 'X_MIN' : numpy.float64, 'Y_MAX' : numpy.float64,
'Y_MIN' : numpy.float64, 'ZENITH_MAX' : numpy.float64, 
'ZENITH_MIN' : numpy.float64, 'Z_MAX' : numpy.float64, 'Z_MIN' : numpy.float64}

HEADER_ARRAY_FIELDS = ('BANDWIDTHS', 'WAVELENGTHS')

# types for the spatial index
SPDV3_SI_COUNT_DTYPE = numpy.uint32
SPDV3_SI_INDEX_DTYPE = numpy.uint64

# types of indexing in the file
SPDV3_INDEX_CARTESIAN = 1
SPDV3_INDEX_SPHERICAL = 2
SPDV3_INDEX_CYLINDRICAL = 3
SPDV3_INDEX_POLAR = 4
SPDV3_INDEX_SCAN = 5
    
@jit
def flatten3dMaskedArray(flatArray, in3d, mask3d, idx3d):
    """
    Used by writeData to flatten out masked 3d data into a 1d
    using the indexes and masked saved from when the array was created.
    """
    (maxPts, nRows, nCols) = in3d.shape
    for n in range(maxPts):
        for row in range(nRows):
            for col in range(nCols):
                if not mask3d[n, row, col]:
                    idx = idx3d[n, row, col]
                    val = in3d[n, row, col]
                    flatArray[idx] = val

@jit
def flatten2dMaskedArray(flatArray, in2d, mask2d, idx2d):
    """
    Used by writeData to flatten out masked 2d data into a 1d
    using the indexes and masked saved from when the array was created.
    """
    (maxPts, nRows) = in2d.shape
    for n in range(maxPts):
        for row in range(nRows):
            if not mask2d[n, row]:
                idx = idx2d[n, row]
                val = in2d[n, row]
                flatArray[idx] = val

    
@jit
def BuildSpatialIndexInternal(binNum, sortedBinNumNdx, si_start, si_count):
    """
    Internal function used by SPDV3File.CreateSpatialIndex.
    
    Fills in the spatial index from the sorted bins created by CreateSpatialIndex.
    
    binNum is row * nCols + col where row and col are arrays of the row and col 
        of each element being spatially indexed.
    sortedBinNumNdx is argsorted binNum    
    si_start and si_count are output spatial indices
    """
    nCols = si_start.shape[1]
    nRows = si_start.shape[0]
    nThings = binNum.shape[0]
    for i in range(nThings):
        # get the bin of the sorted location of this element
        bn = binNum[sortedBinNumNdx[i]]
        # extract the row and column
        row = bn // nCols
        col = bn % nCols
        # range check in case outside of the bouds of index 
        # being created - not needed anymore as caller does this
        #if row >= 0 and col >= 0 and row < nRows and col < nCols:    
        if si_count[row, col] == 0:
            # first element of a new bin - save the start 
            # rest of elements in this bin should be next
            # since it is sorted
            si_start[row, col] = i
        # update count of elements
        si_count[row, col] += 1
    
@jit
def convertIdxBool2D(start_idx_array, count_array, outBool, outRow, outCol, 
                        outIdx, counts, outMask):
    """
    Internal function called by convertSPDIdxToReadIdxAndMaskInfo
    
    Convert SPD's default regular spatial index of pulse offsets and pulse 
    counts per bin into a boolean array (for passing to h5py for reading data), 
    an array of indexes and a mask for creating a 3d masked array of the data 
    extracted.
    
    Note: indexes returned are relative to the subset, not the file.

    start_idx_array 2d - array of start indexes
       input - from the SPD spatial index
    count_array 2d - array of counts
       input - from the SPD spatial index
    outBool 1d - same shape as the dataset size, but bool inited to False
       for passing to h5py for reading data
    outIdx 3d - (max(count_array), nRows, nCols) int32 inited to 0
       for constructing a masked array - relative to subset size
    outMask 3d - bool same shape as outIdx inited to True
       for constructing a masked array. Result will be False were valid data
    outRow same shape as outBool but uint32 created with numpy.empty()
       used internally only
    outCol same shape as outBool but uint32 empty()
       used internally only
    counts (nRows, nCols) int32 inited to 0
       used internally only
    """
    
    nRows = start_idx_array.shape[0]
    nCols = start_idx_array.shape[1]
    
    for col in range(nCols):
        for row in range(nRows):
            # go through each bin in the spatial index
        
            cnt = count_array[row, col]
            startidx = start_idx_array[row, col]
            for i in range(cnt):
                # work out the location in the file we need
                # to read this element from
                # seems a strange bug in numba/llvm where the
                # result of this add gets promoted to a double
                # so cast it back
                idx = int(startidx + i)
                outBool[idx] = True # tell h5py we want this one
                outRow[idx] = row   # store the row/col for this element
                outCol[idx] = col   # relative to this subset
                
    # go through again setting up the indexes and mask
    counter = 0
    n = outBool.shape[0]
    for j in range(n):
        if outBool[j]:
            # ok will be data extracted here
            # get the row and col of where it came from
            row = outRow[j]
            col = outCol[j]
            # get the current number of elements from this bin
            c = counts[row, col]
            # save the current element number at the right level
            # in the outIdx (using the current number of elements in this bin)
            outIdx[c, row, col] = counter
            # tell the masked array there is valid data here
            outMask[c, row, col] = False
            # update the current number of elements in this bin
            counts[row, col] += 1
            # update the current element number
            counter += 1

@jit
def convertIdxBool1D(start_idx_array, count_array, outBool, outRow, outIdx, 
                        counts, outMask):
    """
    Internal function called by convertSPDIdxToReadIdxAndMaskInfo
    
    Convert SPD's indexing of points from pulses, or waveforms from pulses
    into a single boolean array (for passing to h5py for reading data), 
    an array of indexes and a mask for creating a 3d masked array of the data 
    extracted.
    
    Note: indexes returned are relative to the subset, not the file.
    
    start_idx_array 1d - array of start indexes
       input - from the SPD index
    count_array 1d - array of counts
       input - from the SPD index
    outBool 1d - same shape as the dataset size, but bool inited to False
       for passing to h5py for reading data
    outIdx 2d - (max(count_array), nRows) int32 inited to 0
       for constructing a masked array - relative to subset size
    outMask 2d - bool same shape as outIdx inited to True
       for constructing a masked array. Result will be False were valid data
    outRow same shape as outBool but uint32 created with numpy.empty()
       used internally only
    counts (nRows, nCols) int32 inited to 0
       used internally only

    """
    
    nRows = start_idx_array.shape[0]
    if nRows == 0:
        return
    
    for row in range(nRows):
        # go through each bin in the index
        
        cnt = count_array[row]
        startidx = start_idx_array[row]
        for i in range(cnt):
            # work out the location in the file we need
            # to read this element from
            # seems a strange bug in numba/llvm where the
            # result of this add gets promoted to a double
            # so cast it back
            idx = int(startidx + i)
            outBool[idx] = True # tell h5py we want this one
            outRow[idx] = row # store the row for this element
                
    # go through again setting up the indexes and mask
    n = outBool.shape[0]
    counter = 0
    for j in range(n):
        if outBool[j]:
            # ok will be data extracted here
            # get the row of where it came from
            row = outRow[j]
            # get the current number of elements from this bin
            c = counts[row]
            # save the current element number at the right level
            # in the outIdx (using the current number of elements in this bin)
            outIdx[c, row] = counter
            # tell the masked array there is valid data here
            outMask[c, row] = False
            # update the current number of elements in this bin
            counts[row] += 1
            # update the current element number
            counter += 1
            
class SPDV3File(generic.LiDARFile):
    """
    Class to support reading and writing of SPD Version 3.x files.
    
    Uses h5py to handle access to the underlying HDF5 file.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)
    
        # convert mode into h5py mode string
        if mode == generic.READ:
            h5py_mode = 'r'
        elif mode == generic.UPDATE:
            h5py_mode = 'r+'
        elif mode == generic.CREATE:
            h5py_mode = 'w'
        else:
            raise ValueError('Unknown value for mode parameter')
    
        # attempt to open the file
        try:
            self.fileHandle = h5py.File(fname, h5py_mode)
        except OSError as err:
            # always seems to through an OSError
            raise generic.LiDARFormatNotUnderstood(str(err))
            
        # check that it is indeed the right version
        # and get header
        if mode == generic.READ or mode == generic.UPDATE:
            header = self.fileHandle['HEADER']
            headerKeys = header.keys()
            if (not 'VERSION_MAJOR_SPD' in headerKeys or 
                        not 'VERSION_MINOR_SPD' in headerKeys):
                msg = "File appears not to be SPD"
                raise generic.LiDARFormatNotUnderstood(msg)
            elif header['VERSION_MAJOR_SPD'][0] != 2:
                msg = "File seems to be wrong version for this driver"
                raise generic.LiDARFormatNotUnderstood(msg)
                
            self.headerDict = self.convertHeaderToDictionary(header)
            self.headerUpdated = False
        else:
            # just create a blank dictionary with the right names
            self.headerDict = {}
            for key in HEADER_FIELDS:
                cls = HEADER_FIELDS[key]
                # blank value - 0 for numbers, '' for strings
                if key in HEADER_ARRAY_FIELDS:
                    self.headerDict[key] = numpy.array([cls()])
                else:
                    self.headerDict[key] = cls() 
            self.headerUpdated = False
                
        # read in the bits I need for the spatial index
        if mode == generic.READ or mode == generic.UPDATE:
            indexKeys = self.fileHandle['INDEX'].keys()
            if 'PLS_PER_BIN' in indexKeys and 'BIN_OFFSETS' in indexKeys:
                self.si_cnt = self.fileHandle['INDEX']['PLS_PER_BIN'][...]
                self.si_idx = self.fileHandle['INDEX']['BIN_OFFSETS'][...]
                self.si_binSize = header['BIN_SIZE'][0]

                # also check the type of indexing used on this file
                self.indexType = header['INDEX_TYPE'][0]
                if self.indexType == SPDV3_INDEX_CARTESIAN:
                    self.si_xMin = header['X_MIN'][0]
                    self.si_yMax = header['Y_MAX'][0]
                    self.si_xPulseColName = 'X_IDX'
                    self.si_yPulseColName = 'Y_IDX'
                elif self.indexType == SPDV3_INDEX_SPHERICAL:
                    self.si_xMin = header['AZIMUTH_MIN'][0]
                    self.si_yMax = header['ZENITH_MIN'][0]
                    self.si_xPulseColName = 'AZIMUTH'
                    self.si_yPulseColName = 'ZENITH'
                elif self.indexType == SPDV3_INDEX_SCAN:
                    self.si_xMin = header['SCANLINE_IDX_MIN'][0]
                    self.si_yMax = header['SCANLINE_MIN'][0]
                    self.si_xPulseColName = 'SCANLINE_IDX'
                    self.si_yPulseColName = 'SCANLINE'
                else:
                    msg = 'Unsupported index type %d' % self.indexType
                    raise generic.LiDARInvalidSetting(msg)                    
                    
                # bottom right coords don't seem right (of data rather than si)
                self.si_xMax = self.si_xMin + (self.si_idx.shape[1] * self.si_binSize)
                self.si_yMin = self.si_yMax - (self.si_idx.shape[0] * self.si_binSize)
            
                self.wkt = header['SPATIAL_REFERENCE'][0]
                if sys.version_info[0] == 3:
                    self.wkt = self.wkt.decode()

            else:
                self.si_cnt = None
                self.si_idx = None
                self.si_binSize = None
                self.indexType = SPDV3_INDEX_CARTESIAN
                self.si_xMin = None
                self.si_yMax = None
                self.si_xMax = None
                self.si_yMin = None
                self.si_xPulseColName = None
                self.si_yPulseColName = None
                self.wkt = None
        else:
            # set on setPixelGrid
            self.si_cnt = None
            self.si_idx = None
            self.si_binSize = None
            self.si_xMin = None
            self.si_yMax = None
            self.si_xMax = None
            self.si_yMin = None
            self.wkt = None
            
        # so we can be clever about when to read from disk
        
        # the following is for caching reads so we don't need to 
        # keep re-reading each time the user asks. Also handy since
        # reading points requires pulses etc
        self.lastExtent = None
        self.lastPulseRange = None
        self.lastPoints = None
        self.lastPulses = None
        
        # the current extent or range for data being read
        self.extent = None
        self.pulseRange = None
        
        # bool array to pass to h5py to read/write current pts
        self.lastPointsBool = None
        # index to turn into 2d pointsbypulse
        self.lastPoints_Idx = None
        # mask for 2d pointsbypulse
        self.lastPoints_IdxMask = None
         # bool array to pass to h5py to read/write current pls
        self.lastPulsesBool = None
        # index to turn into 3d pulsebybins
        self.lastPulses_Idx = None
        # mask for 3d pulsebybins
        self.lastPulses_IdxMask = None
        # index to turn into 3d pointsbybins
        self.lastPoints3d_Idx = None
        # mask for 3d pointsbybins
        self.lastPoints3d_IdxMask = None
        # mask of the points within the current extent
        # since the spatial index is on the pulses, points can be outside
        self.lastPoints3d_InRegionMask = None
        # bool array to pass to h5py to read/write current transmitted
        self.lastTransBool = None
        # index to turn into 2d transbypulses
        self.lastTrans_Idx = None
        # mask for 2d transbypulses
        self.lastTrans_IdxMask = None
         # bool array to pass to h5py to read/write current received
        self.lastRecvBool = None
        # index to turn into 2d recvbypulses
        self.lastRecv_Idx = None
        # mask for 2d recvbypulses
        self.lastRecv_IdxMask = None
        
        self.pixGrid = None
        
        self.extentAlignedWithSpatialIndex = True
        self.unalignedWarningGiven = False
        
    @staticmethod
    def convertHeaderToDictionary(header):
        """
        Static method to convert the header returned by h5py
        into a normal dictionary
        """
        dict = {}
        headerKeys = header.keys()
        for key in headerKeys:
            value = header[key][...]
            if len(value) == 1 and key not in HEADER_ARRAY_FIELDS:
                value = value[0]
            if sys.version_info[0] == 3 and isinstance(value, bytes):
                value = value.decode()
            dict[key] = value
        return dict

    @staticmethod
    def getDriverName():
        """
        Name of this driver
        """
        return "SPDV3"

    @staticmethod
    def getTranslationDict(arrayType):
        """
        Translation dictionary between formats
        """
        dict = {}
        if arrayType == generic.ARRAY_TYPE_POINTS:
            dict[generic.FIELD_POINTS_RETURN_NUMBER] = 'RETURN_ID'
        elif arrayType == generic.ARRAY_TYPE_PULSES:
            dict[generic.FIELD_PULSES_TIMESTAMP] = 'GPS_TIME'
        return dict

    def getPixelGrid(self):
        """
        Return the pixel grid of this spatial index. 
        """
        if self.si_idx is not None:
            if self.pixGrid is None:
                pixGrid = pixelgrid.PixelGridDefn(projection=self.wkt,
                    xMin=self.si_xMin, xMax=self.si_xMax,
                    yMin=self.si_yMin, yMax=self.si_yMax,
                    xRes=self.si_binSize, yRes=self.si_binSize)
                # cache it
                self.pixGrid = pixGrid
            else:
                # return cache
                pixGrid = self.pixGrid
        else:
            # no spatial index - no pixgrid
            pixGrid = None
        return pixGrid
    
    def setPixelGrid(self, pixGrid):
        """
        Set the pixel grid on creation or update
        """
        if self.mode == generic.READ:
            msg = 'Can only set new pixel grid when updating or creating'
            raise generic.LiDARInvalidData(msg)
        self.si_binSize = pixGrid.xRes
        self.si_xMin = pixGrid.xMin
        self.si_yMax = pixGrid.yMax
        self.si_xMax = pixGrid.xMax
        self.si_yMin = pixGrid.yMin
        self.wkt = pixGrid.projection
        
        # create spatial index - assume existing one (if it exists)
        # is invalid
        if self.userClass.writeSpatialIndex:
            (nrows, ncols) = pixGrid.getDimensions()
            self.si_cnt = numpy.zeros((ncols, nrows), dtype=SPDV3_SI_COUNT_DTYPE)
            self.si_idx = numpy.zeros((ncols, nrows), dtype=SPDV3_SI_INDEX_DTYPE)
            
        # cache it
        self.pixGrid = pixGrid
    
    def setExtent(self, extent):
        """
        Set the extent to use for the *ForExtent() functions.
        """
        if not self.hasSpatialIndex():
            msg = 'Format has no spatial Index. Processing must be done non-spatially'
            raise generic.LiDARInvalidSetting(msg)
        self.extent = extent
        
        # need to check that the given extent is on the same grid as the 
        # spatial index. If not a new spatial index will have to be calculated
        # for each block before we can access the data.
        totalPixGrid = self.getPixelGrid()
        extentPixGrid = pixelgrid.PixelGridDefn(xMin=extent.xMin, 
                xMax=extent.xMax, yMin=extent.yMin, yMax=extent.yMax,
                xRes=extent.binSize, yRes=extent.binSize, projection=totalPixGrid.projection)
        
        self.extentAlignedWithSpatialIndex = (
                extentPixGrid.alignedWith(totalPixGrid) and 
                extent.binSize == totalPixGrid.xRes)
        
        if (not self.extentAlignedWithSpatialIndex and 
                    not self.unalignedWarningGiven):
            msg = """Extent not on same grid or resolution as file.
spatial index will be recomputed on the fly"""
            self.controls.messageHandler(msg, generic.MESSAGE_INFORMATION)
            self.unalignedWarningGiven = True
        
    @staticmethod
    def subsetColumns(array, colNames):
        """
        Internal method. Subsets the given column names from the array and
        returns it. colNames can be either a string or a sequence of column
        names. If None the input array is returned.
        """
        if colNames is not None:
            if isinstance(colNames, str):
                array = array[colNames]
            else:
                # assume a sequence. For some reason numpy
                # doesn't like a tuple here
                colNames = list(colNames)
                
                # need to check that all the named columns
                # actually exist in the structured array.
                # Numpy gives no error/warning if they do not
                # just simply ignores ones that don't exist.
                existingNames = array.dtype.fields.keys()
                for col in colNames:
                    if col not in existingNames:
                        msg = 'column %s does not exist for this format' % col
                        raise generic.LiDARArrayColumnError(msg)
                
                # have to do a copy to avoid numpy warning
                # that updating returned array will break in future
                # numpy release.
                array = array[colNames].copy()
            
        return array
    
    def readPointsForExtent(self, colNames=None):
        """
        Read out the points for the given extent as a 1d structured array.
        """
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == self.extent and 
                        not self.lastPoints is None):
            return self.subsetColumns(self.lastPoints, colNames)
            
        # this should also return anything cached
        pulses = self.readPulsesForExtent()
        
        nReturns = pulses['NUMBER_OF_RETURNS']
        startIdxs = pulses['PTS_START_IDX']
        
        # h5py prefers to take it's index by numpy bool array
        # of the same shape as the dataset
        # so we do this. If you give it the indices themselves
        # this must be done as a list which is slow
        nOut = self.fileHandle['DATA']['POINTS'].shape[0]
        point_bool, point_idx, point_idx_mask = self.convertSPDIdxToReadIdxAndMaskInfo(
                        startIdxs, nReturns, nOut)
        
        points = self.fileHandle['DATA']['POINTS'][point_bool]
        
        # self.lastExtent updated in readPulsesForExtent()
        # keep these indices from pulses to points - handy for the indexing 
        # functions.
        self.lastPointsBool = point_bool
        self.lastPoints = points
        self.lastPoints_Idx = point_idx
        self.lastPoints_IdxMask = point_idx_mask
        return self.subsetColumns(points, colNames)
            
    def readPulsesForExtent(self, colNames=None):
        """
        Return the pulses for the given extent as a 1d structured array
        """
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == self.extent and 
                        not self.lastPulses is None):
            return self.subsetColumns(self.lastPulses, colNames)

        # snap the extent to the grid of the spatial index
        pixGrid = self.getPixelGrid()
        xMin = pixGrid.snapToGrid(self.extent.xMin, pixGrid.xMin, pixGrid.xRes)
        xMax = pixGrid.snapToGrid(self.extent.xMax, pixGrid.xMax, pixGrid.xRes)
        yMin = pixGrid.snapToGrid(self.extent.yMin, pixGrid.yMin, pixGrid.yRes)
        yMax = pixGrid.snapToGrid(self.extent.yMax, pixGrid.yMax, pixGrid.yRes)

        # size of spatial index we need to read
        nrows = int(numpy.ceil((yMax - yMin) / self.si_binSize))
        ncols = int(numpy.ceil((xMax - xMin) / self.si_binSize))
        # add overlap 
        nrows += (self.controls.overlap * 2)
        ncols += (self.controls.overlap * 2)

        # create subset of spatial index to read data into
        cnt_subset = numpy.zeros((nrows, ncols), dtype=SPDV3_SI_COUNT_DTYPE)
        idx_subset = numpy.zeros((nrows, ncols), dtype=SPDV3_SI_INDEX_DTYPE)
        
        # work out where on the whole of file spatial index to read from
        xoff = int((xMin - self.si_xMin) / self.si_binSize)
        yoff = int((self.si_yMax - yMax) / self.si_binSize)
        xright = int(numpy.ceil((xMax - self.si_xMin) / self.si_binSize))
        xbottom = int(numpy.ceil((self.si_yMax - yMin) / self.si_binSize))
        xsize = xright - xoff
        ysize = xbottom - yoff
        
        # adjust for overlap
        xoff_margin = xoff - self.controls.overlap
        yoff_margin = yoff - self.controls.overlap
        xSize_margin = xsize + self.controls.overlap * 2
        ySize_margin = ysize + self.controls.overlap * 2
        
        # Code below adapted from rios.imagereader.readBlockWithMargin
        # Not sure if it can be streamlined for this case

        # The bounds of the whole image in the file        
        imgLeftBound = 0
        imgTopBound = 0
        imgRightBound = self.si_cnt.shape[1]
        imgBottomBound = self.si_cnt.shape[0]
        
        # The region we will, in principle, read from the file. Note that xSize_margin 
        # and ySize_margin are already calculated above
        
        # Restrict this to what is available in the file
        xoff_margin_file = max(xoff_margin, imgLeftBound)
        xoff_margin_file = min(xoff_margin_file, imgRightBound)
        xright_margin_file = xoff_margin + xSize_margin
        xright_margin_file = min(xright_margin_file, imgRightBound)
        xSize_margin_file = xright_margin_file - xoff_margin_file

        yoff_margin_file = max(yoff_margin, imgTopBound)
        yoff_margin_file = min(yoff_margin_file, imgBottomBound)
        ySize_margin_file = min(ySize_margin, imgBottomBound - yoff_margin_file)
        ybottom_margin_file = yoff_margin + ySize_margin
        ybottom_margin_file = min(ybottom_margin_file, imgBottomBound)
        ySize_margin_file = ybottom_margin_file - yoff_margin_file
        
        # How many pixels on each edge of the block we end up NOT reading from 
        # the file, and thus have to leave as null in the array
        notRead_left = xoff_margin_file - xoff_margin
        notRead_right = xSize_margin - (notRead_left + xSize_margin_file)
        notRead_top = yoff_margin_file - yoff_margin
        notRead_bottom = ySize_margin - (notRead_top + ySize_margin_file)
        
        # The upper bounds on the slices specified to receive the data
        slice_right = xSize_margin - notRead_right
        slice_bottom = ySize_margin - notRead_bottom
        
        if xSize_margin_file > 0 and ySize_margin_file > 0:
            # Now read in the part of the array which we can actually read from the file.
            # Read each layer separately, to honour the layerselection
            
            # The part of the final array we are filling
            imageSlice = (slice(notRead_top, slice_bottom), slice(notRead_left, slice_right))
            # the input from the spatial index
            siSlice = (slice(yoff_margin_file, yoff_margin_file+ySize_margin_file), 
                        slice(xoff_margin_file, xoff_margin_file+xSize_margin_file))

            cnt_subset[imageSlice] = self.si_cnt[siSlice]
            idx_subset[imageSlice] = self.si_idx[siSlice]
        
        # h5py prefers to take it's index by numpy bool array
        # of the same shape as the dataset
        # so we do this. If you give it the indices themselves
        # this must be done as a list which is slow
        nOut = self.fileHandle['DATA']['PULSES'].shape[0]
        pulse_bool, pulse_idx, pulse_idx_mask = self.convertSPDIdxToReadIdxAndMaskInfo(
                idx_subset, cnt_subset, nOut)
        pulses = self.fileHandle['DATA']['PULSES'][pulse_bool]

        if not self.extentAlignedWithSpatialIndex:
            # need to recompute subset of spatial index to bins
            # are aligned with current extent
            nrows = int(numpy.ceil((self.extent.yMax - self.extent.yMin) / 
                        self.extent.binSize))
            ncols = int(numpy.ceil((self.extent.xMax - self.extent.xMin) / 
                        self.extent.binSize))
            nrows += (self.controls.overlap * 2)
            ncols += (self.controls.overlap * 2)
            mask, sortedbins, new_idx, new_cnt = self.CreateSpatialIndex(
                    pulses[self.si_xPulseColName], pulses[self.si_yPulseColName], 
                    self.extent.binSize, 
                    self.extent.yMax, self.extent.xMin, nrows, ncols)
            # ok calculate indices on new spatial indexes
            pulse_bool, pulse_idx, pulse_idx_mask = self.convertSPDIdxToReadIdxAndMaskInfo(
                            new_idx, new_cnt, nOut)
            # re-sort the pulses to match the new spatial index
            pulses = pulses[mask]
            pulses = pulses[sortedbins]

        
        self.lastExtent = copy.copy(self.extent)
        self.lastPulses = pulses
        # keep these indices from spatial index to pulses as they are
        # handy for the ByBins functions
        self.lastPulsesBool = pulse_bool
        self.lastPulses_Idx = pulse_idx
        self.lastPulses_IdxMask = pulse_idx_mask
        self.lastPoints = None # are now invalid
        return self.subsetColumns(pulses, colNames)

    def readPulsesForExtentByBins(self, extent=None, colNames=None):
        """
        Return the pulses as a 3d structured masked array.
        """
        # if they have given us a new extent then use that
        if extent is not None:
            oldExtent = self.lastExtent
            self.setExtent(extent)
        # go and get the pulses - should returned cached if 
        # already got.
        pulses = self.readPulsesForExtent()
        # get these 'last' indices which map spatial index to pulses
        idx = self.lastPulses_Idx
        idxMask = self.lastPulses_IdxMask 
        # re-map into 3d
        pulsesByBins = pulses[idx]
        
        # set extent back to the 'normal' one for this block
        # in case they call this again without the extent param
        if extent is not None:
            self.setExtent(oldExtent)
            
        # make masked array
        pulsesByBins = self.subsetColumns(pulsesByBins, colNames)
        pulses = numpy.ma.array(pulsesByBins, mask=idxMask)
        return pulses
        
    def readPointsForExtentByBins(self, extent=None, colNames=None, 
                    indexByPulse=False, returnPulseIndex=False):
        """
        Return the points as a 3d structured masked array.
        
        Note that because the spatial index on a SPDV3 file is on pulses
        this may miss points that are attached to pulses outside the current
        extent. If this is a problem then select an overlap large enough.
        
        Pass indexByPulse=True to bin the points by the locations of the pulses
            (using X_IDX and Y_IDX rather than the locations of the points)
        Pass returnPulseIndex=True to also return a masked 3d array of 
            the indices into the 1d pulse array (as returned by 
            readPulsesForExtent())
            
        """
        # if they have given us a new extent then use that
        if extent is not None:
            oldExtent = self.lastExtent
            self.setExtent(extent)
        # have to spatially index the points 
        # since SPDV3 files have only a spatial index on pulses.
        points = self.readPointsForExtent()
        
        nrows = int((self.lastExtent.yMax - self.lastExtent.yMin) / 
                        self.lastExtent.binSize)
        ncols = int((self.lastExtent.xMax - self.lastExtent.xMin) / 
                        self.lastExtent.binSize)
                        
        # add overlap
        nrows += (self.controls.overlap * 2)
        ncols += (self.controls.overlap * 2)
        xMin = self.lastExtent.xMin - (self.controls.overlap * self.lastExtent.binSize)
        yMax = self.lastExtent.yMax + (self.controls.overlap * self.lastExtent.binSize)
        
        # create point spatial index
        if indexByPulse:
            # TODO: check if is there is a better way of going about this
            # in theory spatial index already exists but may be more work 
            # it is worth to use
            x_idx = numpy.repeat(self.lastPulses[self.si_xPulseColName],
                        self.lastPulses['NUMBER_OF_RETURNS'])
            y_idx = numpy.repeat(self.lastPulses[self.si_yPulseColName],
                        self.lastPulses['NUMBER_OF_RETURNS'])
        else:
            x_idx = points['X'] 
            y_idx = points['Y']            
        
        mask, sortedbins, idx, cnt = self.CreateSpatialIndex(
                y_idx, x_idx, self.lastExtent.binSize, 
                yMax, xMin, nrows, ncols)
                
        # for writing
        gridindexutils.updateBoolArray(self.lastPointsBool, mask)
                
        # TODO: don't really want the bool array returned - need
        # to make it optional
        nOut = len(points)
        pts_bool, pts_idx, pts_idx_mask = self.convertSPDIdxToReadIdxAndMaskInfo(
                                idx, cnt, nOut)

        points = points[mask]                  
        sortedPoints = points[sortedbins]
        
        pointsByBins = sortedPoints[pts_idx]

        # set extent back to the 'normal' one for this block
        # in case they call this again without the extent param
        if extent is not None:
            self.setExtent(oldExtent)

        self.lastPoints3d_Idx = pts_idx
        self.lastPoints3d_IdxMask = pts_idx_mask
        self.lastPoints3d_InRegionMask = mask
        
        pointsByBins = self.subsetColumns(pointsByBins, colNames)
        points = numpy.ma.array(pointsByBins, mask=pts_idx_mask)
        
        if returnPulseIndex:
            # have to generate array the same lengths as the 1d points
            # but containing the indexes of the pulses
            pulse_count = numpy.arange(0, self.lastPulses.size)
            # convert this into an array with an element for each point
            pulse_idx_1d = numpy.repeat(pulse_count, 
                            self.lastPulses['NUMBER_OF_RETURNS'])
            # mask the ones that are within the spatial index
            pulse_idx_1d = pulse_idx_1d[mask]
            # sort the right way
            sortedpulse_idx_1d = pulse_idx_1d[sortedbins]
            # turn into a 3d in the same way as the points themselves
            pulse_idx_3d = sortedpulse_idx_1d[pts_idx]
            
            # create a masked array 
            pulse_idx_3dmask = numpy.ma.array(pulse_idx_3d, mask=pts_idx_mask)
            
            # return 2 things
            return points, pulse_idx_3dmask
        else:
            # just return the points
            return points

    def readPointsByPulse(self, colNames=None):
        """
        Return a 2d masked structured array of point that matches
        the pulses.
        """
        if self.controls.spatialProcessing:
            points = self.readPointsForExtent()
        else:
            points = self.readPointsForRange()
        idx = self.lastPoints_Idx
        idxMask = self.lastPoints_IdxMask
        
        pointsByPulse = points[idx]
        pointsByPulse = self.subsetColumns(pointsByPulse, colNames)
        points = numpy.ma.array(pointsByPulse, mask=idxMask)
        return points

    @staticmethod
    def convertSPDIdxToReadIdxAndMaskInfo(start_idx_array, count_array, outSize):
        """
        Convert either a 2d SPD spatial index or 1d index (pulse to points, 
        pulse to waveform etc) information for reading with h5py and creating
        a masked array with the indices into the read subset.
        
        Parameters:
        start_idx_array is the 2 or 1d input array of file start indices from SPD
        count_array is the 2 or 1d input array of element counts from SPD
        outSize is the size of the h5py dataset to be read. 
        
        Returns:
        A 1d bool array to pass to h5py for reading the info out of the dataset.
            This will be outSize elements long
        A 3 or 2d (depending on if a 2 or 1 array was input) array containing
            indices into the new subset of the data. This array is arranged so
            that the first axis contains the indices for each bin (or pulse)
            and the other axis is the row (and col axis for 3d output)
            This array can be used to rearrange the data ready from h5py into
            a ragged array of the correct shape constaining the data from 
            each bin.
        A 3 or 2d (depending on if a 2 or 1 array was input) bool array that
            can be used as a mask in a masked array of the ragged array (above)
            of the actual data.
        """
        # create the bool array for h5py
        outBool = numpy.zeros((outSize,), dtype=numpy.bool)
        # work out the size of the first dimension of the index
        if count_array.size > 0:
            maxCount = count_array.max()
        else:
            maxCount = 0
        
        if count_array.ndim == 2:
            # 2d input - 3d output index and mask
            nRows, nCols = count_array.shape
            outIdx = numpy.zeros((maxCount, nRows, nCols), dtype=numpy.uint32)
            outMask = numpy.ones((maxCount, nRows, nCols), numpy.bool)
            # for internal use by convertIdxBool2D
            outRow = numpy.empty((outSize,), dtype=numpy.uint32)
            outCol = numpy.empty((outSize,), dtype=numpy.uint32)
            counts = numpy.zeros((nRows, nCols), dtype=numpy.uint32)
        
            convertIdxBool2D(start_idx_array, count_array, outBool, outRow, 
                            outCol, outIdx, counts, outMask)
                            
        elif count_array.ndim == 1:
            # 1d input - 2d output index and mask
            nRows = count_array.shape[0]
            outIdx = numpy.zeros((maxCount, nRows), dtype=numpy.uint32)
            outMask = numpy.ones((maxCount, nRows), numpy.bool)
            # for internal use by convertIdxBool1D
            outRow = numpy.empty((outSize,), dtype=numpy.uint32)
            counts = numpy.zeros((nRows,), dtype=numpy.uint32)
            
            convertIdxBool1D(start_idx_array, count_array, outBool, outRow,
                                outIdx, counts, outMask)
        else:
            msg = 'only 1 or 2d indexing supported'
            raise ValueError(msg)
        
        # return the arrays
        return outBool, outIdx, outMask

    def readTransmitted(self):
        """
        Return the 2d masked integer array of transmitted for each of the
        current pulses. 
        """
        if self.controls.spatialProcessing:
            pulses = self.readPulsesForExtent()
        else:
            pulses = self.readPulsesForRange()
        
        idx = pulses['TRANSMITTED_START_IDX']
        cnt = pulses['NUMBER_OF_WAVEFORM_TRANSMITTED_BINS']
        
        nOut = self.fileHandle['DATA']['TRANSMITTED'].shape[0]
        trans_bool, trans_idx, trans_idx_mask = self.convertSPDIdxToReadIdxAndMaskInfo(
                    idx, cnt, nOut)
        
        transmitted = self.fileHandle['DATA']['TRANSMITTED'][trans_bool]
        
        transByPulse = transmitted[trans_idx]
        trans_masked = numpy.ma.array(transByPulse, mask=trans_idx_mask)
        
        self.lastTransBool = trans_bool
        self.lastTrans_Idx = trans_idx
        self.lastTrans_IdxMask = trans_idx_mask
        
        return trans_masked
        
    def readReceived(self):
        """
        Return the 2d masked integer array of received for each of the
        current pulses. 
        """
        if self.controls.spatialProcessing:
            pulses = self.readPulsesForExtent()
        else:
            pulses = self.readPulsesForRange()
            
        idx = pulses['RECEIVED_START_IDX']
        cnt = pulses['NUMBER_OF_WAVEFORM_RECEIVED_BINS']
        
        nOut = self.fileHandle['DATA']['RECEIVED'].shape[0]
        recv_bool, recv_idx, recv_idx_mask = self.convertSPDIdxToReadIdxAndMaskInfo(
                    idx, cnt, nOut)
        
        received = self.fileHandle['DATA']['RECEIVED'][recv_bool]
        
        recvByPulse = received[recv_idx]
        recv_masked = numpy.ma.array(recvByPulse, mask=recv_idx_mask)
        
        self.lastRecvBool = recv_bool
        self.lastRecv_Idx = recv_idx
        self.lastRecv_IdxMask = recv_idx_mask
        
        return recv_masked
        
    def preparePulsesForWriting(self, pulses):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        if pulses.size == 0:
            return None
            
        if pulses.ndim == 3:
            # must flatten back to be 1d using the indexes
            # used to create the 3d version (pulsesbybin)
            if self.mode == generic.UPDATE:
                flatSize = self.lastPulses_Idx.max() + 1
                flatPulses = numpy.empty((flatSize,), dtype=pulses.data.dtype)
                flatten3dMaskedArray(flatPulses, pulses,
                            self.lastPulses_IdxMask, self.lastPulses_Idx)
                pulses = flatPulses
            else:
                # TODO: flatten somehow
                raise NotImplementedError()
                
        if pulses.ndim != 1:
            msg = 'Pulse array must be either 1d or 3d'
            raise generic.LiDARInvalidSetting(msg)
                        

        if self.mode == generic.UPDATE:
            # we need these for 
            # 1) inserting missing fields when they have read a subset of them
            # 2) ensuring that they x and y fields haven't been changed
            if self.controls.spatialProcessing:
                origPulses = self.readPulsesForExtent()
            else:
                origPulses = self.readPulsesForRange()
                
            for locField in (self.si_xPulseColName, self.si_yPulseColName):
                if locField in pulses.dtype.fields:
                    if (pulses[locField] != origPulses[locField]).any():
                        msg = 'Coordinate changed on update'
                        raise generic.LiDARInvalidData(msg)

            if pulses.dtype != PULSE_DTYPE:
                # passed in array does not have all the fields we need to write
                # so get the original data read 
                # copy fields from pulses into origPulses
                for fieldName in pulses.dtype.fields.keys():
                    origPulses[fieldName] = pulses[fieldName]
                    
                # change them over so we have the full data
                pulses = origPulses

        else:
            # need to check that passed in data has all the required fields
            if pulses.dtype != PULSE_DTYPE:
                msg = 'Pulse array does not have all the required fields'
                raise generic.LiDARInvalidData(msg)
                
        if self.extent is not None and self.controls.spatialProcessing:
            # if we doing spatial index we need to strip out areas in the overlap
            # self.extent is the size of the block without the overlap
            # so just strip out everything outside of it
            mask = ( (pulses[self.si_xPulseColName] >= self.extent.xMin) & 
                        (pulses[self.si_xPulseColName] <= self.extent.xMax) & 
                        (pulses[self.si_yPulseColName] >= self.extent.yMin) &
                        (pulses[self.si_yPulseColName] <= self.extent.yMax))
            pulses = pulses[mask]
            gridindexutils.updateBoolArray(self.lastPulsesBool, mask)
            
        return pulses

    def preparePointsForWriting(self, points):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        if points.size == 0:
            return None
            
        origPointsDims = points.ndim

        if points.ndim == 3:
            # must flatten back to be 1d using the indexes
            # used to create the 3d version (pointsbybin)
            if self.mode == generic.UPDATE:
                flatSize = self.lastPoints3d_Idx.max() + 1
                flatPoints = numpy.empty((flatSize,), dtype=points.data.dtype)
                flatten3dMaskedArray(flatPoints, points, 
                            self.lastPoints3d_IdxMask, self.lastPoints3d_Idx)
                points = flatPoints
            else:
                # TODO: flatten somehow                
                raise NotImplementedError()
                            
        if points.ndim == 2:
            # must flatten back to be 1d using the indexes
            # used to create the 2d version (pointsbypulses)
            if self.mode == generic.UPDATE:
                flatSize = self.lastPoints_Idx.max() + 1
                flatPoints = numpy.empty((flatSize,), dtype=points.data.dtype)
                flatten2dMaskedArray(flatPoints, points, 
                            self.lastPoints_IdxMask, self.lastPoints_Idx)
                points = flatPoints
            else:
                # TODO: flatten somehow
                raise NotImplementedError()
            
        if points.ndim != 1:
            msg = 'Point array must be either 1d, 2 or 3d'
            raise generic.LiDARInvalidData(msg)

        if self.mode == generic.UPDATE:

            if points.dtype != POINT_DTYPE:
                # we need these for 
                # 1) inserting missing fields when they have read a subset of them
                if self.controls.spatialProcessing:
                    origPoints = self.readPointsForExtent()
                else:
                    origPoints = self.readPointsForRange()

                # just the ones that are within the region
                # this makes the length of origPoints the same as 
                # that returned by pointsbybins flattened
                if origPointsDims == 3:
                    # first update the inregion stuff with the not overlap mask
                    origPoints = origPoints[self.lastPoints3d_InRegionMask]
                        
                # strip out the points that were originally outside
                # the window and within the overlap.
                # TODO: is this ok for points read in by indexByPulse=True?
                if self.controls.spatialProcessing:
                    mask = ( (origPoints['X'] >= self.extent.xMin) & 
                        (origPoints['X'] <= self.extent.xMax) &
                        (origPoints['Y'] >= self.extent.yMin) &
                        (origPoints['Y'] <= self.extent.yMax))
                    points = points[mask]
                    origPoints = origPoints[mask]
                    gridindexutils.updateBoolArray(self.lastPointsBool, mask)

                # passed in array does not have all the fields we need to write
                # so get the original data read 
                for fieldName in points.dtype.fields.keys():
                    origPoints[fieldName] = points[fieldName]

                # change them over so we have the full data
                points = origPoints

        else:
            # need to check that passed in data has all the required fields
            if points.dtype != POINT_DTYPE:
                msg = 'Point array does not have all the required fields'
                raise generic.LiDARInvalidData(msg)
        
        return points

    def prepareTransmittedForWriting(self, transmitted):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        if transmitted.size == 0:
            return None

        if transmitted.ndim != 2:
            msg = 'transmitted data must be 2d'
            raise generic.LiDARInvalidData(msg)

        if self.mode == generic.UPDATE:

            origShape = transmitted.shape

            # flatten it back to 1d so it can be written
            flatSize = self.lastTrans_Idx.max() + 1
            flatTrans = numpy.empty((flatSize,), dtype=transmitted.data.dtype)
            flatten2dMaskedArray(flatTrans, transmitted,
                self.lastTrans_IdxMask, self.lastTrans_Idx)
            transmitted = flatTrans
                
            # mask out those in the overlap using the pulses
            if self.controls.spatialProcessing:

                origPulses = self.readPulsesForExtent()
                mask = ( (origPulses[self.si_xPulseColName] >= self.extent.xMin) & 
                        (origPulses[self.si_xPulseColName] <= self.extent.xMax) & 
                        (origPulses[self.si_yPulseColName] >= self.extent.yMin) &
                        (origPulses[self.si_yPulseColName] <= self.extent.yMax))

                # Repeat the mask so that it is the same shape as the 
                # original transmitted and then flatten in the same way
                # we can then remove the transmitted outside the extent.         
                # We can't do this earlier since removing from transmitted
                # would mean the above flattening trick won't work.
                mask = numpy.expand_dims(mask, axis=0)
                mask = numpy.repeat(mask, origShape[1], axis=1)
                flatMask = numpy.empty((flatSize,), dtype=mask.dtype)
                flatten2dMaskedArray(flatMask, mask, 
                    self.lastTrans_IdxMask, self.lastTrans_Idx)
            
                transmitted = transmitted[flatMask]
                gridindexutils.updateBoolArray(self.lastTransBool, flatMask)
                
        else:
            raise NotImplementedError()
                
        return transmitted

    def prepareReceivedForWriting(self, received):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        if received.size == 0:
            return None

        if received.ndim != 2:
            msg = 'received data must be 2d'
            raise generic.LiDARInvalidData(msg)

        if self.mode == generic.UPDATE:

            origShape = received.shape

            # flatten it back to 1d so it can be written
            flatSize = self.lastRecv_Idx.max() + 1
            flatRecv = numpy.empty((flatSize,), dtype=received.data.dtype)
            flatten2dMaskedArray(flatRecv, received,
                self.lastRecv_IdxMask, self.lastRecv_Idx)
            received = flatRecv
                
            # mask out those in the overlap using the pulses
            if self.controls.spatialProcessing:

                origPulses = self.readPulsesForExtent()
                mask = ( (origPulses[self.si_xPulseColName] >= self.extent.xMin) & 
                        (origPulses[self.si_xPulseColName] <= self.extent.xMax) & 
                        (origPulses[self.si_yPulseColName] >= self.extent.yMin) &
                        (origPulses[self.si_yPulseColName] <= self.extent.yMax))

                # Repeat the mask so that it is the same shape as the 
                # original received and then flatten in the same way
                # we can then remove the received outside the extent.         
                # We can't do this earlier since removing from received
                # would mean the above flattening trick won't work.
                mask = numpy.expand_dims(mask, axis=0)
                mask = numpy.repeat(mask, origShape[0], axis=1)
                flatMask = numpy.empty((flatSize,), dtype=mask.dtype)
                flatten2dMaskedArray(flatMask, mask, 
                    self.lastRecv_IdxMask, self.lastRecv_Idx)
            
                received = received[flatMask]
                gridindexutils.updateBoolArray(self.lastRecvBool, flatMask)

        else:
            raise NotImplementedError()

        return received
    
    def writeData(self, pulses=None, points=None, transmitted=None, received=None):
        """
        Write all the updated data. Pass None for data that do not need to be updated.
        It is assumed that each parameter has been read by the reading functions
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return
            
        if pulses is not None:
            pulses = self.preparePulsesForWriting(pulses)
            
        if points is not None:
            points = self.preparePointsForWriting(points)
            
        if transmitted is not None:
            transmitted = self.prepareTransmittedForWriting(transmitted)
            
        if received is not None:
            received = self.prepareReceivedForWriting(received)
            
        if self.mode == generic.CREATE:
            # need to extend the hdf5 dataset before writing
            # TODO: do pulses always need to be provided?
            if pulses is not None:
                oldSize = self.fileHandle['DATA']['PULSES'].shape[0]
                nPulses = len(pulses)
                newSize = oldSize + nPulses
                self.fileHandle['DATA']['PULSES'].resize((newSize,))
                
            if points is not None:
                oldSize = self.fileHandle['DATA']['POINTS'].shape[0]
                nPoints = len(points)
                newSize = oldSize + nPoints
                self.fileHandle['DATA']['POINTS'].resize((newSize,))
                
            if transmitted is not None:
                oldSize = self.fileHandle['DATA']['TRANSMITTED'].shape[0]
                nTrans = len(transmitted)
                newSize = oldSize + nTrans
                self.fileHandle['DATA']['TRANSMITTED'].resize((newSize,))
                
            if received is not None:
                oldSize = self.fileHandle['DATA']['RECEIVED'].shape[0]
                nRecv = len(received)
                newSize = oldSize + nRecv
                self.fileHandle['DATA']['RECEIVED'].resize((newSize,))

            raise NotImplementedError()
                
        else:
            if points is not None:
                self.fileHandle['DATA']['POINTS'][self.lastPointsBool] = points
            if pulses is not None:
                self.fileHandle['DATA']['PULSES'][self.lastPulsesBool] = pulses
            if transmitted is not None:
                self.fileHandle['DATA']['TRANSMITTED'][self.lastTransBool] = transmitted
            if received is not None:
                self.fileHandle['DATA']['RECEIVED'][self.lastRecvBool] = received
        # TODO: now update the spatial index
        pass

    @staticmethod
    def CreateSpatialIndex(coordOne, coordTwo, binSize, coordOneMax, 
                    coordTwoMin, nRows, nCols):
        """
        Create a SPD V3 spatial index given arrays of the coordinates of 
        the elements.
        
        This can then be used for writing a SPD V3 spatial index to file,
        or to re-bin data to a new grid.
        
        Any elements outside of the new spatial index are ignored and the
        arrays returned will not refer to them.

        Parameters:
            coordOne is the coordinate corresponding to bin row. 
            coordTwo corresponds to bin col.
                Note that coordOne will always be reversed, in keeping with widespread
                conventions that a Y coordinate increases going north, but a grid row number
                increases going south. This same assumption will be applied even when
                the coordinates are not cartesian (e.g. angles). 
            binSize is the size (in world coords) of each bin. The V3 index definition
                requires that bins are square. 
            coordOneMax and coordTwoMin define the top left of the 
                spatial index to be built. This is the world coordinate of the
                top-left corner of the top-left bin
            nRows, nCols - size of the spatial index
            
        Returns:
            mask - a 1d array of bools of the valid elements. This must be applied
                before sortedBins.
            sortedBins - a 1d array of indices that is used to 
                re-sort the data into the correct order for using 
                the created spatial index. Since the spatial index puts
                all elements in the same bin together this is a very important
                step!
            si_start - a 2d array of start indexes into the sorted data (see
                above)
            si_count - the count of elements in each bin.
        """
        # work out the row and column of each element to be put into the
        # spatial index
        row = numpy.floor((coordOneMax - coordOne) / binSize)
        col = numpy.floor((coordTwo - coordTwoMin) / binSize)
        
        # work out the elements that aren't within the new spatial 
        # index and remove them
        validMask = (row >= 0) & (col >= 0) & (row < nRows) & (col < nCols)
        row = row[validMask].astype(numpy.uint32)
        col = col[validMask].astype(numpy.uint32)
        coordOne = coordOne[validMask]
        coordTwo = coordTwo[validMask]
        
        # convert this to a 'binNum' which is a combination of row and col
        # and can be sorted to make a complete ordering of the 2-d grid of bins
        binNum = row * nCols + col
        # get an array of indices of the sorted version of the bins
        sortedBinNumNdx = numpy.argsort(binNum)
    
        # output spatial index arrays
        si_start = numpy.zeros((nRows, nCols), dtype=SPDV3_SI_INDEX_DTYPE)
        si_count = numpy.zeros((nRows, nCols), dtype=SPDV3_SI_COUNT_DTYPE)
        
        # call our helper function to put the elements into the spatial index
        BuildSpatialIndexInternal(binNum, sortedBinNumNdx, si_start, si_count)
        
        # return array to get back to sorted version of the elements
        # and the new spatial index
        return validMask, sortedBinNumNdx, si_start, si_count
        
    def hasSpatialIndex(self):
        """
        Return True if we have a spatial index.
        """
        return self.si_cnt is not None
        
    # The functions below are for when there is no spatial index.
    def setPulseRange(self, pulseRange):
        """
        Set the range of pulses to read
        """
        # copy it so we can change the values if beyond the 
        # range of data
        self.pulseRange = copy.copy(pulseRange)
        nTotalPulses = self.getTotalNumberPulses()
        bMore = True
        if self.pulseRange.startPulse > nTotalPulses:
            # no data to read
            self.pulseRange.startPulse = 0
            self.pulseRange.endPulse = 0
            bMore = False
            
        elif self.pulseRange.endPulse > nTotalPulses:
            self.pulseRange.endPulse = nTotalPulses + 1
            
        return bMore
        
    def readPointsForRange(self, colNames=None):
        """
        Read all the points for the specified range of pulses
        """
        if (self.lastPulseRange is not None and
                self.lastPulseRange == self.pulseRange and
                self.lastPoints is not None):
            return self.subsetColumns(self.lastPoints, colNames)
            
        # this should return anything cached
        pulses = self.readPulsesForRange()
        
        nReturns = pulses['NUMBER_OF_RETURNS']
        startIdxs = pulses['PTS_START_IDX']
        
        # h5py prefers to take it's index by numpy bool array
        # of the same shape as the dataset
        # so we do this. If you give it the indices themselves
        # this must be done as a list which is slow
        nOut = self.fileHandle['DATA']['POINTS'].shape[0]
        point_bool, point_idx, point_idx_mask = self.convertSPDIdxToReadIdxAndMaskInfo(
                        startIdxs, nReturns, nOut)
        
        points = self.fileHandle['DATA']['POINTS'][point_bool]
        
        # keep these indices from pulses to points - handy for the indexing 
        # functions.
        self.lastPoints = points
        self.lastPoints_Idx = point_idx
        self.lastPoints_IdxMask = point_idx_mask
        # self.lastPulseRange copied in readPulsesForRange()
        return self.subsetColumns(points, colNames)
    
    def readPulsesForRange(self, colNames=None):
        """
        Read the specified range of pulses
        """
        if (self.lastPulseRange is not None and
                self.lastPulseRange == self.pulseRange and 
                self.lastPulses is not None):
            return self.subsetColumns(self.lastPulses, colNames)
    
        pulses = self.fileHandle['DATA']['PULSES'][
            self.pulseRange.startPulse:self.pulseRange.endPulse]
            
        self.lastPulses = pulses
        self.lastPulseRange = copy.copy(self.pulseRange)
        self.lastPoints = None # now invalid
        return self.subsetColumns(pulses, colNames)
    
    def getTotalNumberPulses(self):
        """
        Return the total number of pulses
        """
        return self.fileHandle['DATA']['PULSES'].shape[0]
        
    def getHeader(self):
        """
        Return our cached dictionary
        """
        return self.headerDict
        
    def setHeader(self, newHeaderDict):
        """
        Update our cached dictionary
        """
        for key in newHeaderDict.keys():
            if key not in self.headerDict:
                msg = 'Header field %s not supported in SPDV3 files' % key
                raise ValueError(msg)
            self.headerDict[key] = newHeaderDict[key]
        self.headerUpdated = True
            
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.headerDict[name]
        
    def setHeaderValue(self, name, value):
        """
        Just update one value in the header
        """
        if name not in self.headerDict:
            msg = 'Header field %s not supported in SPDV3 files' % name
            raise ValueError(msg)
        self.headerDict[name] =  value
        self.headerUpdated = True 
        
    def close(self):
        """
        Write out the spatial index, header and close file handle.
        """
        # header - note values reladed to spatial index may be overwritten
        # with the correct values below
        if self.mode == generic.CREATE or (self.headerUpdated and 
                    self.mode == generic.UPDATE):
            # loop through the dictionary we have cached and 
            # ensure correct type and write out
            for key in self.headerDict:
                value = self.headerDict[key]
                # convert to correct type
                cls = HEADER_FIELDS[key]
                try:
                    if sys.version_info[0] == 3 and cls == bytes:
                        value = value.encode()
                    value = cls(value)
                except ValueError:
                    msg = "Unable to convert field %s to the expected type"
                    msg = msg % key
                    raise generic.LiDARInvalidData(msg)
                if key in HEADER_ARRAY_FIELDS and numpy.isscalar(value):
                    value = numpy.array([value])
                    
                self.fileHandle['HEADER'][key][...] = value
                    

        if (self.mode == generic.CREATE and self.userClass.writeSpatialIndex and 
                    self.si_cnt is not None):
            # write out to file
            self.fileHandle['INDEX']['PLS_PER_BIN'][...] = self.si_cnt
            self.fileHandle['INDEX']['BIN_OFFSETS'][...] = self.si_idx
            self.fileHandle['HEADER']['BIN_SIZE'][...] = self.si_binSize
            self.fileHandle['HEADER']['X_MIN'][...] = self.si_xmin
            self.fileHandle['HEADER']['Y_MAX'][...] = self.si_ymax
            self.fileHandle['HEADER']['X_MAX'][...] = self.si_xmax
            self.fileHandle['HEADER']['Y_MIN'][...] = self.si_ymin
            self.fileHandle['HEADER']['SPATIAL_REFERENCE'][...] = self.wkt.encode()
           
            
        # close
        self.fileHandle.close()
        self.fileHandle = None        
        self.lastExtent = None
        self.lastPoints = None
        self.lastPulses = None

class SPDV3FileInfo(generic.LiDARFileInfo):
    """
    Class that gets information about a SPDV3 file
    and makes it available as fields.
    """
    def __init__(self, fname):
        generic.LiDARFileInfo.__init__(self, fname)
    
        # attempt to open the file
        try:
            fileHandle = h5py.File(fname, 'r')
        except OSError as err:
            # always seems to through an OSError
            raise generic.LiDARFormatNotUnderstood(str(err))
            
        # check that it is indeed the right version
        header = fileHandle['HEADER']
        headerKeys = header.keys()
        if (not 'VERSION_MAJOR_SPD' in headerKeys or 
                    not 'VERSION_MINOR_SPD' in headerKeys):
            msg = "File appears not to be SPD"
            raise generic.LiDARFormatNotUnderstood(msg)
        elif header['VERSION_MAJOR_SPD'][0] != 2:
            msg = "File seems to be wrong version for this driver"
            raise generic.LiDARFormatNotUnderstood(msg)
            
        # save the header as a dictionary
        self.header = SPDV3File.convertHeaderToDictionary(header)

        # pull a few things out to the top level
        self.wkt = self.header['SPATIAL_REFERENCE']

        self.zMax = self.header['Z_MAX']
        self.zMin = self.header['Z_MIN']
        self.wavelengths = self.header['WAVELENGTHS']
        self.bandwidths = self.header['BANDWIDTHS']
        # probably other things too
        
        
        
        
