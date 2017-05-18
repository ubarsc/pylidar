
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
from __future__ import print_function, division

import sys
import copy
import numpy
import h5py
from numba import jit
from rios import pixelgrid
from . import generic
from . import gridindexutils
from . import h5space

PULSE_DTYPE = numpy.dtype([('GPS_TIME', 'u8'), ('PULSE_ID', 'u8'), 
('X_ORIGIN', 'f8'), ('Y_ORIGIN', 'f8'), ('Z_ORIGIN', 'f4'), 
('H_ORIGIN', 'f4'), ('X_IDX', 'f8'), ('Y_IDX', 'f8'), ('AZIMUTH', 'f4'), 
('ZENITH', 'f4'), ('NUMBER_OF_RETURNS', 'u1'), 
('NUMBER_OF_WAVEFORM_TRANSMITTED_BINS', 'u2'), 
('NUMBER_OF_WAVEFORM_RECEIVED_BINS', 'u2'), ('RANGE_TO_WAVEFORM_START', 'f4'),
('AMPLITUDE_PULSE', 'f4'), ('WIDTH_PULSE', 'f4'), ('USER_FIELD', 'u4'), 
('SOURCE_ID', 'u2'), ('SCANLINE', 'u4'), ('SCANLINE_IDX', 'u2'), 
('RECEIVE_WAVE_NOISE_THRES', 'f4'), ('TRANS_WAVE_NOISE_THRES', 'f4'), 
('WAVELENGTH', 'f4'), ('RECEIVE_WAVE_GAIN', 'f4'), 
('RECEIVE_WAVE_OFFSET', 'f4'), ('TRANS_WAVE_GAIN', 'f4'), 
('TRANS_WAVE_OFFSET', 'f4'), ('PTS_START_IDX', 'u8'), 
('TRANSMITTED_START_IDX', 'u8'), ('RECEIVED_START_IDX', 'u8')])
"so we can check the user has passed in expected array type"

POINT_DTYPE = numpy.dtype([('RETURN_ID', 'u1'), ('GPS_TIME', 'f8'), 
('X', 'f8'), ('Y', 'f8'), ('Z', 'f4'), ('HEIGHT', 'f4'), ('RANGE', 'f4'), 
('AMPLITUDE_RETURN', 'f4'), ('WIDTH_RETURN', 'f4'), ('RED', 'u2'), 
('GREEN', 'u2'), ('BLUE', 'u2'), ('CLASSIFICATION', 'u1'), 
('USER_FIELD', 'u4'), ('IGNORE', 'u1'), ('WAVE_PACKET_DESC_IDX', 'i2'), 
('WAVEFORM_OFFSET', 'u4')])
"so we can check the user has passed in expected array type"

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
"Header fields and their types"

HEADER_ARRAY_FIELDS = ('BANDWIDTHS', 'WAVELENGTHS')
"header fields that are actually arrays"

SPDV3_SI_COUNT_DTYPE = numpy.uint32
"types for the spatial index"
SPDV3_SI_INDEX_DTYPE = numpy.uint64
"types for the spatial index"

SPDV3_INDEX_CARTESIAN = 1
"types of indexing in the file"
SPDV3_INDEX_SPHERICAL = 2
"types of indexing in the file"
SPDV3_INDEX_CYLINDRICAL = 3
"types of indexing in the file"
SPDV3_INDEX_POLAR = 4
"types of indexing in the file"
SPDV3_INDEX_SCAN = 5
"types of indexing in the file"

SPDV3_CLASSIFICATION_UNDEFINED = 0
"classification codes"
SPDV3_CLASSIFICATION_UNCLASSIFIED = 1
"classification codes"
SPDV3_CLASSIFICATION_CREATED = 2
"classification codes"
SPDV3_CLASSIFICATION_GROUND = 3
"classification codes"
SPDV3_CLASSIFICATION_LOWVEGE = 4
"classification codes"
SPDV3_CLASSIFICATION_MEDVEGE = 5
"classification codes"
SPDV3_CLASSIFICATION_HIGHVEGE = 6
"classification codes"
SPDV3_CLASSIFICATION_BUILDING = 7
"classification codes"
SPDV3_CLASSIFICATION_WATER = 8
"classification codes"
SPDV3_CLASSIFICATION_TRUNK = 9
"classification codes"
SPDV3_CLASSIFICATION_FOLIAGE = 10
"classification codes"
SPDV3_CLASSIFICATION_BRANCH = 11
"classification codes"
SPDV3_CLASSIFICATION_WALL = 12
"classification codes"
SPDV3_CLASSIFICATION_ALLCLASSES = 100
"classification codes"
SPDV3_CLASSIFICATION_ALLCLASSES_TOP = 101
"classification codes"
SPDV3_CLASSIFICATION_VEGETOP = 102
"classification codes"
SPDV3_CLASSIFICATION_VEGE = 103
"classification codes"
SPDV3_CLASSIFICATION_NOTGROUND = 104
"classification codes"
SPDV3_CLASSIFICATION_KEYGRDPTS = 105
"classification codes"

POINTS_HEADER_UPDATE_DICT = {'X' : ('X_MIN', 'X_MAX'), 'Y' : ('Y_MIN', 'Y_MAX'),
        'Z' : ('Z_MIN', 'Z_MAX'), 'RANGE' : ('RANGE_MIN', 'RANGE_MAX')}
"for updating the header"
PULSES_HEADER_UPDATE_DICT = {'ZENITH' : ('ZENITH_MIN', 'ZENITH_MAX'),
        'AZIMUTH' : ('AZIMUTH_MIN', 'AZIMUTH_MAX'), 
        'SCANLINE_IDX' : ('SCANLINE_IDX_MIN', 'SCANLINE_IDX_MAX'),
        'SCANLINE' : ('SCANLINE_MIN', 'SCANLINE_MAX')}
"for updating the header"

HEADER_TRANSLATION_DICT = {generic.HEADER_NUMBER_OF_POINTS : 'NUMBER_OF_POINTS'}
"Translation of header field names"

class SPDV3File(generic.LiDARFile):
    """
    Class to support reading and writing of SPD Version 3.x files.
    
    Uses h5py to handle access to the underlying HDF5 file.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)

        # TODO: disable the creation of SPDV3 files until 
        # more testing is done
        if mode == generic.CREATE:
            msg = 'Cannot create SPDV3 files'
            raise generic.LiDARWritingNotSupported(msg)
    
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
        except (OSError, IOError) as err:
            # always seems to through an OSError
            # found another one!
            raise generic.LiDARFormatNotUnderstood(str(err))
            
        # check that it is indeed the right version
        # and get header
        if mode == generic.READ or mode == generic.UPDATE:
            if 'HEADER' not in self.fileHandle:
                msg = "File appears not to be SPD V3"
                raise generic.LiDARFormatNotUnderstood(msg)
            header = self.fileHandle['HEADER']
            headerKeys = header.keys()
            if (not 'VERSION_MAJOR_SPD' in headerKeys or 
                        not 'VERSION_MINOR_SPD' in headerKeys):
                msg = "File appears not to be SPD V3"
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

            # set the MIN and MAX fields to the max and min values
            # possible so we notice if they are not set and we can update
            # appriately
            for updateKey in POINTS_HEADER_UPDATE_DICT.keys():
                minKey, maxKey = POINTS_HEADER_UPDATE_DICT[updateKey]
                info = numpy.finfo(HEADER_FIELDS[minKey])
                # note order
                self.headerDict[maxKey] = info.min
                self.headerDict[minKey] = info.max
            for updateKey in PULSES_HEADER_UPDATE_DICT.keys():
                minKey, maxKey = PULSES_HEADER_UPDATE_DICT[updateKey]
                info = numpy.finfo(HEADER_FIELDS[minKey])
                self.headerDict[maxKey] = info.min
                self.headerDict[minKey] = info.max
                        
            self.headerUpdated = False
                
        # read in the bits I need for the spatial index
        # need to handle case where SPDV3 does not have an index
        if mode == generic.READ or mode == generic.UPDATE:
            if 'INDEX' in self.fileHandle:
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
                    self.indexType = None
                    self.si_xMin = None
                    self.si_yMax = None
                    self.si_xMax = None
                    self.si_yMin = None
                    self.si_xPulseColName = None
                    self.si_yPulseColName = None
                    self.wkt = None
            
            else:
                # no spatial index
                self.si_cnt = None
                self.si_idx = None
                self.si_binSize = None
                self.indexType = None
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
        
        # h5space.H5Space
        self.lastPointsSpace = None
        # index to turn into 2d pointsbypulse
        self.lastPoints_Idx = None
        # mask for 2d pointsbypulse
        self.lastPoints_IdxMask = None
        # h5space.H5Space
        self.lastPulsesSpace = None
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
        # needs sorting also
        self.lastPoints3d_InRegionSort = None
        # h5space.H5Space
        self.lastTransSpace = None
        # index to turn into 2d transbypulses
        self.lastTrans_Idx = None
        # mask for 2d transbypulses
        self.lastTrans_IdxMask = None
        # h5space.H5Space
        self.lastRecvSpace = None
        # index to turn into 2d recvbypulses
        self.lastRecv_Idx = None
        # mask for 2d recvbypulses
        self.lastRecv_IdxMask = None
        
        self.pixGrid = None
        
        self.extentAlignedWithSpatialIndex = True
        self.unalignedWarningGiven = False

        # set up list for conversion of CLASSIFICATION column
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_CREATED,
                                generic.CLASSIFICATION_CREATED))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_GROUND,
                                generic.CLASSIFICATION_GROUND))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_LOWVEGE,
                                generic.CLASSIFICATION_LOWVEGE))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_MEDVEGE,
                                generic.CLASSIFICATION_MEDVEGE))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_HIGHVEGE,
                                generic.CLASSIFICATION_HIGHVEGE))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_BUILDING,
                                generic.CLASSIFICATION_BUILDING))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_WATER,
                                generic.CLASSIFICATION_WATER))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_TRUNK,
                                generic.CLASSIFICATION_TRUNK))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_FOLIAGE,
                                generic.CLASSIFICATION_FOLIAGE))
        self.classificationTranslation.append((SPDV3_CLASSIFICATION_BRANCH,
                                generic.CLASSIFICATION_BRANCH))

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

    @staticmethod
    def getHeaderTranslationDict():
        """
        Return dictionary with non-standard header names
        """
        return HEADER_TRANSLATION_DICT

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
                self.pixGrid = copy.copy(pixGrid)
            else:
                # return cache
                pixGrid = copy.copy(self.pixGrid)
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
        # is invalid. This function is only called for spatial processing anyway.
        if self.userClass.writeSpatialIndex:
            (nrows, ncols) = pixGrid.getDimensions()
            self.si_cnt = numpy.zeros((ncols, nrows), dtype=SPDV3_SI_COUNT_DTYPE)
            self.si_idx = numpy.zeros((ncols, nrows), dtype=SPDV3_SI_INDEX_DTYPE)
            
        # cache it
        self.pixGrid = pixGrid
    
    def setExtent(self, extent):
        """
        Set the extent to use for the ForExtent() functions.
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
        
        nOut = self.fileHandle['DATA']['POINTS'].shape[0]
        point_space, point_idx, point_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                        startIdxs, nReturns, nOut)
        
        points = point_space.read(self.fileHandle['DATA']['POINTS'])

        # translate any classifications
        self.recodeClassification(points, generic.RECODE_TO_LAS, colNames)
        
        # self.lastExtent updated in readPulsesForExtent()
        # keep these indices from pulses to points - handy for the indexing 
        # functions.
        self.lastPointsSpace = point_space
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
        if self.extentAlignedWithSpatialIndex:
            xMin = self.extent.xMin
            xMax = self.extent.xMax
            yMin = self.extent.yMin
            yMax = self.extent.yMax
        else:
            xMin = gridindexutils.snapToGrid(self.extent.xMin, pixGrid.xMin, 
                pixGrid.xRes, gridindexutils.SNAPMETHOD_LESS)
            xMax = gridindexutils.snapToGrid(self.extent.xMax, pixGrid.xMax, 
                pixGrid.xRes, gridindexutils.SNAPMETHOD_GREATER)
            yMin = gridindexutils.snapToGrid(self.extent.yMin, pixGrid.yMin, 
                pixGrid.yRes, gridindexutils.SNAPMETHOD_LESS)
            yMax = gridindexutils.snapToGrid(self.extent.yMax, pixGrid.yMax, 
                pixGrid.yRes, gridindexutils.SNAPMETHOD_GREATER)

        # size of spatial index we need to read
        # round() ok since points should already be on the grid, nasty 
        # rounding errors propogated with ceil()                   
        nrows = int(numpy.round((yMax - yMin) / self.si_binSize))
        ncols = int(numpy.round((xMax - xMin) / self.si_binSize))
        # add overlap 
        nrows += (self.controls.overlap * 2)
        ncols += (self.controls.overlap * 2)

        # create subset of spatial index to read data into
        cnt_subset = numpy.zeros((nrows, ncols), dtype=SPDV3_SI_COUNT_DTYPE)
        idx_subset = numpy.zeros((nrows, ncols), dtype=SPDV3_SI_INDEX_DTYPE)
        
        imageSlice, siSlice = gridindexutils.getSlicesForExtent(pixGrid, 
             self.si_cnt.shape, self.controls.overlap, xMin, xMax, yMin, yMax)

        # chop out the data             
        if imageSlice is not None and siSlice is not None:

            cnt_subset[imageSlice] = self.si_cnt[siSlice]
            idx_subset[imageSlice] = self.si_idx[siSlice]
        
        nOut = self.fileHandle['DATA']['PULSES'].shape[0]
        pulse_space, pulse_idx, pulse_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                idx_subset, cnt_subset, nOut)
        pulses = pulse_space.read(self.fileHandle['DATA']['PULSES'])

        if not self.extentAlignedWithSpatialIndex:
            # need to recompute subset of spatial index to bins
            # are aligned with current extent
            # round() ok since points should already be on the grid, nasty 
            # rounding errors propogated with ceil()         
            nrows = int(numpy.round((self.extent.yMax - self.extent.yMin) / 
                        self.extent.binSize))
            ncols = int(numpy.round((self.extent.xMax - self.extent.xMin) / 
                        self.extent.binSize))
            nrows += (self.controls.overlap * 2)
            ncols += (self.controls.overlap * 2)
            mask, sortedbins, new_idx, new_cnt = gridindexutils.CreateSpatialIndex(
                    pulses[self.si_yPulseColName], pulses[self.si_xPulseColName], 
                    self.extent.binSize, 
                    self.extent.yMax, self.extent.xMin, nrows, ncols, 
                    SPDV3_SI_INDEX_DTYPE, SPDV3_SI_COUNT_DTYPE)
            # ok calculate indices on new spatial indexes
            pulse_space, pulse_idx, pulse_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                            new_idx, new_cnt, nOut)
            # re-sort the pulses to match the new spatial index
            pulses = pulses[mask]
            pulses = pulses[sortedbins]

        
        self.lastExtent = copy.copy(self.extent)
        self.lastPulses = pulses
        # keep these indices from spatial index to pulses as they are
        # handy for the ByBins functions
        self.lastPulsesSpace = pulse_space
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
        
        mask, sortedbins, idx, cnt = gridindexutils.CreateSpatialIndex(
                y_idx, x_idx, self.lastExtent.binSize, 
                yMax, xMin, nrows, ncols, SPDV3_SI_INDEX_DTYPE, SPDV3_SI_COUNT_DTYPE)
                
        pts_idx, pts_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                idx, cnt)

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
        self.lastPoints3d_InRegionSort = sortedbins
        
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

    def readWaveformInfo(self):
        """
        Return 2d masked array of information about
        the waveforms.
        """
        if self.controls.spatialProcessing:
            pulses = self.readPulsesForExtent()
        else:
            pulses = self.readPulsesForRange()

        trans_gain = pulses['TRANS_WAVE_GAIN']
        trans_offset = pulses['TRANS_WAVE_OFFSET']
        recv_gain = pulses['RECEIVE_WAVE_GAIN']
        recv_offset = pulses['RECEIVE_WAVE_OFFSET']
        rangeToStart = pulses['RANGE_TO_WAVEFORM_START']
        
        # create an empty array and copy info in
        infoDtype = [('TRANS_WAVE_GAIN', trans_gain.dtype), 
            ('TRANS_WAVE_OFFSET', trans_offset.dtype), 
            ('RECEIVE_WAVE_GAIN', recv_gain.dtype),
            ('RECEIVE_WAVE_OFFSET', recv_offset.dtype),
            ('RANGE_TO_WAVEFORM_START', rangeToStart.dtype)]
        info = numpy.empty((1, trans_gain.size), dtype=infoDtype)
        info[0]['TRANS_WAVE_GAIN'] = trans_gain
        info[0]['TRANS_WAVE_OFFSET'] = trans_gain
        info[0]['RECEIVE_WAVE_GAIN'] = trans_gain
        info[0]['RECEIVE_WAVE_OFFSET'] = trans_gain
        info[0]['RANGE_TO_WAVEFORM_START'] = rangeToStart
        
        # mask where there is no data
        mask = ((pulses['NUMBER_OF_WAVEFORM_TRANSMITTED_BINS'] == 0) & 
                    (pulses['NUMBER_OF_WAVEFORM_RECEIVED_BINS'] == 0))
        mask = numpy.expand_dims(mask, axis=0)
        
        return numpy.ma.array(info, mask=mask)
        

    def readTransmitted(self):
        """
        Return the 3d masked integer array of transmitted for each of the
        current pulses.
        SPDV3 only has 1 transmitted per pulse so the second axis is empty.
        First axis is waveform bin and last is pulse.
        """
        # TODO: cache?
        if self.controls.spatialProcessing:
            pulses = self.readPulsesForExtent()
        else:
            pulses = self.readPulsesForRange()
        
        idx = pulses['TRANSMITTED_START_IDX']
        cnt = pulses['NUMBER_OF_WAVEFORM_TRANSMITTED_BINS']
        gain = pulses['TRANS_WAVE_GAIN']
        offset = pulses['TRANS_WAVE_OFFSET']
        
        nOut = self.fileHandle['DATA']['TRANSMITTED'].shape[0]
        trans_space, trans_idx, trans_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                    idx, cnt, nOut)
        
        transmitted = trans_space.read(self.fileHandle['DATA']['TRANSMITTED'])

        # add a dummy axis
        trans_idx = numpy.expand_dims(trans_idx, 1)
        trans_idx_mask = numpy.expand_dims(trans_idx_mask, 1)
        
        # reshape
        transByPulse = transmitted[trans_idx]
        # apply scaling
        transByPulse = (transByPulse / gain) + offset
        # create masked array
        trans_masked = numpy.ma.array(transByPulse, mask=trans_idx_mask)
        
        self.lastTransSpace = trans_space
        self.lastTrans_Idx = trans_idx
        self.lastTrans_IdxMask = trans_idx_mask

        return trans_masked
        
    def readReceived(self):
        """
        Return the 3d masked integer array of received for each of the
        current pulses.
        SPDV3 only has 1 transmitted per pulse so the second axis is empty.
        First axis is waveform bin and last is pulse.
        """
        # TODO: cache?
        if self.controls.spatialProcessing:
            pulses = self.readPulsesForExtent()
        else:
            pulses = self.readPulsesForRange()
            
        idx = pulses['RECEIVED_START_IDX']
        cnt = pulses['NUMBER_OF_WAVEFORM_RECEIVED_BINS']
        gain = pulses['RECEIVE_WAVE_GAIN']
        offset = pulses['RECEIVE_WAVE_OFFSET']
        
        nOut = self.fileHandle['DATA']['RECEIVED'].shape[0]
        recv_space, recv_idx, recv_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                    idx, cnt, nOut)
        
        received = recv_space.read(self.fileHandle['DATA']['RECEIVED'])

        # add a dummy axis
        recv_idx = numpy.expand_dims(recv_idx, 1)
        recv_idx_mask = numpy.expand_dims(recv_idx_mask, 1)
        
        # reshape
        recvByPulse = received[recv_idx]
        # apply scaling
        recvByPulse = (recvByPulse / gain) + offset
        # create masked array
        recv_masked = numpy.ma.array(recvByPulse, mask=recv_idx_mask)
        
        self.lastRecvSpace = recv_space
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
                gridindexutils.flatten3dMaskedArray(flatPulses, pulses,
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
                        (pulses[self.si_xPulseColName] < self.extent.xMax) & 
                        (pulses[self.si_yPulseColName] > self.extent.yMin) &
                        (pulses[self.si_yPulseColName] <= self.extent.yMax))
            pulses = pulses[mask]
            self.lastPulsesSpace.updateBoolArray(mask)
            
        return pulses

    def preparePointsForWriting(self, points, pulses):
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
                gridindexutils.flatten3dMaskedArray(flatPoints, points, 
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
                gridindexutils.flatten2dMaskedArray(flatPoints, points, 
                            self.lastPoints_IdxMask, self.lastPoints_Idx)
                points = flatPoints
            else:
                # TODO: flatten somehow
                raise NotImplementedError()
            
        if points.ndim != 1:
            msg = 'Point array must be either 1d, 2 or 3d'
            raise generic.LiDARInvalidData(msg)

        if self.mode == generic.UPDATE:
        
            # put back in the order we read so fields
            # line up and spatial index still works
            # the points have been re-ordered by binning
            if self.controls.spatialProcessing and origPointsDims == 3:
                sortedPointsundo = numpy.empty_like(points)
                gridindexutils.unsortArray(points, 
                        self.lastPoints3d_InRegionSort, sortedPointsundo)
                points = sortedPointsundo

                x_idx = numpy.repeat(self.lastPulses[self.si_xPulseColName],
                    self.lastPulses['NUMBER_OF_RETURNS'])
                y_idx = numpy.repeat(self.lastPulses[self.si_yPulseColName],
                    self.lastPulses['NUMBER_OF_RETURNS'])

                mask = ( (x_idx >= self.extent.xMin) & 
                    (x_idx < self.extent.xMax) &
                    (y_idx > self.extent.yMin) &
                    (y_idx <= self.extent.yMax))

                # strip out the points connected with pulses that were 
                # originally outside
                # the window and within the overlap.
                mask = mask[self.lastPoints3d_InRegionMask]
                self.lastPointsSpace.updateBoolArray(self.lastPoints3d_InRegionMask)

                points = points[mask]
                self.lastPointsSpace.updateBoolArray(mask)

            if points.dtype != POINT_DTYPE:
                # we need these for 
                # 1) inserting missing fields when they have read a subset of them
                if self.controls.spatialProcessing:
                    origPoints = self.readPointsForExtent()
                else:
                    origPoints = self.readPointsForRange()

                if self.controls.spatialProcessing:
                    if origPointsDims == 3:
                        # just the ones that are within the region
                        # this makes the length of origPoints the same as 
                        # that returned by pointsbybins flattened
                        origPoints = origPoints[self.lastPoints3d_InRegionMask]                           
                        origPoints = origPoints[mask]

                # passed in array does not have all the fields we need to write
                # so get the original data read 
                for fieldName in points.dtype.fields.keys():
                    origPoints[fieldName] = points[fieldName]

                #change them over so we have the full data
                points = origPoints

        else:
            # need to check that passed in data has all the required fields
            if points.dtype != POINT_DTYPE:
                msg = 'Point array does not have all the required fields'
                raise generic.LiDARInvalidData(msg)
                
            # strip points outside extent in this case
            if self.controls.spatialProcessing:
                x_idx = numpy.repeat(pulses[self.si_xPulseColName],
                        pulses['NUMBER_OF_RETURNS'])
                y_idx = numpy.repeat(pulses[self.si_yPulseColName],
                        pulses['NUMBER_OF_RETURNS'])
                
                mask = ( (x_idx >= self.extent.xMin) & 
                        (x_idx < self.extent.xMax) &
                        (y_idx > self.extent.yMin) &
                        (y_idx <= self.extent.yMax))
                        
                points = points[mask]

        # translate any classifications
        self.recodeClassification(points, generic.RECODE_TO_DRIVER)
        
        return points

    def prepareTransmittedForWriting(self, transmitted, waveformInfo):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        if transmitted.size == 0:
            return None

        if transmitted.ndim != 3:
            msg = 'transmitted data must be 3d'
            raise generic.LiDARInvalidData(msg)

        if self.mode == generic.UPDATE:

            origShape = transmitted.shape

            # un scale back to DN
            offset = waveformInfo[0]['TRANS_WAVE_OFFSET']
            gain = waveformInfo[0]['TRANS_WAVE_GAIN']
            transmitted = (transmitted - offset) * gain

            # flatten it back to 1d so it can be written
            flatSize = self.lastTrans_Idx.max() + 1
            flatTrans = numpy.empty((flatSize,), dtype=transmitted.data.dtype)
            gridindexutils.flatten3dMaskedArray(flatTrans, transmitted,
                self.lastTrans_IdxMask, self.lastTrans_Idx)
            transmitted = flatTrans
                
            # mask out those in the overlap using the pulses
            if self.controls.spatialProcessing:

                origPulses = self.readPulsesForExtent()
                mask = ( (origPulses[self.si_xPulseColName] >= self.extent.xMin) & 
                        (origPulses[self.si_xPulseColName] < self.extent.xMax) & 
                        (origPulses[self.si_yPulseColName] > self.extent.yMin) &
                        (origPulses[self.si_yPulseColName] <= self.extent.yMax) )
                
                # Repeat the mask so that it is the same shape as the 
                # original transmitted and then flatten in the same way
                # we can then remove the transmitted outside the extent.         
                # We can't do this earlier since removing from transmitted
                # would mean the above flattening trick won't work.
                mask = numpy.expand_dims(mask, axis=0)
                mask = numpy.expand_dims(mask, axis=0)
                mask = numpy.repeat(mask, origShape[1], axis=2)
                flatMask = numpy.empty((flatSize,), dtype=mask.dtype)
                gridindexutils.flatten3dMaskedArray(flatMask, mask, 
                    self.lastTrans_IdxMask, self.lastTrans_Idx)
            
                transmitted = transmitted[flatMask]
                self.lastTransSpace.updateBoolArray(flatMask)
                
        else:
            raise NotImplementedError()
                
        return transmitted

    def prepareReceivedForWriting(self, received, waveformInfo):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        if received.size == 0:
            return None

        if received.ndim != 3:
            msg = 'received data must be 3d'
            raise generic.LiDARInvalidData(msg)

        if self.mode == generic.UPDATE:

            origShape = received.shape
            
            # un scale back to DN
            offset = waveformInfo[0]['RECEIVE_WAVE_OFFSET']
            gain = waveformInfo[0]['RECEIVE_WAVE_GAIN']
            received = (received - offset) * gain

            # flatten it back to 1d so it can be written
            flatSize = self.lastRecv_Idx.max() + 1
            flatRecv = numpy.empty((flatSize,), dtype=received.data.dtype)
            gridindexutils.flatten3dMaskedArray(flatRecv, received,
                self.lastRecv_IdxMask, self.lastRecv_Idx)
            received = flatRecv
                
            # mask out those in the overlap using the pulses
            if self.controls.spatialProcessing:

                origPulses = self.readPulsesForExtent()
                mask = ( (origPulses[self.si_xPulseColName] >= self.extent.xMin) & 
                        (origPulses[self.si_xPulseColName] < self.extent.xMax) & 
                        (origPulses[self.si_yPulseColName] > self.extent.yMin) &
                        (origPulses[self.si_yPulseColName] <= self.extent.yMax))

                # Repeat the mask so that it is the same shape as the 
                # original received and then flatten in the same way
                # we can then remove the received outside the extent.         
                # We can't do this earlier since removing from received
                # would mean the above flattening trick won't work.
                mask = numpy.expand_dims(mask, axis=0)
                mask = numpy.expand_dims(mask, axis=0)
                mask = numpy.repeat(mask, origShape[1], axis=2)
                flatMask = numpy.empty((flatSize,), dtype=mask.dtype)
                gridindexutils.flatten3dMaskedArray(flatMask, mask, 
                    self.lastRecv_IdxMask, self.lastRecv_Idx)
            
                received = received[flatMask]
                self.lastRecvSpace.updateBoolArray(flatMask)

        else:
            raise NotImplementedError()

        return received
    
    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        Write all the updated data. Pass None for data that do not need to be updated.
        It is assumed that each parameter has been read by the reading functions
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return

        elif self.mode == generic.CREATE:
            # we only accept new data in a particular form so we can attach
            # points to pulses
            if pulses is None and points is None:
                msg = 'Must provide points and pulses when writing new data'
                raise generic.LiDARInvalidData(msg)
                
            if pulses.ndim != 1:
                msg = 'pulses must be 1d as returned from getPulses'
                raise generic.LiDARInvalidData(msg)
            if points.ndim != 2:
                msg = 'points must be 2d as returned from getPointsByPulse'
                raise generic.LiDARInvalidData(msg)
            
        writeWaveformInfo = waveformInfo is not None
        if writeWaveformInfo and pulses is None:
            # the waveform info actually lives in the pulses
            # waveformInfo actually has the same field names 
            # as pulses, so we can just pretend they are the 
            # same the self.preparePulsesForWriting will add the 
            # other fields
            pulses = writeWaveformInfo
            # else case is handled below when we know
            # all the pulses fields exist
            
        # so we can unscale the transmitted and received
        if waveformInfo is None and transmitted is not None or received is not None:
            waveformInfo = self.readWaveformInfo()
            
        if pulses is not None:
            pulses = self.preparePulsesForWriting(pulses)
            
            if writeWaveformInfo:
                # since waveformInfo takes precedence copy 
                # data into now that preparePulsesForWriting has
                # ensured all fields exist
                for name in waveformInfo.dtype.fields.keys():
                    pulses[name] = waveformInfo[name]
            
        if points is not None:
            points = self.preparePointsForWriting(points, pulses)
            
        if transmitted is not None:
            transmitted = self.prepareTransmittedForWriting(transmitted, waveformInfo)
            
        if received is not None:
            received = self.prepareReceivedForWriting(received, waveformInfo)
            
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
                ds = self.fileHandle['DATA']['POINTS']
                self.lastPointsSpace.write(ds, points)
            if pulses is not None:
                ds = self.fileHandle['DATA']['PULSES']               
                self.lastPulsesSpace.write(ds, pulses)                
            if transmitted is not None:
                ds = self.fileHandle['DATA']['TRANSMITTED']
                self.lastTransSpace.write(ds, transmitted)
            if received is not None:
                ds = self.fileHandle['DATA']['RECEIVED']
                self.lastRecvSpace.write(ds, received)

        # we don't update the spatial index
        # TODO: maybe we should?

        # update the header with any info that has changed.
        self.updateHeaderFromData(points, pulses)
        
    def updateHeaderFromData(self, points, pulses):
        """
        Given some data, updates the _MIN, _MAX etc
        """
        if points is not None and points.size > 0:
            for key in POINTS_HEADER_UPDATE_DICT.keys():
                if key in points.dtype.names:
                    minVal = points[key].min()
                    maxVal = points[key].max()
                    minKey, maxKey = POINTS_HEADER_UPDATE_DICT[key]
                    if minVal < self.headerDict[minKey]:
                        self.headerDict[minKey] = minVal
                        self.headerUpdated = True
                    if maxVal > self.headerDict[maxKey]:
                        self.headerDict[maxKey] = maxVal
                        self.headerUpdated = True

        if pulses is not None and pulses.size > 0:
            for key in PULSES_HEADER_UPDATE_DICT.keys():
                if key in pulses.dtype.names:
                    minVal = pulses[key].min()
                    maxVal = pulses[key].max()
                    minKey, maxKey = PULSES_HEADER_UPDATE_DICT[key]
                    if minVal < self.headerDict[minKey]:
                        self.headerDict[minKey] = minVal
                        self.headerUpdated = True
                    if maxVal > self.headerDict[maxKey]:
                        self.headerDict[maxKey] = maxVal
                        self.headerUpdated = True

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
        if self.pulseRange.startPulse >= nTotalPulses:
            # no data to read
            self.pulseRange.startPulse = 0
            self.pulseRange.endPulse = 0
            bMore = False
            
        elif self.pulseRange.endPulse >= nTotalPulses:
            self.pulseRange.endPulse = nTotalPulses
            
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
        
        nOut = self.fileHandle['DATA']['POINTS'].shape[0]
        
        point_space, point_idx, point_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                    startIdxs, nReturns, nOut)
                
        points = point_space.read(self.fileHandle['DATA']['POINTS'])

        # translate any classifications
        self.recodeClassification(points, generic.RECODE_TO_LAS, colNames)
        
        # keep these indices from pulses to points - handy for the indexing 
        # functions.
        self.lastPoints = points
        self.lastPointsSpace = point_space
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
    
        size = self.fileHandle['DATA']['PULSES'].shape[0]
        space = h5space.createSpaceFromRange(self.pulseRange.startPulse,
                    self.pulseRange.endPulse, size)
        pulses = space.read(self.fileHandle['DATA']['PULSES'])
                  
        self.lastPulses = pulses
        self.lastPulsesSpace = space
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
        if self.mode == generic.READ:
            msg = 'Can only set header values on read or create'
            raise generic.LiDARInvalidSetting(msg)
            
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
        if self.mode == generic.READ:
            msg = 'Can only set header values on read or create'
            raise generic.LiDARInvalidSetting(msg)
            
        if name not in self.headerDict:
            msg = 'Header field %s not supported in SPDV3 files' % name
            raise ValueError(msg)
        self.headerDict[name] =  value
        self.headerUpdated = True 
        
    def getNativeDataType(self, colName, arrayType):
        """
        Return the native dtype (numpy.int16 etc)that a column is stored
        as internally. Provided so scaling
        can be adjusted when translating between formats.
        
        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        """
        if arrayType == generic.ARRAY_TYPE_PULSES:
            if colName in PULSE_DTYPE.fields:
                return PULSE_DTYPE.fields[colName][0]
            else:
                raise generic.LiDARArrayColumnError('column not found')
        elif arrayType == generic.ARRAY_TYPE_POINTS:
            if colName in POINT_DTYPE.fields:
                return POINT_DTYPE.fields[colName][0]
            else:
                raise generic.LiDARArrayColumnError('column not found')
        else:
            # we don't really have waveform info dtype since we 
            # don't write it to file
            raise generic.LiDARInvalidSetting('Unsupported array type')
        
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
        except (OSError, IOError) as err:
            # always seems to through an OSError
            raise generic.LiDARFormatNotUnderstood(str(err))
            
        # check that it is indeed the right version
        if 'HEADER' not in fileHandle:
            msg = 'File does not appear to be SPD V3'
            raise generic.LiDARFormatNotUnderstood(msg)
            
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
        self.hasSpatialIndex = 'INDEX' in fileHandle
        # probably other things too
        
    @staticmethod
    def getDriverName():
        """
        Name of this driver
        """
        return "SPDV3"

    @staticmethod
    def getHeaderTranslationDict():
        """
        Return dictionary with non-standard header names
        """
        return HEADER_TRANSLATION_DICT
