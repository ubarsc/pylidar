
"""
SPD V4 format driver and support functions
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

HEADER_FIELDS = {'AZIMUTH_MAX' : numpy.float32, 'AZIMUTH_MIN' : numpy.float32,
'BANDWIDTHS' : numpy.float32, 'BIN_SIZE' : numpy.float32,
'BLOCK_SIZE_POINT' : numpy.uint16, 'BLOCK_SIZE_PULSE' : numpy.uint16,
'BLOCK_SIZE_WAVEFORM' : numpy.uint16,
'BLOCK_SIZE_RECEIVED' : numpy.uint16, 'BLOCK_SIZE_TRANSMITTED' : numpy.uint16,
'CAPTURE_DATETIME' : bytes, 'CREATION_DATETIME' : bytes,
'DEFINED_HEIGHT' : numpy.uint8, 'DEFINED_ORIGIN' : numpy.uint8,
'DEFINED_RECEIVE_WAVEFORM' : numpy.uint8, 'DEFINED_RGB' : numpy.uint8,
'DEFINED_TRANS_WAVEFORM' : numpy.uint8, 'FIELD_OF_VIEW' : numpy.float32,
'FILE_SIGNATURE' : bytes, 'FILE_TYPE' : numpy.uint16,
'GENERATING_SOFTWARE' : bytes, 'INDEX_TYPE' : numpy.uint16,
'INDEX_TLX' : numpy.float64, 'INDEX_TLY' : numpy.float64,
'NUMBER_BINS_X' : numpy.uint32, 'NUMBER_BINS_Y' : numpy.uint32,
'NUMBER_OF_POINTS' : numpy.uint64, 'NUMBER_OF_PULSES' : numpy.uint64,
'NUMBER_OF_WAVEFORMS' : numpy.uint16,
'NUM_OF_WAVELENGTHS' : numpy.uint16, 'POINT_DENSITY' : numpy.float32,
'PULSE_ALONG_TRACK_SPACING' : numpy.float32, 
'PULSE_ANGULAR_SPACING_SCANLINE' : numpy.float32,
'PULSE_ANGULAR_SPACING_SCANLINE_IDX' : numpy.float32,
'PULSE_CROSS_TRACK_SPACING' : numpy.float32, 'PULSE_DENSITY' : numpy.float32,
'PULSE_ENERGY' : numpy.float32, 'PULSE_FOOTPRINT' : numpy.float32,
'PULSE_INDEX_METHOD' : numpy.uint16, 'RANGE_MAX' : numpy.float32,
'RANGE_MIN' : numpy.float32, 
'SCANLINE_IDX_MAX' : numpy.uint32, 'SCANLINE_IDX_MIN' : numpy.uint32,
'SCANLINE_MAX' : numpy.uint16, 'SCANLINE_MIN' : numpy.uint16,
'SENSOR_APERTURE_SIZE' : numpy.float32, 'SENSOR_BEAM_DIVERGENCE' : numpy.float32,
'SENSOR_HEIGHT' : numpy.float32, 'SENSOR_MAX_SCAN_ANGLE' : numpy.float32,
'SENSOR_PULSE_REPETITION_FREQ' : numpy.float32,
'SENSOR_SCAN_RATE' : numpy.float32, 'SENSOR_SPEED' : numpy.float32,
'SENSOR_TEMPORAL_BIN_SPACING' : numpy.float64, 
'SENSOR_BEAM_EXIT_DIAMETER' : numpy.float32,
'SPATIAL_REFERENCE' : bytes,
'SYSTEM_IDENTIFIER' : bytes, 'USER_META_DATA' : bytes,
'VERSION_SPD' : numpy.uint8, 'VERSION_DATA' : numpy.uint8, 
'WAVEFORM_BIT_RES' : numpy.uint16, 'WAVELENGTHS' : numpy.float32,
'X_MAX' : numpy.float32, 'X_MIN' : numpy.float32, 'Y_MAX' : numpy.float32,
'Y_MIN' : numpy.float32, 'Z_MAX' : numpy.float64, 'Z_MIN' : numpy.float64,
'HEIGHT_MIN' : numpy.float32, 'HEIGHT_MAX' : numpy.float32, 
'ZENITH_MAX' : numpy.float32, 'ZENITH_MIN' : numpy.float64,
'NUMBER_OF_PULSE_ATTRIBUTES' : numpy.uint16,
'NUMBER_OF_POINT_ATTRIBUTES' : numpy.uint16,
'PULSE_ATTRIBUTES' : bytes, 'POINT_ATTRIBUTES' : bytes,
'RGB_FIELD' : bytes, 'WAVEFORM_GAIN' : numpy.float32, 
'WAVEFORM_OFFSET' : numpy.float32, 'XYZHR_OFFSET' : numpy.float32,
'XYZHR_SCALE' : numpy.float32, 'RAD_SCALE' : numpy.float32, 
'RAD_OFFSET' : numpy.float32 }

HEADER_ARRAY_FIELDS = ('BANDWIDTHS', 'WAVELENGTHS', 'VERSION_SPD', 
'PULSE_ATTRIBUTES', 'POINT_ATTRIBUTES', 'RGB_FIELD', 'WAVEFORM_GAIN',
'WAVEFORM_OFFSET', 'XYZHR_OFFSET', 'XYZHR_SCALE')

# Note: PULSE_ID, NUMBER_OF_RETURNS and PTS_START_IDX always created by pylidar
PULSES_ESSENTIAL_FIELDS = ('X_IDX', 'Y_IDX')
# RETURN_NUMBER always created by pylidar
POINTS_ESSENTIAL_FIELDS = ('X', 'Y', 'Z', 'CLASSIFICATION')

# types of indexing in the file
SPDV4_INDEX_CARTESIAN = 1
SPDV4_INDEX_SPHERICAL = 2
SPDV4_INDEX_CYLINDRICAL = 3
SPDV4_INDEX_POLAR = 4
SPDV4_INDEX_SCAN = 5

# types of spatial indices
SPDV4_INDEXTYPE_SIMPLEGRID = 0
# types for the spatial index
SPDV4_SIMPLEGRID_COUNT_DTYPE = numpy.uint32
SPDV4_SIMPLEGRID_INDEX_DTYPE = numpy.uint64

# flags for POINT_FLAGS
SPDV4_POINT_FLAGS_IGNORE = 1
SPDV4_POINT_FLAGS_OVERLAP = 2
SPDV4_POINT_FLAGS_SYNTHETIC = 4
SPDV4_POINT_FLAGS_KEY_POINT = 8
SPDV4_POINT_FLAGS_WAVEFORM = 16

class SPDV4SpatialIndex(object):
    """
    Class that hides the details of different Spatial Indices
    that can be contained in the SPDV4 file.
    """
    def __init__(self, fileHandle, mode):
        self.fileHandle = fileHandle
        self.mode = mode
        
        # read the pixelgrid info out of the header
        # this is same for all spatial indices on SPD V4
        fileAttrs = fileHandle.attrs
        binSize = fileAttrs['BIN_SIZE']
        shape = fileAttrs['NUMBER_BINS_Y'], fileAttrs['NUMBER_BINS_X']
        xMin = fileAttrs['INDEX_TLX']
        yMax = fileAttrs['INDEX_TLY']
        xMax = xMin + (shape[1] * binSize)
        yMin = yMax - (shape[0] * binSize)
        wkt = fileAttrs['SPATIAL_REFERENCE']
        if sys.version_info[0] == 3:
            wkt = wkt.decode()
            
        self.pixelGrid = pixelgrid.PixelGridDefn(projection=wkt, xMin=xMin,
                xMax=xMax, yMin=yMin, yMax=yMax, xRes=binSize, yRes=binSize)
                
    def close(self):
        """
        Call to write data, close files etc
        """
        # update the header
        if self.mode == generic.CREATE:
            fileAttrs = self.fileHandle.attrs
            fileAttrs['BIN_SIZE'] = self.pixelGrid.xRes
            nrows, ncols = self.pixelGrid.getDimensions()
            fileAttrs['NUMBER_BINS_Y'] = nrows
            fileAttrs['NUMBER_BINS_X'] = ncols
            fileAttrs['INDEX_TLX'] = self.pixelGrid.xMin
            fileAttrs['INDEX_TLY'] = self.pixelGrid.yMax
            fileAttrs['SPATIAL_REFERENCE'] = self.pixelGrid.projection
        
        self.fileHandle = None
        
    def getPulsesBoolForExtent(self, extent):
        raise NotImplementedError()

    def getPointsBoolForExtent(self, extent, pulses):
        raise NotImplementedError()
        
    def createNewIndex(self, pixelGrid):
        raise NotImplementedError()
        
    def setPointsAndPulsesForExtent(self, extent, points, pulses, lastPulseID):
        raise NotImplementedError()
        
    @staticmethod
    def getHandlerForFile(fileHandle, mode, prefType=SPDV4_INDEXTYPE_SIMPLEGRID):
        """
        Returns the 'most appropriate' spatial
        index handler for the given file.
        
        prefType contains the user's preferred spatial index.
        If it is not defined for a file another type will be chosen
        
        None is returned if no spatial index is available and mode == READ
        
        If mode != READ and the prefType spatial index type not available 
        it will be created.
        """
        handler = None
        if prefType == SPDV4_INDEXTYPE_SIMPLEGRID:
            cls = SPDV4SimpleGridSpatialIndex
            
        # TODO: other types here
        
        # now try and create the instance
        try:
            handler = cls(fileHandle, mode)
        except generic.LiDARSpatialIndexNotAvailable:
            # TODO: code to try other indices if READ
            pass
            
        return handler
        
SPATIALINDEX_GROUP = 'SPATIALINDEX'
SIMPLEPULSEGRID_GROUP = 'SIMPLEPULSEGRID'
        
class SPDV4SimpleGridSpatialIndex(SPDV4SpatialIndex):
    def __init__(self, fileHandle, mode):
        SPDV4SpatialIndex.__init__(self, fileHandle, mode)
        self.si_cnt = None
        self.si_idx = None
        
        if mode == generic.READ or mode == generic.UPDATE:
            group = fileHandle[SPATIALINDEX_GROUP]
            if group is not None:
                group = group[SIMPLEPULSEGRID_GROUP]
        
            if group is None:
                raise generic.LiDARSpatialIndexNotAvailable()
            else:
                self.si_cnt = group['PLS_PER_BIN'][...]
                self.si_idx = group['BIN_OFFSETS'][...]
                
    def close(self):
        if self.mode == generic.CREATE:
            # create it if it does not exist
            if SPATIALINDEX_GROUP not in self.fileHandle:
                group = self.fileHandle.create_group(SPATIALINDEX_GROUP)
            else:
                group = self.fileHandle[SPATIALINDEX_GROUP]
                
            if SIMPLEPULSEGRID_GROUP not in group:
                group = group.create_group(SIMPLEPULSEGRID_GROUP)
            else:
                group = group[SIMPLEPULSEGRID_GROUP]
                
            nrows, ncols = self.pixelGrid.getDimensions()
            # params adapted from SPDLib
            countDataset = group.create_dataset('PLS_PER_BIN', 
                    (nrows, ncols), 
                    chunks=(1, ncols), dtype=SPDV4_SIMPLEGRID_COUNT_DTYPE,
                    shuffle=True, compression="gzip", compression_opts=1)
            if self.si_cnt is not None:
                countDataset[...] = self.si_cnt
                    
            offsetDataset = group.create_dataset('BIN_OFFSETS', 
                    (nrows, ncols), 
                    chunks=(1, ncols), dtype=SPDV4_SIMPLEGRID_INDEX_DTYPE,
                    shuffle=True, compression="gzip", compression_opts=1)
            if self.si_idx is not None:
                offsetDataset[...] = self.si_idx
                    
        SPDV4SpatialIndex.close(self)

    def getPulsesBoolForExtent(self, extent):
        raise NotImplementedError()

    def getPointsBoolForExtent(self, extent, pulses):
        raise NotImplementedError()

    def createNewIndex(self, pixelGrid):
        nrows, ncols = pixelGrid.getDimensions()
        self.si_cnt = numpy.zeros((nrows, ncols), 
                        dtype=SPDV4_SIMPLEGRID_COUNT_DTYPE)
        self.si_idx = numpy.zeros((nrows, ncols), 
                        dtype=SPDV4_SIMPLEGRID_INDEX_DTYPE)

        # save the pixelGrid
        self.pixelGrid = pixelGrid

    def setPointsAndPulsesForExtent(self, extent, points, pulses, lastPulseID):
        xMin = self.pixelGrid.snapToGrid(extent.xMin, self.pixelGrid.xMin, 
                self.pixelGrid.xRes)
        xMax = self.pixelGrid.snapToGrid(extent.xMax, self.pixelGrid.xMax, 
                self.pixelGrid.xRes)
        yMin = self.pixelGrid.snapToGrid(extent.yMin, self.pixelGrid.yMin, 
                self.pixelGrid.yRes)
        yMax = self.pixelGrid.snapToGrid(extent.yMax, self.pixelGrid.yMax, 
                self.pixelGrid.yRes)
                
        # size of spatial index we need to write
        nrows = int(numpy.ceil((yMax - yMin) / self.pixelGrid.xRes))
        ncols = int(numpy.ceil((xMax - xMin) / self.pixelGrid.xRes))
                
        mask, sortedBins, idx_subset, cnt_subset = gridindexutils.CreateSpatialIndex(
                pulses['Y_IDX'], pulses['X_IDX'], self.pixelGrid.xRes, yMax, xMin,
                nrows, ncols, SPDV4_SIMPLEGRID_INDEX_DTYPE, 
                SPDV4_SIMPLEGRID_COUNT_DTYPE)
                   
        # note overlap is zero as we have already removed them by not including 
        # them in the spatial index
        imageSlice, siSlice = gridindexutils.getSlicesForExtent(self.pixelGrid, 
                     self.si_cnt.shape, 0, xMin, xMax, yMin, yMax)

        if imageSlice is not None and siSlice is not None:

            self.si_cnt[siSlice] = cnt_subset[imageSlice]
            self.si_idx[siSlice] = idx_subset[imageSlice]

        # re-sort the pulses to match the new spatial index
        pulses = pulses[mask]
        pulses = pulses[sortedBins]
        
        # return the new ones in the correct order to write
        return points, pulses
                                    
class SPDV4File(generic.LiDARFile):
    """
    Class to support reading and writing of SPD Version 4.x files.
    
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
            # always seems to throw an OSError
            raise generic.LiDARFormatNotUnderstood(str(err))
            
        # check that it is indeed the right version
        # and get attributes
        fileAttrs = self.fileHandle.attrs
        if mode == generic.READ or mode == generic.UPDATE:
            if not 'VERSION_SPD' in fileAttrs:
                msg = "File appears not to be SPD"
                raise generic.LiDARFormatNotUnderstood(msg)
            elif fileAttrs['VERSION_SPD'][0] != 3:
                msg = "File seems to be wrong version for this driver"
                raise generic.LiDARFormatNotUnderstood(msg)
                
        else:
            # make sure the attributes have the right names
            for key in HEADER_FIELDS:
                cls = HEADER_FIELDS[key]
                # blank value - 0 for numbers, '' for strings
                if key in HEADER_ARRAY_FIELDS:
                    fileAttrs[key] = numpy.array([cls()])
                else:
                    fileAttrs[key] = cls() 
                    
            # create the POINTS and PULSES groups
            data = self.fileHandle.create_group('DATA')
            data.create_group('POINTS')
            data.create_group('PULSES')

        # Spatial Index
        # TODO: prefType
        self.si_handler = SPDV4SpatialIndex.getHandlerForFile(self.fileHandle, mode)
         
        # the following is for caching reads so we don't need to 
        # keep re-reading each time the user asks. Also handy since
        # reading points requires pulses etc
        self.lastExtent = None
        self.lastPulseRange = None
        self.lastPoints = None
        self.lastPointsBool = None
        self.lastPulses = None
        self.lastPulsesBool = None
        
        # the current extent or range for data being read
        self.extent = None
        self.pulseRange = None

        self.pixGrid = None
        
        # for writing a new file, we generate PULSE_ID uniquely
        self.lastPulseID = numpy.uint64(0)
        
    @staticmethod 
    def getDriverName():
        """
        Name of this driver
        """
        return "SPDV4"

    @staticmethod
    def getTranslationDict(arrayType):
        """
        Translation dictionary between formats
        """
        dict = {}
        if arrayType == generic.ARRAY_TYPE_POINTS:
            dict[generic.FIELD_POINTS_RETURN_NUMBER] = 'RETURN_NUMBER'
        elif arrayType == generic.ARRAY_TYPE_PULSES:
            dict[generic.FIELD_PULSES_TIMESTAMP] = 'TIMESTAMP'
        return dict

    def setExtent(self, extent):
        """
        Set the extent for reading or writing
        """
        if not self.hasSpatialIndex():
            msg = 'Format has no spatial Index. Processing must be done non-spatially'
            raise generic.LiDARInvalidSetting(msg)
        self.extent = extent
        
        # TODO: check grid
        
    def getPixelGrid(self):
        """
        Return the PixelGridDefn for this file
        """
        if self.hasSpatialIndex():
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
        Set the PixelGridDefn for the reading or 
        writing
        """
        if self.mode == generic.READ or self.mode == generic.UPDATE:
            msg = 'Can only set new pixel grid when creating'
            raise generic.LiDARInvalidData(msg)
            
        else:
            self.si_handler.createNewIndex(pixGrid)
            
        # cache it
        self.pixGrid = pixGrid
        
    def hasSpatialIndex(self):
        """
        Return True if we have a spatial index.
        """
        return self.si_handler is not None
        
    def close(self):
        """
        Close all open file handles
        """
        self.si_handler.close()
        
        # close
        self.fileHandle.close()
        self.fileHandle = None        
        self.lastExtent = None
        self.lastPoints = None
        self.lastPointsBool = None
        self.lastPoints_Idx = None
        self.lastPoints_IdxMask = None
        self.lastPointsColumns = None
        self.lastPulses = None
        self.lastPulsesBool = None
        self.lastPulses_Idx = None
        self.lastPulses_IdxMask = None
        self.lastPulsesColumns = None

    def readPointsForExtent(self, colNames=None):
        """
        Read all the points within the given extent
        as 1d structured array. 

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        pointsHandle = self.fileHandle['DATA']['POINTS']
        if colNames is None:
            # get all names
            colNames = pointsHandle.keys()
            
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == self.extent and 
            not self.lastPoints is None and self.lastPointsColumns is not None
            and self.lastPointsColumns == colNames):
            return self.lastPoints
        
        point_bool, idx, mask_idx = (
                            self.si_handler.getPointsBoolForExtent(self.extent))
        
        if isinstance(colNames, str):
            points = pointsHandle[colNames][point_bool]

        else:            
            # create a blank structured array to read the data into
            dtypeList = []
            for name in colNames:
                dtype = pointsHandle[name].dtype
                dtypeList.append(dtype)
            
            points = numpy.empty(point_bool.sum(), dtypeList)
        
            for name in colNames:
                data = pointsHandle[name][point_bool]
                points[name] = data
            
        self.lastPoints = points
        self.lastPointsBool = point_bool
        self.lastPoints_Idx = idx
        self.lastPoints_IdxMask = mask_idx
        self.lastPointsColumns = colNames

    def readPulsesForExtent(self, colNames=None):
        """
        Read all the pulses within the given extent
        as 1d structured array. 

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        pulsesHandle = self.fileHandle['DATA']['PULSES']
        if colNames is None:
            # get all names
            colNames = pulsesHandle.keys()
            
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == self.extent and 
            not self.lastPulses is None and self.lastPulsesColumns is not None
            and self.lastPulsesColumns == colNames):
            return self.lastPulses
        
        pulse_bool, idx, mask_idx = (
                            self.si_handler.getPulsesBoolForExtent(self.extent))
        
        if isinstance(colNames, str):
            pulses = pulsesHandle[colNames][pulse_bool]

        else:            
            # create a blank structured array to read the data into
            dtypeList = []
            for name in colNames:
                dtype = pulsesHandle[name].dtype
                dtypeList.append(dtype)
            
            pulses = numpy.empty(pulse_bool.sum(), dtypeList)
        
            for name in colNames:
                data = pulsesHandle[name][pulses_bool]
                pulses[name] = data
            
        self.lastPulses = pulses
        self.lastPulsesBool = pulse_bool
        self.lastPulses_Idx = idx
        self.lastPulses_IdxMask = mask_idx
        self.lastPulsesColumns = colNames
    
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
        pulses = self.readPulsesForExtent(colNames)
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
        pulses = numpy.ma.array(pulsesByBins, mask=idxMask)
        return pulses

    def readPointsForExtentByBins(self, extent=None, colNames=None, 
                    indexByPulse=False, returnPulseIndex=False):
        # TODO:
        pass   
        
    def readPointsByPulse(self, colNames=None):
        """
        Return a 2d masked structured array of point that matches
        the pulses.
        """
        if self.controls.spatialProcessing:
            points = self.readPointsForExtent(colNames)
        else:
            points = self.readPointsForRange(colNames)
        idx = self.lastPoints_Idx
        idxMask = self.lastPoints_IdxMask
        
        pointsByPulse = points[idx]
        points = numpy.ma.array(pointsByPulse, mask=idxMask)
        return points
        
    def readTransmitted(self):
        """
        Return the 2d masked integer array of transmitted for each of the
        current pulses. 
        """
        # TODO:
        pass

    def readReceived(self):
        """
        Return the 2d masked integer array of received for each of the
        current pulses. 
        """
        # TODO:
        pass

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
                
        # NOTE: on update now we just don't write the X_IDX
        # and Y_IDX fields since they shouldn't be updated.
        # SPDV3 gives you a warning if you change these fields
        # in SPDV4 the updates are silently lost        

        # essential fields exist?
        if self.mode == generic.CREATE:
            for essential in PULSES_ESSENTIAL_FIELDS:
                if essential not in pulses.dtype.names:
                    msg = ('Essential field %s must exist in pulse data ' +
                             'when writing new file') % essential
                    raise generic.LiDARInvalidData(msg)
                    
            # while we are at it, grab the X_IDX and Y_IDX fields since
            # they are essential
            x_idx = pulses['X_IDX']
            y_idx = pulses['Y_IDX']
            
        else:
            # update
            # we need x_idx and y_idx for removing overlap, 
            # but it may not exist in the input, or be altered
            # so re-read it
            if self.controls.spatialProcessing:
                x_idx = self.readPulsesForExtent('X_IDX')
                y_idx = self.readPulsesForExtent('Y_IDX')

        if self.extent is not None and self.controls.spatialProcessing:
            # if we doing spatial index we need to strip out areas in the overlap
            # self.extent is the size of the block without the overlap
            # so just strip out everything outside of it
            mask = ( (x_idx >= self.extent.xMin) & 
                        (x_idx <= self.extent.xMax) & 
                        (y_idx >= self.extent.yMin) &
                        (y_idx <= self.extent.yMax))
            pulses = pulses[mask]
            if self.mode == generic.UPDATE:
                gridindexutils.updateBoolArray(self.lastPulsesBool, mask)
            
        return pulses

    def preparePointsForWriting(self, points, pulses):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        pts_start = None
        nreturns = None
        
        if points.size == 0:
            return None, pts_start, nreturns
            
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
                # flatten somehow
                
                # get the number of returns for each pulse
                # this doesn't work with structured arrays so need
                # to use one of the fields
                firstField = points.dtype.names[0]
                
                nreturns = points[firstField].count(axis=0)
                pointsHandle = self.fileHandle['DATA']['POINTS']
                currPointsCount = 0
                if firstField in pointsHandle:
                    currPointsCount = pointsHandle[firstField].shape[0]
                    
                pts_start = nreturns + currPointsCount
                # unfortunately points.compressed() doesn't work
                # for structured arrays. Use our own version instead
                outPoints = numpy.empty(points[firstField].count(), 
                                    dtype=points.dtype)
                                    
                gridindexutils.flattenMaskedStructuredArray(points.data, 
                            points[firstField].mask, outPoints)
                
                points = outPoints
                
        if points.ndim != 1:
            msg = 'Point array must be either 1d, 2 or 3d'
            raise generic.LiDARInvalidData(msg)

        if self.mode == generic.CREATE:
            # need to check that passed in data has all the required fields
            for essential in POINTS_ESSENTIAL_FIELDS:
                if essential not in points.dtype.names:
                    msg = ('Essential field %s must exist in point data ' +
                             'when writing new file') % essential
                    raise generic.LiDARInvalidData(msg)

        # strip out the points that were originally outside
        # the window and within the overlap.
        if self.controls.spatialProcessing:
            if self.mode == generic.UPDATE:
                # get data in case it is not passed in or changed
                xloc = self.readPulsesForExtent('X_IDX')
                yloc = self.readPulsesForExtent('Y_IDX')
            else:
                # on CREATE we can guarantee these exist (see above)
                xloc = pulses['X_IDX']
                yloc = pulses['Y_IDX']
                
            mask = ( (xloc >= self.extent.xMin) & 
                (xloc <= self.extent.xMax) &
                (yloc >= self.extent.yMin) &
                (yloc <= self.extent.yMax))
            points = points[mask]
            if self.mode == generic.UPDATE:
                gridindexutils.updateBoolArray(self.lastPointsBool, mask)

        
        return points, pts_start, nreturns
        
    def prepareTransmittedForWriting(self, transmitted):
        # TODO:
        return transmitted
        
    def prepareReceivedForWriting(self, received):
        # TODO:
        return received

    @staticmethod
    def createDataColumn(groupHandle, name, data):
        """
        Creates a new data column under groupHandle with the
        given name with standard HDF5 params.
        
        The type is the same as the numpy array data and data
        is written to the column
        """
        # From SPDLib
        dset = groupHandle.create_dataset(name, data.shape, 
                chunks=(250,), dtype=data.dtype, shuffle=True, 
                compression="gzip", compression_opts=1)
        dset[:] = data
        
    def writeData(self, pulses=None, points=None, transmitted=None, received=None):
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
            
        if pulses is not None:
            pulses = self.preparePulsesForWriting(pulses)
            
        if points is not None:
            points, pts_start, nreturns = self.preparePointsForWriting(points, pulses)
            
        if transmitted is not None:
            transmitted = self.prepareTransmittedForWriting(transmitted)
            
        if received is not None:
            received = self.prepareReceivedForWriting(received)
            
        if self.mode == generic.CREATE:

            if self.controls.spatialProcessing and pulses is not None:
                # write spatial index. The pulses/points may need
                # to be re-ordered before writing so we do this first
                points, pulses = self.si_handler.setPointsAndPulsesForExtent(
                        self.extent, points, pulses, self.lastPulseID)

            if pulses is not None and len(pulses) > 0:
                        
                pulsesHandle = self.fileHandle['DATA']['PULSES']
                firstField = pulses.dtype.names[0]
                if firstField in pulsesHandle:
                    oldSize = pulsesHandle[firstField].shape[0]
                else:
                    oldSize = 0
                nPulses = len(pulses)
                newSize = oldSize + nPulses
                
                # index into points and pulseid generated fields
                pulseid = numpy.arange(self.lastPulseID, 
                        self.lastPulseID + nPulses, dtype=numpy.uint64)
                self.lastPulseID = self.lastPulseID + nPulses
                generatedColumns = {'PTS_START_IDX' : pts_start,
                        'NUMBER_OF_RETURNS' : nreturns, 'PULSE_ID' : pulseid}

                for name in pulses.dtype.names:
                    # don't bother writing out the ones we generate ourselves
                    if name not in generatedColumns:
                        if name in pulsesHandle:
                            pulsesHandle[name].resize((newSize,))
                            pulsesHandle[name][oldSize:newSize+1] = pulses[name]
                        else:
                            self.createDataColumn(pulsesHandle, name, 
                                    pulses[name])
                    
                # now write the generated ones
                for name in generatedColumns.keys():
                    data = generatedColumns[name]
                    if name in pulsesHandle:
                        pulsesHandle[name].resize((newSize,))
                        pulsesHandle[name][oldSize:newSize+1] = data
                    else:
                        self.createDataColumn(pulsesHandle, name, data)
                
            if points is not None and len(points) > 0:

                pointsHandle = self.fileHandle['DATA']['POINTS']
                firstField = points.dtype.names[0]
                if firstField in pointsHandle:
                    oldSize = pointsHandle[firstField].shape[0]
                else:
                    oldSize = 0
                nPoints = len(points)
                newSize = oldSize + nPoints
                for name in points.dtype.names:
                    if name in pointsHandle:
                        pointsHandle[name].resize((newSize,))
                        pointsHandle[name][oldSize:newSize+1] = points[name]
                    else:
                        self.createDataColumn(pointsHandle, name, points[name])
                
            # TODO:
            #if transmitted is not None:
            #    oldSize = self.fileHandle['DATA']['TRANSMITTED'].shape[0]
            #    nTrans = len(transmitted)
            #    newSize = oldSize + nTrans
            #    self.fileHandle['DATA']['TRANSMITTED'].resize((newSize,))
                
            #if received is not None:
            #    oldSize = self.fileHandle['DATA']['RECEIVED'].shape[0]
            #    nRecv = len(received)
            #    newSize = oldSize + nRecv
            #    self.fileHandle['DATA']['RECEIVED'].resize((newSize,))

        else:
            if points is not None:
                self.fileHandle['DATA']['POINTS'][self.lastPointsBool] = points
            if pulses is not None:
                self.fileHandle['DATA']['PULSES'][self.lastPulsesBool] = pulses
            #if transmitted is not None:
            #    self.fileHandle['DATA']['TRANSMITTED'][self.lastTransBool] = transmitted
            #if received is not None:
            #    self.fileHandle['DATA']['RECEIVED'][self.lastRecvBool] = received
        
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
        pointsHandle = self.fileHandle['DATA']['POINTS']
        if colNames is None:
            # get all names
            colNames = pointsHandle.keys()
            
        if (self.lastPulseRange is not None and
                self.lastPulseRange == self.pulseRange and
                self.lastPoints is not None and
                colNames == self.lastPointsColumns):
            return self.lastPoints
            
        pulses = self.readPulsesForRange()
        
        nReturns = self.readPulsesForRange('NUMBER_OF_RETURNS')
        startIdxs = self.readPulsesForRange('PTS_START_IDX')
        
        # h5py prefers to take it's index by numpy bool array
        # of the same shape as the dataset
        # so we do this. If you give it the indices themselves
        # this must be done as a list which is slow
        nOut = pointsHandle['RETURN_NUMBER'].shape[0]
        point_bool, point_idx, point_idx_mask = self.convertSPDIdxToReadIdxAndMaskInfo(
                        startIdxs, nReturns, nOut)
        
        if isinstance(colNames, str):
            points = pointsHandle[colNames][point_bool]

        else:            
            # create a blank structured array to read the data into
            dtypeList = []
            for name in colNames:
                dtype = pointsHandle[name].dtype
                dtypeList.append(dtype)
            
            points = numpy.empty(point_bool.sum(), dtypeList)
        
            for name in colNames:
                data = pointsHandle[name][point_bool]
                points[name] = data
        
        # keep these indices from pulses to points - handy for the indexing 
        # functions.
        self.lastPoints = points
        self.lastPoints_Idx = point_idx
        self.lastPoints_IdxMask = point_idx_mask
        self.lastPointsColumns = colNames
        # self.lastPulseRange copied in readPulsesForRange()
        return self.subsetColumns(points, colNames)
    
    def readPulsesForRange(self, colNames=None):
        """
        Read the specified range of pulses
        """
        pulsesHandle = self.fileHandle['DATA']['PULSES']
        if colNames is None:
            # get all names
            colNames = pulsesHandle.keys()
            
        if (self.lastPulseRange is not None and
                self.lastPulseRange == self.pulseRange and 
                self.lastPulses is not None and
                self.lastPulsesColumns is not None and 
                self.lastPulsesColumns != colNames):
            return self.lastPulses

        if isinstance(colNames, str):
            pulses = pulsesHandle[colNames][
                        self.pulseRange.startPulse:self.pulseRange.endPulse]

        else:            
            # create a blank structured array to read the data into
            dtypeList = []
            for name in colNames:
                dtype = pulsesHandle[name].dtype
                dtypeList.append(dtype)
            
            nPulses = self.pulseRange.endPulse - self.pulseRange.startPulse
            pulses = numpy.empty(nPulses, dtypeList)
        
            for name in colNames:
                data = pulsesHandle[name][
                    self.pulseRange.startPulse:self.pulseRange.endPulse]
                pulses[name] = data
    
        self.lastPulses = pulses
        self.lastPulseRange = copy.copy(self.pulseRange)
        self.lastPoints = None # now invalid
        self.lastPointsColumns = colNames
        return pulses

    def getTotalNumberPulses(self):
        """
        Return the total number of pulses
        """
        return self.fileHandle['DATA']['PULSES'].shape[0]
        
    def getHeader(self):
        """
        Return our attributes on the file
        """
        return self.fileHandle.attrs
        
    def setHeader(self, newHeaderDict):
        """
        Update our cached dictionary
        """
        for key in newHeaderDict.keys():
            self.fileHandle.attrs[key] = newHeaderDict[key]
            
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.fileHandle.attrs[name]
        
    def setHeaderValue(self, name, value):
        """
        Just update one value in the header
        """
        self.fileHandle.attrs[name] = value
    
                                            