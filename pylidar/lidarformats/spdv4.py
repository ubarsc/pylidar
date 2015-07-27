
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

# types of indexing in the file
SPDV4_INDEX_CARTESIAN = 1
SPDV4_INDEX_SPHERICAL = 2
SPDV4_INDEX_CYLINDRICAL = 3
SPDV4_INDEX_POLAR = 4
SPDV4_INDEX_SCAN = 5

# types of spatial indices
SPDV4_INDEXTYPE_SIMPLEGRID = 0

class SPDV4SpatialIndex(object):
    """
    Class that hides the details of different Spatial Indices
    that can be contained in the SPDV4 file.
    """
    def __init__(self, filehandle, mode):
        self.fileHandle = filehandle
        self.mode = mode
        
    def getPulsesBoolForExtent(self, extent):
        raise NotImplementedError()

    def getPointsBoolForExtent(self, extent):
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
    def __init__(self, filehandle, mode):
        SPDV4SpatialIndex.__init__(self, fileHandle, mode)
        self.si_group = None
        
        if mode == generic.READ:
            group = fileHandle[SPATIALINDEX_GROUP]
            if group is not None:
                group = group[SIMPLEPULSEGRID_GROUP]
        
            if group is None:
                raise generic.LiDARSpatialIndexNotAvailable()
            else:
                self.si_group = group
                
        else:
            # be more forgiving - create it if it does not exist
            group = fileHandle[SPATIALINDEX_GROUP]
            if group is None:
                group = fileHandle.create_group(SPATIALINDEX_GROUP)
                
            group = group[SIMPLEPULSEGRID_GROUP]
            if group is None:
                group = group.create_group(SIMPLEPULSEGRID_GROUP)
                
            self.si_group = group

    def getPulsesForExtent(self, extent):
        raise NotImplementedError()

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
            h5py_mode == 'w'
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

        # Spatial Index
        # TODO: prefType
        self.si_handler = SPDV4SpatialIndex.getHandlerForFile(self.fileHandle, mode)
        self.si_binSize = fileAttrs['BIN_SIZE'][0]
        self.si_Shape = fileAttrs['NUMBER_BINS_X'][0], fileAttrs['NUMBER_BINS_Y'][0]
        self.si_xMin = fileAttrs['INDEX_TLX'][0]
        self.si_yMax = fileAttrs['INDEX_TLY'][0]
        self.si_xMax = self.si_xMin + (self.si_Shape[0] * self.si_binSize)
        self.si_yMin = self.si_yMax - (self.si_Shape[1] * self.si_binSize)
        self.wkt = fileAttrs['SPATIAL_REFERENCE'][0]
        if sys.version_info[0] == 3:
            self.wkt = self.wkt.decode()
         
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

        self.pixGrid = None
         
    def getDriverName(self):
        """
        Name of this driver
        """
        return "SPDV4"

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
        if self.mode == generic.READ:
            msg = 'Can only set new pixel grid when updating or creating'
            raise generic.LiDARInvalidData(msg)
        self.si_binSize = pixGrid.xRes
        self.si_xMin = pixGrid.xMin
        self.si_yMax = pixGrid.yMax
        self.si_xMax = pixGrid.xMax
        self.si_yMin = pixGrid.yMin
        self.wkt = pixGrid.projection
        
        # TODO: set this in si_handler etc
        # TODO: write info back to file
            
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
        # TODO: spatial index etc
        
        # close
        self.fileHandle.close()
        self.fileHandle = None        
        self.lastExtent = None
        self.lastPoints = None
        self.lastPointsColumns = None
        self.lastPulses = None
        self.lastPulsesColumns = None

    def readPointsForExtent(self, colNames=None):
        """
        Read all the points within the given extent
        as 1d structured array. 

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == self.extent and 
            not self.lastPoints is None and self.lastPointsColumns == colNames):
            return self.lastPoints
        
        point_bool = self.si_handler.getPointsBoolForExtent(self.extent)
        
        if colNames is None:
            # get all names
            colNames = self.fileHandle['DATA']['POINTS'].keys()
        
        # create a blank structured array to read the data into
        dtypeList = []
        for name in colNames:
            dtype = self.fileHandle['DATA']['POINTS'][name].dtype
            dtypeList.append(dtype)
            
        points = None
        
        for name in colNames:
            data = self.fileHandle['DATA']['POINTS'][name][point_bool]
            dataList.append(data)
            
        points = numpy.array(dataList)

        