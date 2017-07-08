
"""
The spatial index handing for SPDV4
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

import os
import sys
import abc
import copy
import numpy
from rios import pixelgrid
from . import generic
from . import gridindexutils
from . import h5space

SPDV4_INDEX_CARTESIAN = 1
"types of indexing in the file"
SPDV4_INDEX_SPHERICAL = 2
"types of indexing in the file"
SPDV4_INDEX_CYLINDRICAL = 3
"types of indexing in the file"
SPDV4_INDEX_POLAR = 4
"types of indexing in the file"
SPDV4_INDEX_SCAN = 5
"types of indexing in the file"

SPDV4_INDEXTYPE_SIMPLEGRID = 0
"types of spatial indices"

SPDV4_SIMPLEGRID_COUNT_DTYPE = numpy.uint32
"data types for the spatial index"
SPDV4_SIMPLEGRID_INDEX_DTYPE = numpy.uint64
"data types for the spatial index"

class SPDV4SpatialIndex(object):
    """
    Class that hides the details of different Spatial Indices
    that can be contained in the SPDV4 file.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, fileHandle, mode):
        self.fileHandle = fileHandle
        self.mode = mode
        
        # read the pixelgrid info out of the header
        # this is same for all spatial indices on SPD V4
        fileAttrs = fileHandle.attrs
        self.binSize = fileAttrs['BIN_SIZE']
        shape = fileAttrs['NUMBER_BINS_Y'], fileAttrs['NUMBER_BINS_X']
        xMin = fileAttrs['INDEX_TLX']
        yMax = fileAttrs['INDEX_TLY']
        xMax = xMin + (shape[1] * self.binSize)
        yMin = yMax - (shape[0] * self.binSize)
        self.wkt = fileAttrs['SPATIAL_REFERENCE']
        if sys.version_info[0] == 3 and isinstance(self.wkt, bytes):
            self.wkt = self.wkt.decode()

        if shape[0] != 0 or shape[1] != 0:        
            self.pixelGrid = pixelgrid.PixelGridDefn(projection=self.wkt, xMin=xMin,
                xMax=xMax, yMin=yMin, yMax=yMax, xRes=self.binSize, 
                yRes=self.binSize)
        else:
            self.pixelGrid = None
                
    def close(self):
        """
        Call to write data, close files etc
        """
        # update the header
        if self.mode != generic.READ and self.pixelGrid is not None:
            fileAttrs = self.fileHandle.attrs
            fileAttrs['BIN_SIZE'] = self.pixelGrid.xRes
            nrows, ncols = self.pixelGrid.getDimensions()
            fileAttrs['NUMBER_BINS_Y'] = nrows
            fileAttrs['NUMBER_BINS_X'] = ncols
            fileAttrs['INDEX_TLX'] = self.pixelGrid.xMin
            fileAttrs['INDEX_TLY'] = self.pixelGrid.yMax
            if self.pixelGrid.projection is None:
                fileAttrs['SPATIAL_REFERENCE'] = ''
            else:
                fileAttrs['SPATIAL_REFERENCE'] = self.pixelGrid.projection
        
        self.fileHandle = None
        
    @abc.abstractmethod
    def getPulsesSpaceForExtent(self, extent, overlap, extentAlignedWithIndex):
        raise NotImplementedError()

    @abc.abstractmethod
    def getPointsSpaceForExtent(self, extent, overlap, extentAlignedWithIndex):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def createNewIndex(self, pixelGrid):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def setPulsesForExtent(self, extent, pulses, lastPulseID, 
                extentAlignedWithIndex):
        """
        When creating a new file, lastPulseID is the id of the last pulse
        currently written to file (which should equal the number of pulses
        in the file). 

        When updating lastPulseID will actually be a h5space.H5Space object
        that was used to read these pulses.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def canUpdateInPlace(self):
        """
        Return True if data doesn't have to be re-sorted when writing
        a spatial index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def canAccessUnaligned(self):
        """
        Return True if the requested extent does have to be aligned
        with the spatial index
        """
        raise NotImplementedError()

    @staticmethod
    def getClassForType(indexType):
        """
        Internal method - returns the cls for given index code
        """
        if indexType == SPDV4_INDEXTYPE_SIMPLEGRID:
            cls = SPDV4SimpleGridSpatialIndex
        else:
            msg = 'Unknown indextype %d' % indexType
            raise generic.LiDARInvalidSetting(msg)

        return cls
        
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

        # in order of preference
        availableIndices = [SPDV4_INDEXTYPE_SIMPLEGRID]

        cls = SPDV4SpatialIndex.getClassForType(prefType)
        availableIndices.remove(prefType)
        
        # now try and create the instance
        bContinue = True
        while bContinue:
            try:
                handler = cls(fileHandle, mode)
                bContinue = False
            except generic.LiDARSpatialIndexNotAvailable:
                if mode == generic.READ:
                    # try the next one
                    if len(availableIndices) > 0:
                        prefType = availableIndices.pop()
                        cls = SPDV4SpatialIndex.getClassForType(prefType)
                    else:
                        bContinue = False
                else:
                    bContinue = False

        return handler
        
SPATIALINDEX_GROUP = 'SPATIALINDEX'
SIMPLEPULSEGRID_GROUP = 'SIMPLEPULSEGRID'
        
class SPDV4SimpleGridSpatialIndex(SPDV4SpatialIndex):
    """
    Implementation of a simple grid index, which is currently
    the only type used in SPD V4 files.
    """
    def __init__(self, fileHandle, mode):
        SPDV4SpatialIndex.__init__(self, fileHandle, mode)
        self.si_cnt = None
        self.si_idx = None
        self.si_xPulseColName = 'X_IDX'
        self.si_yPulseColName = 'Y_IDX'
        self.indexType = SPDV4_INDEX_CARTESIAN
        # for caching
        self.lastExtent = None
        self.lastPulseSpace = None
        self.lastPulseIdx = None
        self.lastPulseIdxMask = None
        
        if mode == generic.READ or mode == generic.UPDATE:
            # read it in if it exists.
            group = None
            if SPATIALINDEX_GROUP in fileHandle:
                group = fileHandle[SPATIALINDEX_GROUP]
            if group is not None and SIMPLEPULSEGRID_GROUP in group:
                group = group[SIMPLEPULSEGRID_GROUP]
            
            if group is None:
                raise generic.LiDARSpatialIndexNotAvailable()
            else:
                self.si_cnt = group['PLS_PER_BIN'][...]
                self.si_idx = group['BIN_OFFSETS'][...]
                    
                # define the pulse data columns to use for the spatial index
                self.indexType = self.fileHandle.attrs['INDEX_TYPE']
                if self.indexType == SPDV4_INDEX_CARTESIAN:
                    self.si_xPulseColName = 'X_IDX'
                    self.si_yPulseColName = 'Y_IDX'
                elif self.indexType == SPDV4_INDEX_SPHERICAL:
                    self.si_xPulseColName = 'AZIMUTH'
                    self.si_yPulseColName = 'ZENITH'
                elif self.indexType == SPDV4_INDEX_SCAN:
                    self.si_xPulseColName = 'SCANLINE_IDX'
                    self.si_yPulseColName = 'SCANLINE'
                else:
                    msg = 'Unsupported index type %d' % indexType
                    raise generic.LiDARInvalidSetting(msg)                    
                
    def close(self):
        if self.mode == generic.CREATE and self.si_cnt is not None:
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
        
    def getSISubset(self, extent, overlap, extentAlignedWithIndex):
        """
        Internal method. Reads the required block out of the spatial
        index for the requested extent.
        """
        # snap the extent to the grid of the spatial index
        pixGrid = self.pixelGrid
        if extentAlignedWithIndex:
            xMin = extent.xMin
            xMax = extent.xMax
            yMin = extent.yMin
            yMax = extent.yMax
        else:
            xMin = gridindexutils.snapToGrid(extent.xMin, pixGrid.xMin, 
                pixGrid.xRes, gridindexutils.SNAPMETHOD_LESS)
            xMax = gridindexutils.snapToGrid(extent.xMax, pixGrid.xMax, 
                pixGrid.xRes, gridindexutils.SNAPMETHOD_GREATER)
            yMin = gridindexutils.snapToGrid(extent.yMin, pixGrid.yMin, 
                pixGrid.yRes, gridindexutils.SNAPMETHOD_LESS)
            yMax = gridindexutils.snapToGrid(extent.yMax, pixGrid.yMax, 
                pixGrid.yRes, gridindexutils.SNAPMETHOD_GREATER)

        # size of spatial index we need to read
        # round() ok since points should already be on the grid, nasty 
        # rounding errors propogated with ceil()                                    
        nrows = int(numpy.round((yMax - yMin) / self.pixelGrid.yRes))
        ncols = int(numpy.round((xMax - xMin) / self.pixelGrid.xRes))
        # add overlap 
        nrows += (overlap * 2)
        ncols += (overlap * 2)

        # create subset of spatial index to read data into
        cnt_subset = numpy.zeros((nrows, ncols), dtype=SPDV4_SIMPLEGRID_COUNT_DTYPE)
        idx_subset = numpy.zeros((nrows, ncols), dtype=SPDV4_SIMPLEGRID_INDEX_DTYPE)
        
        imageSlice, siSlice = gridindexutils.getSlicesForExtent(pixGrid, 
             self.si_cnt.shape, overlap, xMin, xMax, yMin, yMax)
             
        if imageSlice is not None and siSlice is not None:

            cnt_subset[imageSlice] = self.si_cnt[siSlice]
            idx_subset[imageSlice] = self.si_idx[siSlice]

        return idx_subset, cnt_subset

    def getPulsesSpaceForExtent(self, extent, overlap, extentAlignedWithIndex):
        """
        Get the space and indexes for pulses of the given extent.
        """
        # return cache
        if self.lastExtent is not None and self.lastExtent == extent:
            return self.lastPulseSpace, self.lastPulseIdx, self.lastPulseIdxMask
        
        idx_subset, cnt_subset = self.getSISubset(extent, overlap,
                extentAlignedWithIndex)
        nOut = self.fileHandle['DATA']['PULSES']['PULSE_ID'].shape[0]
        pulse_space, pulse_idx, pulse_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                idx_subset, cnt_subset, nOut)
                
        self.lastPulseSpace = pulse_space
        self.lastPulseIdx = pulse_idx
        self.lastPulseIdxMask = pulse_idx_mask
        self.lastExtent = copy.copy(extent)
                
        return pulse_space, pulse_idx, pulse_idx_mask

    def getPointsSpaceForExtent(self, extent, overlap, extentAlignedWithIndex):
        """
        Get the space and indexes for points of the given extent.
        """
        # TODO: cache
    
        # should return cached if exists
        pulse_space, pulse_idx, pulse_idx_mask = self.getPulsesSpaceForExtent(
                                    extent, overlap, extentAlignedWithIndex)
        
        pulsesHandle = self.fileHandle['DATA']['PULSES']
        nReturns = pulse_space.read(pulsesHandle['NUMBER_OF_RETURNS'])
        startIdxs = pulse_space.read(pulsesHandle['PTS_START_IDX'])

        nOut = self.fileHandle['DATA']['POINTS']['RETURN_NUMBER'].shape[0]
        point_space, point_idx, point_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                        startIdxs, nReturns, nOut)

        return point_space, point_idx, point_idx_mask

    def createNewIndex(self, pixelGrid):
        """
        Create a new spatial index
        """
        nrows, ncols = pixelGrid.getDimensions()
        self.si_cnt = numpy.zeros((nrows, ncols), 
                        dtype=SPDV4_SIMPLEGRID_COUNT_DTYPE)
        self.si_idx = numpy.zeros((nrows, ncols), 
                        dtype=SPDV4_SIMPLEGRID_INDEX_DTYPE)

        # save the pixelGrid
        self.pixelGrid = pixelGrid

    def setPulsesForExtent(self, extent, pulses, lastPulseID, 
                extentAlignedWithIndex):
        """
        Update the spatial index. Given extent and data works out what
        needs to be written.

        We can only do this on create so lastPulseID is an id to be added
        """
        pixGrid = self.pixelGrid
        if extentAlignedWithIndex:
            xMin = extent.xMin
            xMax = extent.xMax
            yMin = extent.yMin
            yMax = extent.yMax
        else:
            xMin = gridindexutils.snapToGrid(extent.xMin, pixGrid.xMin, 
                pixGrid.xRes, gridindexutils.SNAPMETHOD_LESS)
            xMax = gridindexutils.snapToGrid(extent.xMax, pixGrid.xMax, 
                pixGrid.xRes, gridindexutils.SNAPMETHOD_GREATER)
            yMin = gridindexutils.snapToGrid(extent.yMin, pixGrid.yMin, 
                pixGrid.yRes, gridindexutils.SNAPMETHOD_LESS)
            yMax = gridindexutils.snapToGrid(extent.yMax, pixGrid.yMax, 
                pixGrid.yRes, gridindexutils.SNAPMETHOD_GREATER)
                
        # size of spatial index we need to write
        # round() ok since points should already be on the grid, nasty 
        # rounding errors propogated with ceil()                                    
        nrows = int(numpy.round((yMax - yMin) / self.pixelGrid.xRes))
        ncols = int(numpy.round((xMax - xMin) / self.pixelGrid.xRes))

        mask, sortedBins, idx_subset, cnt_subset = gridindexutils.CreateSpatialIndex(
                pulses[self.si_yPulseColName], pulses[self.si_xPulseColName], self.pixelGrid.xRes, yMax, xMin,
                nrows, ncols, SPDV4_SIMPLEGRID_INDEX_DTYPE, 
                SPDV4_SIMPLEGRID_COUNT_DTYPE)
                
        # so we have unique indexes
        idx_subset = idx_subset + lastPulseID
                   
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
        # and the mask to remove points, waveforms etc
        # and the bin index to sort points, waveforms, etc
        return pulses, mask, sortedBins

    def canUpdateInPlace(self):
        "Data must always be sorted"
        return False

    def canAccessUnaligned(self):
        "Requests must always be aligned with index"
        return False

