"""
Driver for .las files. Uses lastools (https://github.com/LAStools/LAStools).

Read Driver Options
-------------------

These are contained in the READSUPPORTEDOPTIONS module level variable.

+-----------------------+-------------------------------------------+
| Name                  | Use                                       |
+=======================+===========================================+
| BUILD_PULSES          | a boolean. If set to true (the default)   |
|                       | pylidar attempts to build pulses assuming |
|                       | that data is in time sequential order. If |
|                       | false, a 'fake' pulse is created for each |
|                       | point.                                    |
+-----------------------+-------------------------------------------+
| BIN_SIZE              | A number. For files with a spatial index  |
|                       | present this is the bin size that the     |
|                       | File be presented at. Las indexes can use |
|                       | any arbitary bin size it seems, but       |
|                       | works to specific ones which can be set   |
|                       | with this option for this file. An error  |
|                       | will be raised if a spatial read is       |
|                       | attempted and this hasn't been set.       |
+-----------------------+-------------------------------------------+
| PULSE_INDEX           | Either FIRST_RETURN or LAST_RETURN        |
|                       | Dictates which point will be used to set  |
|                       | the X_IDX and Y_IDX pulse fields          |
+-----------------------+-------------------------------------------+

Write Driver Options
--------------------

No driver options for writing are supported yet.

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
import copy
import numpy
from osgeo import osr

from rios import pixelgrid

from . import generic
from . import _las
from . import gridindexutils

READSUPPORTEDOPTIONS = _las.getReadSupportedOptions()
WRITESUPPORTEDOPTIONS = _las.getWriteSupportedOptions()

# bring constants over
FIRST_RETURN = _las.FIRST_RETURN
LAST_RETURN = _las.LAST_RETURN

# types for the spatial index
LAS_SIMPLEGRID_COUNT_DTYPE = numpy.uint32
LAS_SIMPLEGRID_INDEX_DTYPE = numpy.uint64

def isLasFile(fname):
    """
    Helper function that looks at the start of the file
    to determine if it is a las file or not
    """
    # The las library reads the whole file before failing
    # and prints a whole lot of stuff to stdout.
    # Easier just to see if file starts with 'LASF'
    fh = open(fname, 'rb')
    data = fh.read(4)
    fh.close()
        
    if data != b'LASF':
        return False
    else:
        return True

class LasFile(generic.LiDARFile):
    """
    Reader for .las files.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)
        if mode != generic.READ and mode != generic.CREATE:
            msg = 'Las driver is read or create only'
            raise generic.LiDARInvalidSetting(msg)

        if mode == generic.READ and not isLasFile(fname):
            msg = 'not a las file'
            raise generic.LiDARFileException(msg)

        # check if the options are all valid. Good to check in case of typo.
        # hard to do this in C
        if mode == generic.READ:
            options = READSUPPORTEDOPTIONS
        else:
            options == WRITESUPPORTEDOPTIONS
            
        for key in userClass.lidarDriverOptions:
            if key not in options:
                msg = '%s not a supported las option' % repr(key)
                raise generic.LiDARInvalidSetting(msg)
        
        try:
            self.lasFile = _las.LasFileRead(fname, userClass.lidarDriverOptions)
        except _las.error as e:
            msg = 'cannot open as las file' + str(e)
            raise generic.LiDARFileException(msg)

        self.header = None
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveformInfo = None
        self.lastReceived = None
        self.extent = None
        self.lastExtent = None

    @staticmethod        
    def getDriverName():
        return 'las'
        
    def close(self):
        self.lasFile = None
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveformInfo = None
        self.lastReceived = None
        self.extent = None
        self.lastExtent = None

    def readPointsByPulse(self, colNames=None):
        """
        Read a 3d structured masked array containing the points
        for each pulse.
        """
        pulses = self.readPulsesForRange()
        points = self.readPointsForRange()
        if points.size == 0:
            return None
        nReturns = pulses['NUMBER_OF_RETURNS']
        startIdxs = pulses['PTS_START_IDX']

        point_idx, point_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(        
                startIdxs, nReturns)
                
        pointsByPulse = points[point_idx]
        
        if colNames is None:
            # workaround - seems a structured array returned from
            # C doesn't work with masked arrays. The dtype looks different.
            # TODO: check this with a later numpy
            colNames = pointsByPulse.dtype.names
            
        pointsByPulse = self.subsetColumns(pointsByPulse, colNames)
        points = numpy.ma.array(pointsByPulse, mask=point_idx_mask)
        
        return points
        
    def hasSpatialIndex(self):
        """
        Returns True if the las file has an associated spatial
        index.
        
        """
        return self.lasFile.hasSpatialIndex
    
    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        """
        self.range = copy.copy(pulseRange)
        # return True if we can still read data
        # we just assume we can until we find out
        # after a read that we can't
        return not self.lasFile.finished
        
    def readData(self, extent=None):
        """
        Internal method. Just reads into the self.last* fields
        """
        # assume only one of self.range or self.extent is set...
        if self.range is not None:
            if self.lastRange is None or self.range != self.lastRange:
                pulses, points, info, recv = self.lasFile.readData(self.range.startPulse, 
                            self.range.endPulse)
                self.lastRange = self.range        
                self.lastPoints = points
                self.lastPulses = pulses
                self.lastWaveformInfo = info
                self.lastReceived = recv
                
        else:
            if extent is None:
                extent = self.extent
                
            if extent is not None:
                if self.lastExtent is None or extent != self.lastExtent:
                    # tell liblas to only read data in from the current extent
                    # this may be on a different grid to the pixelgrid - it doesn't matter
                    # since the spatial index isn't grid based.
                    self.lasFile.setExtent(extent.xMin, extent.xMax,
                        extent.yMin, extent.yMax)                
            
                    pulses, points, info, recv = self.lasFile.readData()
                
                    self.lastExtent = extent
                    self.lastPoints = points
                    self.lastPulses = pulses
                    self.lastWaveformInfo = info
                    self.lastReceived = recv
            else:
                msg = 'must set extent or range before reading data'
                raise ValueError(msg)
                                        
    def readPointsForRange(self, colNames=None):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        self.readData()                            
        return self.subsetColumns(self.lastPoints, colNames)
        
    def readPulsesForRange(self, colNames=None):
        """
        Reads the pulses for the current range. Returns a 1d array.

        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        self.readData()                            
        return self.subsetColumns(self.lastPulses, colNames)
        
    def readWaveformInfo(self):
        """
        3d structured masked array containing information
        about the waveforms.
        """
        self.readData()                            
        # workaround - seems a structured array returned from
        # C doesn't work with masked arrays. The dtype looks different.
        # TODO: check this with a later numpy
        colNames = self.lastWaveformInfo.dtype.names
        info = self.subsetColumns(self.lastWaveformInfo, colNames)
        if info is not None:
            info = numpy.expand_dims(info, axis=0)
            info = numpy.expand_dims(info, axis=0)
            mask = numpy.zeros_like(info, dtype=numpy.bool)
            info = numpy.ma.array(info, mask=mask)
        return info
        
    def readTransmitted(self):
        """
        las (AFAIK) doesn't support transmitted
        """
        return None
        
    def readReceived(self):
        self.readData()
        return self.lastReceived
        
    def getTotalNumberPulses(self):
        """
        If BUILD_PULSES == False then the number of pulses
        will equal the number of points and we can return that.
        Otherwise we have no idea how many so we raise an exception
        to flag that.
        
        """
        if not self.lasFile.build_pulses:
            return self.getHeaderValue('NUMBER_OF_POINT_RECORDS')
        else:
            raise generic.LiDARFunctionUnsupported()
            
    # spatial methods - will raise exception if not spatial index present
    def setExtent(self, extent):
        """
        Set the extent for reading for the ForExtent() functions.
        """
        if not self.hasSpatialIndex():
            msg = 'las file has no index. See lasindex'
            raise generic.LiDARFunctionUnsupported(msg)
            
        self.extent = extent

        # we don't have to do anything clever here with extent not 
        # matching the spatial index etc since las doesn't have a 
        # very tied down concept of it. We will just read whatever
        # the extent is        
        
    @staticmethod
    def getWktFromEPSG(epsg):
        """
        Gets the WKT from a given EPSG
        via GDAL.
        """
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg)
        wkt = sr.ExportToWkt()
        return wkt
        
    def getPixelGrid(self):
        """
        Return the PixelGridDefn for this file

        """
        if not self.hasSpatialIndex():
            msg = 'las file has no index. See lasindex'
            raise generic.LiDARFunctionUnsupported(msg)
            
        header = self.getHeader()
        try:
            epsg = self.lasFile.getEPSG()
            wkt = self.getWktFromEPSG(epsg)
        except _las.error:
            # no projection info
            wkt = None
            
        binSize = self.lasFile.binSize
        if binSize == 0:
            msg = 'Must set BIN_SIZE option to read Las files spatially'
            raise generic.LiDARFunctionUnsupported(msg)

        pixgrid = pixelgrid.PixelGridDefn(projection=wkt, xMin=header['MIN_X'],
                        xMax=header['MAX_X'], yMin=header['MIN_Y'], 
                        yMax=header['MAX_Y'], xRes=binSize, yRes=binSize)
        return pixgrid

    def readPointsForExtent(self, colNames=None):
        """
        Read all the points within the given extent
        as 1d structured array. The names of the fields in this array
        will be defined by the driver.

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        self.readData()
        return self.subsetColumns(self.lastPoints, colNames)

    def readPulsesForExtent(self, colNames=None):
        """
        Read all the pulses within the given extent
        as 1d structured array. The names of the fields in this array
        will be defined by the driver.

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        self.readData()
        return self.subsetColumns(self.lastPulses, colNames)
        
    def readPulsesForExtentByBins(extent=None, colNames=None):
        """
        Read all the pulses within the given extent as a 3d structured 
        masked array to match the block/bins being used.
        
        The extent/binning for the read data can be overriden by passing in a
        Extent instance.

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        # now spatially index the pulses
        if extent is None:
            extent = self.extent
            
        self.readData(extent)
        
        # TODO: cache somehow with colNames
        nrows = int(numpy.ceil((extent.yMax - extent.yMin) / 
            extent.binSize))
        ncols = int(numpy.ceil((extent.xMax - extent.xMin) / 
            extent.binSize))
        nrows += (self.controls.overlap * 2)
        ncols += (self.controls.overlap * 2)
                
        xidx = self.lastPulses['X_IDX']
        yidx = self.lastPulses['Y_IDX']
        
        mask, sortedbins, idx, cnt = gridindexutils.CreateSpatialIndex(xidx,
                yidx, extent.binSize, extent.yMax, extent.xMin, 
                nrows, ncols, LAS_SIMPLEGRID_INDEX_DTYPE, 
                LAS_SIMPLEGRID_COUNT_DTYPE)

        pulse_idx, pulse_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(        
                idx, cnt)
                
        pulses = self.lastPulses[mask]
        pulses = pulses[sortedbins]
        
        pulsesByBins = pulses[pulse_idx]
        if colNames is None:
            # workaround - seems a structured array returned from
            # C doesn't work with masked arrays. The dtype looks different.
            # TODO: check this with a later numpy
            colNames = pulsesByBins.dtype.names
            
        pointsByBins = self.subsetColumns(pointsByBins, colNames)
        pulsesByBins = numpy.ma.array(pulsesByBins, mask=pulse_idx_mask)              
        return pulsesByBins
        
    def readPointsForExtentByBins(extent=None, colNames=None, indexByPulse=False, 
                returnPulseIndex=False):
        """
        Read all the points within the given extent as a 3d structured 
        masked array to match the block/bins being used.
        
        The extent/binning for the read data can be overriden by passing in a
        Extent instance.

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        
        Pass indexByPulse=True to bin the points by the locations of the pulses
            instead of the points.
        
        Pass returnPulseIndex=True to also return a masked 3d array of 
            the indices into the 1d pulse array (as returned by 
            readPulsesForExtent())
        
        """
        # now spatially index the points
        if extent is None:
            extent = self.extent
            
        self.readData(extent)
        
        # TODO: cache somehow with colNames
        nrows = int(numpy.ceil((extent.yMax - extent.yMin) / 
            extent.binSize))
        ncols = int(numpy.ceil((extent.xMax - extent.xMin) / 
            extent.binSize))
        nrows += (self.controls.overlap * 2)
        ncols += (self.controls.overlap * 2)
                
        if indexByPulse:
            xidx = self.lastPulses['X_IDX']
            yidx = self.lastPulses['Y_IDX']
            nreturns = self.lastPulses['NUMBER_OF_RETURNS']
            xidx = numpy.repeat(xidx, nreturns)
            yidx = numpy.repeat(yidx, nreturns)
        else:
            xidx = self.lastPoints['X']
            yidx = self.lastPoints['Y']
        
        mask, sortedbins, idx, cnt = gridindexutils.CreateSpatialIndex(xidx,
                yidx, extent.binSize, extent.yMax, extent.xMin, 
                nrows, ncols, LAS_SIMPLEGRID_INDEX_DTYPE, 
                LAS_SIMPLEGRID_COUNT_DTYPE)

        point_idx, point_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(        
                idx, cnt)
                
        points = self.lastPoints[mask]
        points = points[sortedbins]
        
        pointsByBins = point[point_idx]
        if colNames is None:
            # workaround - seems a structured array returned from
            # C doesn't work with masked arrays. The dtype looks different.
            # TODO: check this with a later numpy
            colNames = pointsByBins.dtype.names
            
        pointsByBins = self.subsetColumns(pointsByBins, colNames)
        pointsByBins = numpy.ma.array(pointsByBins, mask=point_idx_mask)
        
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
            pulse_idx_3dmask = numpy.ma.array(pulse_idx_3d, mask=point_idx_mask)
            
            # return 2 things
            return pointsByBins, pulse_idx_3dmask
        else:
            # just return the points
            return pointsByBins

    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        This driver does not support writing so ignore if reading,
        throw and error otherwise.
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return
        
        msg = 'las driver does not support update/creating'
        raise generic.LiDARWritingNotSupported(msg)
                
    def getHeader(self):
        """
        Return the Las header as a dictionary.
        
        """
        if self.header is None:
            self.header = self.lasFile.readHeader()
            
        return self.header
        
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.getHeader()[name]

class LasFileInfo(generic.LiDARFileInfo):
    """
    Class that gets information about a .las file
    and makes it available as fields.
    """
    def __init__(self, fname):
        generic.LiDARFileInfo.__init__(self, fname)
        
        if not isLasFile(fname):
            msg = 'not a Las file'
            raise generic.LiDARFormatNotUnderstood(msg)
            
        # open the file object
        try:
            lasFile = _las.LasFileRead(fname, {})
        except _las.error:
            msg = 'error opening las file'
            raise generic.LiDARFormatNotUnderstood(msg)
            
        # get header
        self.header = lasFile.readHeader()
        
        # projection
        try:
            epsg = lasFile.getEPSG()
            self.wkt = LasFile.getWktFromEPSG(epsg)
        except _las.error:
            # no projection info
            self.wkt = None
            
        self.hasSpatialIndex = lasFile.hasSpatialIndex
        
    
    
