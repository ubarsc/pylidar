"""
Driver for .las files. Uses lastools (https://github.com/LAStools/LAStools).

Driver Options
--------------

These are contained in the SUPPORTEDOPTIONS module level variable.

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
|                       | Will be presented at. Las indexes can use |
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

SUPPORTEDOPTIONS = _las.getSupportedOptions()

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

        if mode != generic.READ:
            msg = 'Las driver is read only'
            raise generic.LiDARInvalidSetting(msg)

        if not isLasFile(fname):
            msg = 'not a las file'
            raise generic.LiDARFileException(msg)
        
        try:
            self.lasFile = _las.LasFile(fname, userClass.lidarDriverOptions)
        except:
            msg = 'cannot open as las file'
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
        
    def readData(self):
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
                
        elif self.extent is not None:
            if self.lastExtent is None or self.extent != self.lastExtent:
                # tell liblas to only read data in from the current extent
                # this may be on a different grid to the pixelgrid - it doesn't matter
                # since the spatial index isn't grid based.
                self.lasFile.setExtent(self.extent.xMin, self.extent.xMax,
                        self.extent.yMin, self.extent.yMax)                
            
                pulses, points, info, recv = self.lasFile.readData()
                
                nrows = int(numpy.ceil((self.extent.yMax - self.extent.yMin) / 
                        self.extent.binSize))
                ncols = int(numpy.ceil((self.extent.xMax - self.extent.xMin) / 
                        self.extent.binSize))
                nrows += (self.controls.overlap * 2)
                ncols += (self.controls.overlap * 2)
                
                # grid the data
                #mask, sortedbins, idx, cnt = gridindexutils.CreateSpatialIndex(
                #                pulses['X'
                
                self.lastExtent = self.extent
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
        return self.lastWaveformInfo
        
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
        
    def getPixelGrid(self):
        """
        Return the PixelGridDefn for this file

        """
        if not self.hasSpatialIndex():
            msg = 'las file has no index. See lasindex'
            raise generic.LiDARFunctionUnsupported(msg)
            
        header = self.getHeader()
        epsg = self.lasFile.getEPSG()
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg)
        wkt = sr.exportToWkt()
        binSize = self.lasFile.binSize
        if binSize == 0:
            msg = 'Must set BIN_SIZE option to read Las files spatially'
            raise generic.LiDARFunctionUnsupported(msg)
        
        pixgrid = pixelgrid.PixelGridDefn(projection=wkt, xMin=header['X_MIN'],
                        xMax=header['X_MAX'], yMin=header['Y_MIN'], 
                        yMax=header['Y_MAX'], xRes=binSize, yRes=binSize)
        return pixgrid

    def readPointsForExtent(self, colNames=None):
        """
        Read all the points within the given extent
        as 1d structured array. The names of the fields in this array
        will be defined by the driver.

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        raise NotImplementedError()

    def readPulsesForExtent(self, colNames=None):
        """
        Read all the pulses within the given extent
        as 1d structured array. The names of the fields in this array
        will be defined by the driver.

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        raise NotImplementedError()

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
            