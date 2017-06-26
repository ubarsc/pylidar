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

These are contained in the WRITESUPPORTEDOPTIONS module level variable.

+-----------------------+-------------------------------------------+
| Name                  | Use                                       |
+=======================+===========================================+
| FORMAT_VERSION        | LAS point format. Defaults to 1. Not sure |
|                       | what it means.                            |
+-----------------------+-------------------------------------------+
| RECORD_LENGTH         | LAS record length. Defaults to 28. Not    |
|                       | sure what it means.                       |
+-----------------------+-------------------------------------------+
| WAVEFORM_DESCR        | Data returned from                        |
|                       | getWavePacketDescriptions() which does an |
|                       | initial run through the data to get the   |
|                       | unique waveform info for writing to the   |
|                       | LAS header. No output waveforms are       |
|                       | written if this is not provided.          |
+-----------------------+-------------------------------------------+

Note that for writing, the extension currently controls the format witten:

+-----------+-----------------------+
| Extension | Format                |
+-----------+-----------------------+
| .las      | LAS                   |
+-----------+-----------------------+
| .laz      | LAZ (compressed las)  |
+-----------+-----------------------+
| .bin      | terrasolid            |
+-----------+-----------------------+
| .qi       | QFIT                  |
+-----------+-----------------------+
| .wrl      | VRML                  |
+-----------+-----------------------+
| other     | ASCII                 |
+-----------+-----------------------+

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
import datetime
from osgeo import osr

from rios import pixelgrid

from . import generic
# Fail slightly less drastically when running from ReadTheDocs
if os.getenv('READTHEDOCS', default='False') != 'True':
    from . import _las
    READSUPPORTEDOPTIONS = _las.getReadSupportedOptions()
    WRITESUPPORTEDOPTIONS = _las.getWriteSupportedOptions()

    # bring constants over
    FIRST_RETURN = _las.FIRST_RETURN
    LAST_RETURN = _las.LAST_RETURN

    # numpy needs a list before it will pull out fields, C returns
    # a tuple. Probably need to sort this at some stage    
    LAS_WAVEFORM_TABLE_FIELDS = list(_las.getExpectedWaveformFieldsForDescr())
else:
    READSUPPORTEDOPTIONS = None
    "driver options"
    WRITESUPPORTEDOPTIONS = None
    "driver options"
    FIRST_RETURN = None
    "for indexing pulses"
    LAST_RETURN = None
    "for indexing pulses"
    LAS_WAVEFORM_TABLE_FIELDS = []
    "for building waveforms - need to build unique table of these"
    
from . import gridindexutils

LAS_SIMPLEGRID_COUNT_DTYPE = numpy.uint32
"types for the spatial index"
LAS_SIMPLEGRID_INDEX_DTYPE = numpy.uint64
"types for the spatial index"

today = datetime.date.today()
DEFAULT_HEADER = {"GENERATING_SOFTWARE" : generic.SOFTWARE_NAME, 
"FILE_CREATION_DAY" : today.toordinal() - datetime.date(today.year, 1, 1).toordinal(), 
"FILE_CREATION_YEAR" : today.year}
"for new files"

HEADER_TRANSLATION_DICT = {generic.HEADER_NUMBER_OF_POINTS : 
                        'NUMBER_OF_POINT_RECORDS'}
"Non standard header names"

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
    Reader/Writer for .las files.
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
            options = WRITESUPPORTEDOPTIONS
            
        for key in userClass.lidarDriverOptions:
            if key not in options:
                msg = '%s not a supported las option' % repr(key)
                raise generic.LiDARInvalidSetting(msg)
        
        if mode == generic.READ:
            try:
                self.lasFile = _las.LasFileRead(fname, userClass.lidarDriverOptions)
            except _las.error as e:
                msg = 'cannot open as las file' + str(e)
                raise generic.LiDARFileException(msg)
                
        else:
            # create
            try:
                self.lasFile = _las.LasFileWrite(fname, userClass.lidarDriverOptions)
            except _las.error as e:
                msg = 'cannot create las file' + str(e)
                raise generic.LiDARFileException(msg)

        if mode == generic.READ:
            self.header = None
        else:
            self.header = DEFAULT_HEADER
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveformInfo = None
        self.lastReceived = None
        self.extent = None
        self.lastExtent = None
        self.firstBlockWritten = False # can't write header values when this is True

    @staticmethod        
    def getDriverName():
        return 'LAS'
        
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

    @staticmethod
    def getHeaderTranslationDict():
        """
        Return dictionary with non-standard header names
        """
        return HEADER_TRANSLATION_DICT

    def readPointsByPulse(self, colNames=None):
        """
        Read a 2d structured masked array containing the points
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
                self.lastRange = copy.copy(self.range)
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
                
                    self.lastExtent = copy.copy(extent)
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
        2d structured masked array containing information
        about the waveforms.
        """
        self.readData()                            
        # workaround - seems a structured array returned from
        # C doesn't work with masked arrays. The dtype looks different.
        # TODO: check this with a later numpy
        colNames = self.lastWaveformInfo.dtype.names
        info = self.subsetColumns(self.lastWaveformInfo, colNames)
        if info is not None:
            # TODO: cache?
            idx = self.lastPulses['WFM_START_IDX']
            cnt = self.lastPulses['NUMBER_OF_WAVEFORM_SAMPLES']

            # ok format the waveform info into a 2d (by pulse) structure using
            # the start and count fields (that connect with the pulse) 
            wave_idx, wave_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        idx, cnt)
            info = info[wave_idx]
            info = numpy.ma.array(info, mask=wave_idx_mask)

        return info
        
    def readTransmitted(self):
        """
        las (AFAIK) doesn't support transmitted
        """
        return None
        
    def readReceived(self):
        # need info for the indexing into the waveforms
        info = self.readWaveformInfo()

        # TODO: cache?

        # now the waveforms. Use the just created 2d array of waveform info's to
        # create the 3d one. 
        idx = info['RECEIVED_START_IDX']
        cnt = info['NUMBER_OF_WAVEFORM_RECEIVED_BINS']
            
        recv_idx, recv_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        idx, cnt)
        recv = self.lastReceived[recv_idx]
        recv = numpy.ma.array(recv, mask=recv_idx_mask)

        return recv
        
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
        if self.mode == generic.READ and not self.hasSpatialIndex():
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
        
    def readPulsesForExtentByBins(self, extent=None, colNames=None):
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
        # round() ok since points should already be on the grid, nasty 
        # rounding errors propogated with ceil()         
        nrows = int(numpy.round((extent.yMax - extent.yMin) / 
            extent.binSize))
        ncols = int(numpy.round((extent.xMax - extent.xMin) / 
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
        
    def readPointsForExtentByBins(self, extent=None, colNames=None, indexByPulse=False, 
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
        # round() ok since points should already be on the grid, nasty 
        # rounding errors propogated with ceil()         
        nrows = int(numpy.round((extent.yMax - extent.yMin) / 
            extent.binSize))
        ncols = int(numpy.round((extent.xMax - extent.xMin) / 
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
        
        pointsByBins = points[point_idx]
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

        if waveformInfo is not None and received is not None:
            # must convert received to uint16. In theory liblas can handle
            # uint8 also, but for simplicity have made it uint16
            for waveform in range(waveformInfo.shape[0]):
                gain = waveformInfo[waveform]['RECEIVE_WAVE_GAIN']
                offset = waveformInfo[waveform]['RECEIVE_WAVE_OFFSET']
                received[:,waveform] = (received[:,waveform] - gain) / offset
            received = received.astype(numpy.uint16)

        #print(pulses.shape, points.shape, received.shape, waveformInfo.shape)
        # TODO: flatten if necessary
        self.lasFile.writeData(self.header, pulses, points, waveformInfo,
                                received)
        self.firstBlockWritten = True

    def setPixelGrid(self, pixGrid):
        """
        Set the PixelGridDefn for the reading or 
        writing. We don't need to do much here
        apart from record the EPSG since LAS doesn't use a grid.
        """
        if self.mode == generic.READ or self.mode == generic.UPDATE:
            msg = 'Can only set new pixel grid when creating'
            raise generic.LiDARInvalidData(msg)

        if self.firstBlockWritten:
            msg = 'Projection can only be updated before first block written'
            raise generic.LiDARFunctionUnsupported(msg)
            
        sr = osr.SpatialReference()
        sr.ImportFromWkt(pixGrid.projection)
        # TODO: check this is ok for all coordinate systems
        epsg = sr.GetAttrValue("PROJCS|GEOGCS|AUTHORITY", 1)
        self.lasFile.setEPSG(int(epsg))
                
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
        
    def setHeader(self, newHeaderDict):
        """
        Update our cached dictionary
        """
        if self.mode == generic.READ:
            msg = 'Can only set header values on create'
            raise generic.LiDARInvalidSetting(msg)
            
        if self.firstBlockWritten:
            msg = 'Header can only be updated before first block written'
            raise generic.LiDARFunctionUnsupported(msg)
            
        for key in newHeaderDict.keys():
            self.header[key] = newHeaderDict[key]

    def setHeaderValue(self, name, value):
        """
        Just update one value in the header
        """
        if self.mode == generic.READ:
            msg = 'Can only set header values on create'
            raise generic.LiDARInvalidSetting(msg)

        if self.firstBlockWritten:
            msg = 'Header can only be updated before first block written'
            raise generic.LiDARFunctionUnsupported(msg)

        self.header[name] = value

    def setScaling(self, colName, arrayType, gain, offset):
        """
        Set the scaling for the given column name. Currently
        scaling is only supported for X, Y and Z columns for points,
        or optional fields.
        """
        if self.mode == generic.READ:
            msg = 'Can only set scaling values on create'
            raise generic.LiDARInvalidSetting(msg)
            
        if arrayType != generic.ARRAY_TYPE_POINTS:
            msg = 'Can only set scaling for points'
            raise generic.LiDARInvalidSetting(msg)

        if self.firstBlockWritten:
            msg = 'Scaling can only be updated before first block written'
            raise generic.LiDARFunctionUnsupported(msg)

        try:            
            self.lasFile.setScaling(colName, gain, offset)
        except _las.error as e:
            raise generic.LiDARArrayColumnError(str(e))

    def getScaling(self, colName, arrayType):
        """
        Returns the scaling (gain, offset) for the given column name
        reads from our cache since only written to file on close

        Raises generic.LiDARArrayColumnError if no scaling (yet) 
        set for this column.
        """
        if self.mode != generic.READ:
            msg = 'Can only get scaling values on read'
            raise generic.LiDARInvalidSetting(msg)
            
        if arrayType != generic.ARRAY_TYPE_POINTS:
            msg = 'Can only get scaling for points'
            raise generic.LiDARInvalidSetting(msg)

        try:
            scaling = self.lasFile.getScaling(colName)
        except _las.error as e:
            raise generic.LiDARArrayColumnError(str(e))
            
        return scaling

    def setNativeDataType(self, colName, arrayType, dtype):
        """
        Set the native dtype (numpy.int16 etc)that a column is stored
        as internally after scaling (if any) is applied.
        
        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        
        generic.LiDARArrayColumnError is raised if this cannot be set for the format.
        
        The default behaviour is to create new columns in the correct type for 
        the format, or if they are optional, in the same type as the input array.
        """
        if self.mode == generic.READ:
            msg = 'Can only set scaling values on update or create'
            raise generic.LiDARInvalidSetting(msg)

        if arrayType != generic.ARRAY_TYPE_POINTS:
            msg = 'Can only get data type for points'
            raise generic.LiDARInvalidSetting(msg)
            
        if self.firstBlockWritten:
            msg = 'Data type can only be updated before first block written'
            raise generic.LiDARFunctionUnsupported(msg)
            
        try:
            self.lasFile.setNativeDataType(colName, dtype)
        except _las.error as e:
            raise generic.LiDARArrayColumnError(str(e))
                                
    def getNativeDataType(self, colName, arrayType):
        """
        Return the native dtype (numpy.int16 etc)that a column is stored
        as internally after scaling is applied. Provided so scaling
        can be adjusted when translating between formats.
        
        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        """
        if arrayType != generic.ARRAY_TYPE_POINTS:
            raise generic.LiDARInvalidSetting('Unsupported array type')
        
        try:
            # implemented for both read and write
            dtype = self.lasFile.getNativeDataType(colName)
        except _las.error as e:
            raise generic.LiDARArrayColumnError(str(e))
            
        return dtype

    def getScalingColumns(self, arrayType):
        """
        Return the list of columns that need scaling. Only valid on write
        """
        if arrayType != generic.ARRAY_TYPE_POINTS:
            raise generic.LiDARInvalidSetting('Unsupported array type')
        return self.lasFile.getScalingColumns()

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
        
    @staticmethod        
    def getDriverName():
        return 'LAS'

    @staticmethod
    def getHeaderTranslationDict():
        """
        Return dictionary with non-standard header names
        """
        return HEADER_TRANSLATION_DICT
    
def getWavePacketDescriptions(fname):
    """
    When writing a LAS file, it is necessary to write information to a
    table in the header that really belongs to the waveforms. This 
    function reads the waveform info from the input file (in any format) and 
    gathers the unique information from it so it can be passed to as
    the WAVEFORM_DESCR LAS driver option.
    
    Note: LAS only supports received waveforms.
    
    """
    from pylidar import lidarprocessor
    
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile(fname, lidarprocessor.READ)
    
    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)
    
    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.uniqueInfo = None
    
    lidarprocessor.doProcessing(gatherWavePackets, dataFiles, otherArgs,
                    controls=controls)
                    
    return otherArgs.uniqueInfo

    
def gatherWavePackets(data, otherArgs):
    """
    Called from lidarprocessor, is the function that does the identification of 
    unique waveform length and gain and offset.
    """
    from . import gridindexutils
    info = data.input.getWaveformInfo()

    if info is None:
        msg = 'input file does not have waveforms'
        raise generic.LiDARInvalidData(msg)
    
    # check they all the fields we need
    for field in LAS_WAVEFORM_TABLE_FIELDS:
        if field not in info.dtype.names:
            msg = 'Input file does not have the %s field' % field
            raise generic.LiDARArrayColumnError(msg)

    # need to flatten it
    firstField = LAS_WAVEFORM_TABLE_FIELDS[0]
    ptsCount = info[firstField].count()
    outInfo = numpy.empty(ptsCount, dtype=info.data.dtype)
    returnNumber = numpy.empty(ptsCount, dtype=numpy.uint64) # not needed
    gridindexutils.flattenMaskedStructuredArray(info.data, 
                info[firstField].mask, outInfo, returnNumber)
            
    info = outInfo
            
    # just the fields we are interested in
    info = info[LAS_WAVEFORM_TABLE_FIELDS]
    
    if otherArgs.uniqueInfo is not None:
        # append it so we can do unique on the whole thing
        info = numpy.append(otherArgs.uniqueInfo, info)
    
    # save the unique ones
    otherArgs.uniqueInfo = numpy.unique(info)

