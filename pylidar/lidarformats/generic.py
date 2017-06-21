
"""
Base class for LiDAR format reader/writers
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

import abc
import numpy
from .. import basedriver
from .. import __version__

READ = basedriver.READ
"access modes passed to driver constructor"
UPDATE = basedriver.UPDATE
"access modes passed to driver constructor"
CREATE = basedriver.CREATE
"access modes passed to driver constructor"

MESSAGE_WARNING = 0
"to be passed to message handler function controls.messageHandler"
MESSAGE_INFORMATION = 1
"to be passed to message handler function controls.messageHandler"
MESSAGE_DEBUG = 2
"to be passed to message handler function controls.messageHandler"

FIELD_POINTS_RETURN_NUMBER = 1
"'standard' fields that have different names for different formats"
FIELD_PULSES_TIMESTAMP = 2
"'standard' fields that have different names for different formats"

HEADER_NUMBER_OF_POINTS = 1
"'standard' header fields that have different names for different formats"

ARRAY_TYPE_POINTS = 0
"""
For use in userclass.LidarData.translateFieldNames() and 
LiDARFile.getTranslationDict()
"""
ARRAY_TYPE_PULSES = 1
"""
For use in userclass.LidarData.translateFieldNames() and 
LiDARFile.getTranslationDict()
"""
ARRAY_TYPE_WAVEFORMS = 2
"""
For use in userclass.LidarData.translateFieldNames() and 
LiDARFile.getTranslationDict()
"""

CLASSIFICATION_CREATED = 0
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_UNCLASSIFIED = 1
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_GROUND = 2
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_LOWVEGE = 3
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_MEDVEGE = 4
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_HIGHVEGE = 5
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_BUILDING = 6
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_LOWPOINT = 7
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_HIGHPOINT = 8
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_WATER = 9
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_RAIL = 10
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_ROAD = 11
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_BRIDGE = 12
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_WIREGUARD = 13
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_WIRECOND = 14
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_TRANSTOWER = 15
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_INSULATOR = 16
"Classifications from the LAS spec. See LiDARFile.recodeClassification"
CLASSIFICATION_TRUNK = 100
"Extended classifications"
CLASSIFICATION_FOLIAGE = 101
"Extended classifications"
CLASSIFICATION_BRANCH = 102
"Extended classifications"

RECODE_TO_DRIVER = 0
"Codes to pass to LiDARFile.recodeClassification"
RECODE_TO_LAS = 1
"Codes to pass to LiDARFile.recodeClassification"

CLASSIFICATION_COLNAME = "CLASSIFICATION"
"Name of column to treat as classification"

# For writing to files when needed
SOFTWARE_NAME = 'PyLidar %s' % __version__

class LiDARFileException(Exception):
    "Base class for LiDAR format reader/writers"
    
class LiDARFormatNotUnderstood(LiDARFileException):
    "Raised when driver cannot open file"
    
class LiDARFormatDriverNotFound(LiDARFileException):
    "None of the drivers can open the file"
    
class LiDARInvalidData(LiDARFileException):
    "Something is wrong with the data read or given"
    
class LiDARInvalidSetting(LiDARFileException):
    "Setting does not make sense"
    
class LiDARNonSpatialProcessing(LiDARFileException):
    "Functionality not available when not processing spatially"
    
class LiDARFunctionUnsupported(LiDARFileException):
    "Function unsupported by LiDAR driver"
    
class LiDARArrayColumnError(LiDARFileException):
    "Unsupported operation on a structured array"
    
class LiDARSpatialIndexNotAvailable(LiDARFileException):
    "The specified spatial index not available for this file"

class LiDARPulseIndexUnsupported(LiDARFileException):
    "The specified pulse index method is currently unsupported"
    
class LiDARWritingNotSupported(LiDARFunctionUnsupported):
    "driver does not support writing"

class LiDARScalingError(LiDARInvalidData):
    "scaled data is outside the bounds of the data type"
    
class PulseRange(object):
    """
    Class for setting the range of pulses to read/write
    for non spatial mode.
    Note: range does not include endPulse
    """
    def __init__(self, startPulse, endPulse):
        self.startPulse = startPulse
        self.endPulse = endPulse
        
    def __eq__(self, other):
        return (self.startPulse == other.startPulse and 
                self.endPulse == other.endPulse)
                
    def __ne__(self, other):
        return (self.startPulse != other.startPulse or
                self.endPulse != other.endPulse)

class LiDARFile(basedriver.Driver):
    """
    Base class for all LiDAR Format reader/writers.
    
    It is intended that very little work happens until the user actually
    asks for the data - then read it in. Subsequent calls for the same
    extent should return cached data.
    """
    def __init__(self, fname, mode, controls, userClass):
        """
        Constructor. Derived drivers should open the file and read any
        spatial index data out. 
        
        Raise generic.LiDARFormatNotUnderstood if file not supported
        by driver - all drivers may be asked to open a file to determine
        which one supports the format of the file. So a good idea to be 
        sure that this file correct for your driver before returning 
        successfully.
        """
        basedriver.Driver.__init__(self, fname, mode, controls, userClass)

        # a list that holds the translation between internal codes
        # and the LAS spec ones (above)
        # each item of the list should be a tuple with 
        # (internalCode, lasCode)
        # derived classes should update this list with codes
        # if they differ from the LAS spec
        self.classificationTranslation = []

    # can't combine static and abstract in Python 2.x
    @staticmethod        
    def getDriverName():
        """
        Return name of driver - just a short unique name is fine.
        """
        raise NotImplementedError()
        
    # can't combine static and abstract in Python 2.x
    @staticmethod
    def getTranslationDict(arrayType):
        """
        Return a dictionary keyed on FIELD_* values (above)
        that can be used to translate field names between the formats
        arrayType is the type of array that is to be translated (ARRAY_TYPE_*)

        For use by the :func:`pylidar.userclases.LidarData.translateFieldNames`
        function.
        """
        raise NotImplementedError()

    @staticmethod
    def getHeaderTranslationDict():
        """
        Return a dictionary keyed on HEADER_* values (above)
        that can be used to translate dictionary field names between the formats
        """
        raise NotImplementedError()

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

    def readPulsesForExtentByBins(extent=None, colNames=None):
        """
        Read all the pulses within the given extent as a 3d structured 
        masked array to match the block/bins being used.
        
        The extent/binning for the read data can be overriden by passing in a
        Extent instance.

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        raise NotImplementedError()
        
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
        raise NotImplementedError()

    @abc.abstractmethod        
    def readPointsByPulse(self):     
        """
        Read a 2d structured masked array containing the points
        for each pulse.
        """
        raise NotImplementedError()

    @abc.abstractmethod        
    def readWaveformInfo(self):
        """
        2d structured masked array containing information
        about the waveforms.
        """
        raise NotImplementedError()

    @abc.abstractmethod        
    def readTransmitted(self):
        """
        Read the transmitted waveform for all pulses
        returns a 3d masked array. 
        """
        raise NotImplementedError()
        
    @abc.abstractmethod
    def readReceived(self):
        """
        Read the received waveform for all pulses
        returns a 2d masked array
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def hasSpatialIndex(self):
        """
        Returns True if file has a spatial index defined
        """
        raise NotImplementedError()
        
    # see below for no spatial index
    @abc.abstractmethod
    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        
        Return False if outside the range of data.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def readPointsForRange(self, colNames=None):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        raise NotImplementedError()
        
    @abc.abstractmethod
    def readPulsesForRange(self, colNames=None):
        """
        Reads the pulses for the current range. Returns a 1d array.

        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        raise NotImplementedError()
        
    @abc.abstractmethod
    def getTotalNumberPulses(self):
        """
        Returns the total number of pulses in this file. Used for progress.
        
        Raise a LiDARFunctionUnsupported error if driver does not support
        easily finding the total number of pulses.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def getHeader(self):
        """
        Return a dictionary of key/value pairs containing header info
        """
        raise NotImplementedError()
        
    def setHeader(self, newHeaderDict):
        """
        Update all of the header values as a dictionary
        """
        raise NotImplementedError()
            
    @abc.abstractmethod
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        raise NotImplementedError()
        
    def setHeaderValue(self, name, value):
        """
        Just update one value in the header
        """
        raise NotImplementedError()
        
    def setScaling(self, colName, arrayType, gain, offset):
        """
        Set the scaling for the given column name
        
        arrayType is one of the ARRAY_TYPE_* constants
        """
        raise NotImplementedError()
        
    def getScaling(self, colName, arrayType):
        """
        Returns the scaling (gain, offset) for the given column name

        arrayType is one of the ARRAY_TYPE_* constants.
        
        Raises generic.LiDARArrayColumnError if no scaling (yet) 
        set for this column.
        """
        raise NotImplementedError()
        
    def setNativeDataType(self, colName, arrayType, dtype):
        """
        Set the native dtype (numpy.int16 etc) that a column is stored
        as internally after scaling (if any) is applied.
        
        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        
        generic.LiDARArrayColumnError is raised if this cannot be set for the column.
        
        The default behaviour is to create new columns in the correct type for 
        the format, or if they are optional, in the same type as the input array.
        """
        raise NotImplementedError()
        
    def getNativeDataType(self, colName, arrayType):
        """
        Return the native dtype (numpy.int16 etc) that a column is stored
        as internally after scaling (if any) is applied. Provided so scaling
        can be adjusted when translating between formats.
        
        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        
        Raises generic.LiDARArrayColumnError if information cannot be
        found for the column.
        """
        raise NotImplementedError()

    def setNullValue(self, colName, arrayType, value, scaled=True):
        """
        Set the 'null' value for the given column.

        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants

        By default the value is treated as the scaled value, but this can
        be changed with the 'scaled' parameter.

        generic.LiDARArrayColumnError is raised if this cannot be set for the column.
        """
        raise NotImplementedError()

    def getNullValue(self, colName, arrayType, scaled=True):
        """
        Get the 'null' value for the given column.

        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants

        By default the returned value is scaled, change this with the 'scaled'
        parameter.

        Raises generic.LiDARArrayColumnError if information cannot be
        found for the column.
        """
        raise NotImplementedError()

    def getScalingColumns(self, arrayType):
        """
        Return a list of columns names that will need scaling to be set 
        when creating a new file.

        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        """
        return []

    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        Write data to file. pulses to be 1d structured array. points to be
        2d points-by-pulses format. waveformInfo, transmitted and 
        received to be 2d by-pulses format. 
        
        Pass None if no data to be written or data unchanged (for update).
        """
        raise NotImplementedError()
        
    @abc.abstractmethod
    def close(self):
        """
        Write any updated spatial index and close any file handles.
        """
        raise NotImplementedError()

    def recodeClassification(self, array, direction, colNames=None):
        """
        Recode classification column (if it exists in array)
        in the specified direction. 

        If array is not structured and colNames is a string equal
        to CLASSIFICATION_COLNAME, then the array is treated as the
        classification column.
        """
        if array.dtype.fields is None:
            # non structured
            if (colNames is None or not isinstance(colNames, str) or 
                    colNames != CLASSIFICATION_COLNAME):
                return
            else:
                classification = array
        else:
            # structured
            if CLASSIFICATION_COLNAME not in array.dtype.fields:
                return
            else:
                classification = array[CLASSIFICATION_COLNAME]

        maskList = []
        # we have to do this in 2 steps since we are changing
        # the data as we go which can lead to unexpected resuls
        # so we calculate all the masks before changing anything,
        # then apply them
        for internalCode, lasCode in self.classificationTranslation:
            if direction == RECODE_TO_DRIVER:
                mask = (classification == lasCode)
                maskList.append((mask, internalCode))
            else:
                mask = (classification == internalCode)
                maskList.append((mask, lasCode))

        for mask, code in maskList:
            classification[mask] = code

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
                        raise LiDARArrayColumnError(msg)
                
                # have to do a copy to avoid numpy warning
                # that updating returned array will break in future
                # numpy release.
                array = array[colNames].copy()
            
        return array

class LiDARFileInfo(basedriver.FileInfo):
    """
    Info for a Lidar file
    """
    def __init__(self, fname):
        basedriver.FileInfo.__init__(self, fname)

    @staticmethod
    def getDriverName():
        """
        Return name of driver - just a short unique name is fine.
        should match the :func:`pylidar.lidarformats.generic.LiDARFile.getDriverName`
        call for the same format.
        """
        raise NotImplementedError()

    @staticmethod
    def getHeaderTranslationDict():
        """
        Return a dictionary keyed on HEADER_* values (above)
        that can be used to translate dictionary field names between the formats
        """
        raise NotImplementedError()
        
def getWriterForLiDARFormat(driverName, fname, mode, controls, userClass):
    """
    Given a driverName returns an instance of the given driver class
    Raises LiDARFormatDriverNotFound if not found
    """
    for cls in LiDARFile.__subclasses__():
        if cls.getDriverName() == driverName:
            # create it
            inst = cls(fname, mode, controls, userClass)
            return inst
    # none matched
    msg = 'Cannot find LiDAR driver %s' % driverName
    raise LiDARFormatDriverNotFound(msg)
            

def getReaderForLiDARFile(fname, mode, controls, userClass, verbose=False):
    """
    Returns an instance of a LiDAR format
    reader/writer or raises an exception if none
    found for the file.
    """
    # try each subclass
    for cls in LiDARFile.__subclasses__():
        try:
            # attempt to create it
            inst = cls(fname, mode, controls, userClass)
            # worked - return it
            if verbose:
                print('Succeeded using class', cls)
            return inst
        except LiDARFileException:
            # failed - onto the next one
            if verbose:
                print('Failed using', cls, e)
    # none worked
    msg = 'Cannot open LiDAR file %s' % fname
    raise LiDARFormatDriverNotFound(msg)

def getLidarFileInfo(fname, verbose=False):
    """
    Returns an instance of a LiDAR format info class.
    Or raises an exception if none found for the file.
    """
    for cls in LiDARFileInfo.__subclasses__():
        try:
            inst = cls(fname)
            if verbose:
                print('Succeeded using class', cls)
            return inst
        except LiDARFileException as e:
            # failed - onto the next one
            if verbose:
                print('Failed using', cls, e)
    # none worked
    msg = 'Cannot open LiDAR file %s' % fname
    raise LiDARFormatDriverNotFound(msg)
                                            

