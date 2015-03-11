
"""
Base class for LiDAR format reader/writers
"""
# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Neil Flood and Sam Gillingham
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

from .. import basedriver

# access modes passed to driver constructor
READ = 0
UPDATE = 1
CREATE = 2

# to be passed to message handler function 
# controls.messageHandler
MESSAGE_WARNING = 0
MESSAGE_INFORMATION = 1
MESSAGE_DEBUG = 2

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
        
    def getDriverName(self):
        """
        Return name of driver - just a short unique name is fine.
        """
        raise NotImplementedError()
        
    def readPointsForExtent(self):
        """
        Read all the points within the given extent
        as 1d structured array. The names of the fields in this array
        will be defined by the driver.
        """
        raise NotImplementedError()
        
    def readPulsesForExtent(self):
        """
        Read all the pulses within the given extent
        as 1d structured array. The names of the fields in this array
        will be defined by the driver.
        """
        raise NotImplementedError()
        
    def readPulsesForExtentByBins(extent=None):
        """
        Read all the pulses within the given extent as a 3d structured 
        masked array to match the block/bins being used.
        
        The extent/binning for the read data can be overriden by passing in a
        Extent instance.
        """
        raise NotImplementedError()
        
    def readPointsForExtentByBins(extent=None):
        """
        Read all the points within the given extent as a 3d structured 
        masked array to match the block/bins being used.
        
        The extent/binning for the read data can be overriden by passing in a
        Extent instance.
        """
        raise NotImplementedError()
        
    def readPointsByPulse(self):     
        """
        Read a 2d structured masked array containing the points
        for each pulse.
        """
        raise NotImplementedError()
        
    def readTransmitted(self):
        """
        Read the transmitted waveform for all pulses
        returns a 2d masked array. 
        """
        raise NotImplementedError()
        
    def readReceived(self):
        """
        Read the received waveform for all pulses
        returns a 2d masked array
        """
        raise NotImplementedError()

    def writeTransmitted(self, transmitted):
        """
        Write the transmitted waveform for all pulses
        as a 2d masked array
        """
        raise NotImplementedError()
        
    def writeReceived(self, received):
        """
        Write the received waveform for all pulses
        as a 2d masked array
        """
        raise NotImplementedError()
        
    def writePointsForExtent(self, points):
        """
        Write the points for the current extent. Can either be
        1d structured array (like that returned by readPointsForExtent())
        or a 3d masked array (like that returned by readPointsByBins())
        """
        raise NotImplementedError()
        
    def writePulsesForExtent(self, pulses):
        """
        Write the pulses for the current extent. Can either be
        1d structured array (like that returned by readPulsesForExtent())
        or a 3d masked array (like that returned by readPulsessByBins())
        """
        raise NotImplementedError()
        
    def hasSpatialIndex(self):
        """
        Returns True if file has a spatial index defined
        """
        raise NotImplementedError()
        
    # see below for no spatial index
    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        
        Return False if outside the range of data.
        """
        raise NotImplementedError()
    
    def readPointsForRange(self):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.
        """
        raise NotImplementedError()
        
    def readPulsesForRange(self):
        """
        Reads the pulses for the current range. Returns a 1d array.

        Returns an empty array if range is outside of the current file.
        """
        raise NotImplementedError()
        
    def getTotalNumberPulses(self):
        """
        Returns the total number of pulses in this file. Used for progress.
        
        Raise a LiDARFunctionUnsupported error if driver does not support
        easily finding the total number of pulses.
        """
        raise NotImplementedError()
        
    def writePointsForRange(self, points):
        """
        Write the points for the current extent. Must be
        1d structured array like that returned by readPointsForRange().
        """
        raise NotImplementedError()
        
    def writePulsesForRange(self, pulses):
        """
        Write the pulses for the current extent. Must be
        1d structured array like that returned by readPulsesForRange().
        """
        raise NotImplementedError()
        
    def close(self):
        """
        Write any updated spatial index and close any file handles.
        """
        raise NotImplementedError()


def getReaderForLiDARFile(fname, mode, controls, userClass):
    """
    Returns an instance of a LiDAR format
    reader/writer or raises an exception if none
    found for the file.
    """
    # try each subclass
    for cls in LiDARFile.__subclasses__():
        #print('trying', cls)
        try:
            # attempt to create it
            inst = cls(fname, mode, controls, userClass)
            # worked - return it
            return inst
        except LiDARFileException:
            # failed - onto the next one
            pass
    # none worked
    msg = 'Cannot open LiDAR file %s' % fname
    raise LiDARFormatDriverNotFound(msg)

