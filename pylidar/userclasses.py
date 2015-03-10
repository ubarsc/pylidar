
"""
Classes that are passed to the user's function
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

from __future__ import print_function, division

import copy
from .lidarformats import generic

class UserInfo(object):
    """
    The 'DataContainer' object (below) contains an 'info' field which is
    an instance of this class. The user function can use these methods to
    obtain information on the current processing state and region.
        
    Equivalent to the RIOS 'info' object.
    
    """
    def __init__(self):
        self.pixGrid = None
        self.extent = None # either extent is not None, or range. Not both.
        self.range = None
        
    def setPixGrid(self, pixGrid):
        """
        For internal use. Used by the processor to set the current state.
        """
        # take a copy so the user can't change it
        self.pixGrid = copy.copy(pixGrid)
        
    def getPixGrid(self):
        """
        Return the current pixgrid. This defines the current total
        processing extent, resolution and projection. 
        
        Is an instance of rios.pixelgrid.PixelGridDefn.
        """
        return self.pixGrid
        
    def setExtent(self, extent):
        """
        For internal use. Used by the processor to set the current state.
        """
        # take a copy so the user can't change it
        self.extent = copy.copy(extent)
        
    def getExtent(self):
        """
        Get the extent of the current block being procesed. This is only
        valid when spatial processing is enabled. Otherwise use getRange()
        
        This is an instance of .basedriver.Extent.
        """
        return self.extent
        
    def setRange(self, range):
        """
        For internal use. Used by the processor to set the current state.
        """
        # take a copy so the user can't change it
        self.range = copy.copy(range)
        
    def getRange(self):
        """
        Get the range of pulses being processed. This is only vaid when 
        spatial processing is disabled. When doing spatial processing, use
        getExtent().
        """
        return self.range

class DataContainer(object):
    """
    This is a container object used for passing as the first parameter to the 
    user function. It contains a UserInfo object (called 'info') plus instances 
    of LidarData and ImageData (see below). These objects will be named in the 
    same way that the LidarFile and ImageFile were in the DataFiles object 
    that was passed to doProcessing().
    
    """
    def __init__(self):
        self.info = UserInfo()

class LidarData(object):
    """
    Class that allows reading and writing to/from a LiDAR file. Passed to the 
    user function from a field on the DataContainer object.
    
    Calls though to the driver instance it was constructed with to do the 
    actual work.
    
    """
    def __init__(self, mode, driver):
        self.mode = mode
        self.driver = driver
        self.extent = None
        self.spatialProcessing = driver.controls.spatialProcessing
        
    def getPoints(self):
        """
        Returns the points for the extent/range of the current
        block as a structured array. The fields on this array
        are defined by the driver being used.
        """
        if self.spatialProcessing:
            points = self.driver.readPointsForExtent()
        else:
            points = self.driver.readPointsForRange()
        return points
        
    def getPulses(self):
        """
        Returns the pulses for the extent/range of the current
        block as a structured array. The fields on this array
        are defined by the driver being used.
        """
        if self.spatialProcessing:
            pulses = self.driver.readPulsesForExtent()
        else:
            pulses = self.driver.readPulsesForRange()
        return pulses
        
    def getPulsesByBins(self, extent=None):
        """
        Returns the pulses for the extent of the current block
        as a 3 dimensional structured masked array. Only valid for spatial 
        processing. The fields on this array are defined by the driver being 
        used.
        
        First axis is the pulses in each bin, second axis is the 
        rows, third is the columns. 
        
        Some bins have more pulses that others so the mask is set to True 
        when data not valid.
        
        The extent/binning for the read data can be overriden by passing in a
        basedriver.Extent instance.
        """
        if self.spatialProcessing:
            pulses = self.driver.readPulsesForExtentByBins(extent)
        else:
            msg = 'Call only valid when doing spatial processing'
            raise generic.LiDARNonSpatialProcessing(msg)
            
        return pulses
        
    def getPointsByBins(self, extent=None):
        """
        Returns the points for the extent of the current block
        as a 3 dimensional structured masked array. Only valid for spatial 
        processing. The fields on this array are defined by the driver being 
        used.
        
        First axis is the points in each bin, second axis is the 
        rows, third is the columns. 
        
        Some bins have more points that others so the mask is set to True 
        when data not valid.
        
        The extent/binning for the read data can be overriden by passing in a
        basedriver.Extent instance.
        """
        if self.spatialProcessing:
            points = self.driver.readPointsForExtentByBins(extent)
        else:
            msg = 'Call only valid when doing spatial processing'
            raise generic.LiDARNonSpatialProcessing(msg)

        return points        

    def getPointsByPulse(self):
        """
        Returns the points as a 2d structured masked array. The first axis
        is the same length as the pulse array but the second axis contains the 
        points for each pulse. The mask will be set to True where no valid data
        since some pulses will have more points than others. 
        """
        return self.driver.readPointsByPulse()
        
    def getTransmitted(self):
        """
        Returns a masked 2d integer array. The first axis will be the same
        length as the pulses. The second axis will contain the transmitted 
        waveform.
        
        Because some pulses will have a longer waveform than others a masked
        array is returned.
        """
        return self.driver.readTransmitted()
        
    def getReceived(self):
        """
        Returns a masked 2d integer array. The first axis will be the same
        length as the pulses. The second axis will contain the received
        waveform.
        
        Because some pulses will have a longer waveform than others a masked
        array is returned.
        """
        return self.driver.readReceived()

    def setTransmitted(self, transmitted):
        """
        Set the transmitted waveform for each pulse as 
        a masked 2d integer array.
        """
        return self.driver.writeTransmitted(transmitted)
        
    def setReceived(self, received):
        """
        Set the received waveform for each pulse as 
        a masked 2d integer array.
        """
        return self.driver.writeReceived(received)
        
    def setPoints(self, points):
        """
        Write the points to a file as a structured array. The same
        field names are expected as those read with the same driver.
        
        Pass either a 1d array (like that read from getPoints()) or a
        3d masked array (like that read from getPointsByBins()).
        """
        if self.spatialProcessing:
            self.driver.writePointsForExtent(points)
        else:
            self.driver.writePointsForRange(points)
            
    def setPulses(self, pulses):
        """
        Write the pulses to a file as a structured array. The same
        field names are expected as those read with the same driver.
        
        Pass either a 1d array (like that read from getPulses()) or a
        3d masked array (like that read from getPulsesByBins()).
        """
        if self.spatialProcessing:
            self.driver.writePulsesForExtent(pulses)
        else:
            self.driver.writePulsesForRange(pulses)
        
class ImageData(object):
    """
    Class that allows reading and writing to/from an image file. Passed to the 
    user function from a field on the DataContainer object.

    Calls though to the driver instance it was constructed with to do the 
    actual work.
    """
    def __init__(self, mode, driver):
        self.mode = mode
        self.driver = driver
        
    def getData(self):
        """
        Returns the data for the current extent as a 3d numpy array in the 
        same data type as the image file.
        """
        return self.driver.getData()
        
    def setData(self, data):
        """
        Sets the image data for the current extent. The data type of the passed 
        in numpy array will be the data type for the newly created file.
        """
        self.driver.setData(data)
