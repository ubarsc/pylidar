
"""
Generic 'driver' class. To be subclassed by both 
LiDAR and raster drivers.

Also contains the Extent class which defines the extent
to use for reading or writing the current block.
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

class Extent(object):
    """
    Class that defines the extent in world coords
    of an area to read or write
    """
    def __init__(self, xMin, xMax, yMin, yMax, binSize):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.binSize = binSize
        
    def __eq__(self, other):
        return (self.xMin == other.xMin and self.xMax == other.xMax and
            self.yMin == other.yMin and self.yMax == other.yMax and
            self.binSize == other.binSize)
            
    def __ne__(self, other):
        return (self.xMin != other.xMin or self.xMax != other.xMax or
            self.yMin != other.yMin or self.yMax != other.yMax or
            self.binSize != other.binSize)
        
    def __str__(self):
        s = "xMin:%s,xMax:%s,yMin:%s,yMax:%s,binSize:%s" % (self.xMin, self.xMax,
                      self.yMin, self.yMax, self.binSize)
        return s


class Driver(object):
    """
    Base Driver object to be subclassed be both the LiDAR and raster drivers
    """
    def __init__(self, fname, mode, controls, userClass):
        """
        fname is the file to open or create
        mode is READ, UPDATE or CREATE
        controls is an instance of lidarprocessing.Controls
        userClass is the instance of userclass.LidarFile or userclass.ImageFile
            used to define the file.
        """
        self.fname = fname 
        self.mode = mode
        self.controls = controls
        self.userClass = userClass
        
    def setExtent(self, extent):
        """
        Set the extent for reading or writing
        """
        raise NotImplementedError()
        
    def getPixelGrid(self):
        """
        Return the PixelGridDefn for this file
        """
        raise NotImplementedError()
        
    def setPixelGrid(self, pixGrid):
        """
        Set the PixelGridDefn for the reading or 
        writing we will do
        """
        raise NotImplementedError()
        
    def close(self):
        """
        Close all open file handles
        """
        raise NotImplementedError()
        
        
