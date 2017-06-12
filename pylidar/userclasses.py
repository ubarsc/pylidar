
"""
Classes that are passed to the user's function
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

import copy
import numpy
from numba import jit
from .lidarformats import generic
from .toolbox import arrayutils

@jit
def stratify3DArrayByValueIdx(inValues, inValuesMask, outIdxs_row, outIdxs_col, outIdxs_p, 
        outIdxsMask, outIdxsCount, 
        bins, counting):
    """
    Note: This function is no longer called, but kept in case it is needed for building 
    a spatial index in future. The stratify3DArrayByValue call below is currently used by
    rebinPtsByHeight(). 
    
    Creates indexes for building a 4d (points, height bin, row, col) point array from the 3d
    (point, row, col) array returned by getPointsByBins() function.
    
    Parameters:
    
    * inValues         3d (ragged) array of values to stratify on (e.g. height)  (nPts, nrows, ncols)
    * inValuesMask     Mask for inValues.
    * outIndxs_row     4d array of row coord of stratified values (nPtsPerHgtBin, nBins, nrows, ncols)
    * outIdxs_col      4d array of col coord of stratified values (nPtsPerHgtBin, nBins, nrows, ncols)
    * outIdxs_p        4d array of p coord (nPtsPerHgtBin, nBins, nrows, ncols)
    * outIdxsMask      4d bool array - True for unused elements (nPtsPerHgtBin, nBins, nrows, ncols)
    * outIdxsCount     3d int array of counts per bin (nBins, rows, ncols) (initialized to zero, always)
    * bins             1d array of height bins. Includes end points, i.e. the number of height bins is
                     (len(bins) - 1). A point is in i-th bin when bin[i] <= z < bins[i+1]. Assumes
                     no points are outside the range of bin values given. 
    * counting         bool flag. If True, then we are just counting, and filling in outIdxsCount,
                         otherwise we are filling in outIdxs_* arrays, too. 
                         
    Returns:
        Nothing
    
    Usage: Call first with counting=True, then find outIdxsCount.max(), use this as nPtsPerHgtBin
    to create other out arrays. Then zero outIdxsCount again, and call again with counting=False. 
    
    """
    (nPts, nRows, nCols) = inValues.shape
    nBins = bins.shape[0] - 1 # because they are bounds
    for r in range(nRows):
        for c in range(nCols):
            for p in range(nPts):
                if not inValuesMask[p, r, c]: # because masked arrays are False where not masked
                    v = inValues[p, r, c]
                    for b in range(nBins):
                        if v >= bins[b] and v < bins[b+1]: # in this bin?
                            if not counting:
                                # only do these steps when running for real
                                j = outIdxsCount[b, r, c]
                                outIdxs_row[j, b, r, c] = r
                                outIdxs_col[j, b, r, c] = c
                                outIdxs_p[j, b, r, c] = p
                                outIdxsMask[j, b, r, c] = False
                            # always update the counts
                            outIdxsCount[b, r, c] += 1

@jit
def stratify3DArrayByValue(inValues, inValuesMask, rebinnedByHeight, 
        rebinnedByHeight_mask, outIdxsCount, 
        bins, counting, heightValues):
    """
    Called by rebinPtsByHeight().
    
    Creates a new 4d (points, height bin, row, col) array if the 3d 
    (points, row, col) data in inValues.
    
    Parameters:
    
    * inValues         3d (ragged) structured array of values to stratify  (nPts, nrows, ncols)
    * inValuesMask     Mask for inValues - note this must be a plain bool array not just inValues.mask
      since inValues.mask will have a bool per record. Best to pass mask for a single
      record - ie heightValues.mask.
    * rebinnedByHeight 4d Output array (nPtsPerHgtBin, nBins, nrows, ncols)
    * rebinnedByHeight_mask 4d output mask for creating ragged array (should be inited to True).
    * outIdxsCount     3d int array of counts per bin (nBins, rows, ncols) (initialized to zero, always)
    * bins             1d array of height bins. Includes end points, i.e. the number of height bins is
      (len(bins) - 1). A point is in i-th bin when bin[i] <= z < bins[i+1]. Assumes
      no points are outside the range of bin values given. 
    * counting         bool flag. If True, then we are just counting, and filling in outIdxsCount,
      otherwise we are filling in inValues* arrays, too. 
    * heightValues    3d (ragged) array of a single value from inValues to stratify on ie inValues['Z'].
                         
    Returns:

    * Nothing
    
    Usage: Call first with counting=True, then find outIdxsCount.max(), use this as nPtsPerHgtBin
    to create other out arrays. Then zero outIdxsCount again, and call again with counting=False. 
    
    """
    (nPts, nRows, nCols) = inValues.shape
    nBins = bins.shape[0] - 1 # because they are bounds
    for r in range(nRows):
        for c in range(nCols):
            for p in range(nPts):
                if not inValuesMask[p, r, c]: # because masked arrays are False where not masked
                    v = heightValues[p, r, c]
                    for b in range(nBins):
                        if v >= bins[b] and v < bins[b+1]: # in this bin?
                            if not counting:
                                # only do these steps when running for real
                                j = outIdxsCount[b, r, c]
                                rebinnedByHeight[j, b, r, c] = inValues[p, r, c]
                                rebinnedByHeight_mask[j, b, r, c] = False
                            # always update the counts
                            outIdxsCount[b, r, c] += 1

class UserInfo(object):
    """
    The 'DataContainer' object (below) contains an 'info' field which is
    an instance of this class. The user function can use these methods to
    obtain information on the current processing state and region.
        
    Equivalent to the RIOS 'info' object.
    
    """
    def __init__(self, controls):
        self.pixGrid = None
        self.extent = None # either extent is not None, or range. Not both.
        self.range = None
        # for isFirstBlock() and isLastBlock()
        self.firstBlock = True
        self.lastBlock = False
        # take a copy so the user can't change it
        self.controls = copy.copy(controls)
        
    def isFirstBlock(self):
        """
        Returns True if this is the first block to be processed
        """
        return self.firstBlock
        
    def isLastBlock(self):
        """
        Returns True if this is the last block to be processed
        """
        return self.lastBlock
        
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
        
    def getControls(self):
        """
        Return the instance of the controls object used for processing
        
        """
        return self.controls
        
    def getBlockCoordArrays(self):
        """
        Return a tuple of the world coordinates for every pixel
        in the current block. Each array has the same shape as the 
        current block. Return value is a tuple::

            (xBlock, yBlock)

        where the values in xBlock are the X coordinates of the centre
        of each pixel, and similarly for yBlock. 
                                                    
        The coordinates returned are for the pixel centres. This is 
        slightly inconsistent with usual GDAL usage, but more likely to
        be what one wants.         
        
        """
        # round() ok since points should already be on the grid, nasty 
        # rounding errors propogated with ceil()                                    
        nRows = int(numpy.round((self.extent.yMax - self.extent.yMin) / self.extent.binSize))
        nCols = int(numpy.round((self.extent.xMax - self.extent.xMin) / self.extent.binSize))
        # add overlap 
        nRows += (self.controls.overlap * 2)
        nCols += (self.controls.overlap * 2)
        # create the indices
        (rowNdx, colNdx) = numpy.mgrid[0:nRows, 0:nCols]
        xBlock = (self.extent.xMin - self.controls.overlap*self.extent.binSize + 
                    self.extent.binSize/2.0 + colNdx * self.extent.binSize)
        yBlock = (self.extent.yMax + self.controls.overlap*self.extent.binSize - 
                    self.extent.binSize/2.0 - rowNdx * self.extent.binSize)
        return (xBlock, yBlock)        

class DataContainer(object):
    """
    This is a container object used for passing as the first parameter to the 
    user function. It contains a UserInfo object (called 'info') plus instances 
    of LidarData and ImageData (see below). These objects will be named in the 
    same way that the LidarFile and ImageFile were in the DataFiles object 
    that was passed to doProcessing().
    
    """
    def __init__(self, controls):
        self.info = UserInfo(controls)

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
        self.controls = driver.controls
        # for writing
        self.pointsToWrite = None
        self.pulsesToWrite = None
        self.receivedToWrite = None
        self.transmittedToWrite = None
        self.waveformInfoToWrite = None
        
    def translateFieldNames(self, otherLidarData, array, arrayType):
        """
        Translates the field names in an array from another format
        (specified by passing the other data object) for use in writing
        to the format for this driver. The array is passed in and updated
        directly (no copy made). The array is returned.
        
        arrayType is one of the ARRAY_TYPE_* values defined in lidarprocessor.py.
        """
        thisDict = self.driver.getTranslationDict(arrayType)
        otherDict = otherLidarData.driver.getTranslationDict(arrayType)
        
        newNames = []
        for name in array.dtype.names:
            for key in otherDict:
                if otherDict[key] == name:
                    name = thisDict[key]
            newNames.append(name)
            
        # see http://stackoverflow.com/questions/14429992/can-i-rename-fields-in-a-numpy-record-array
        array.dtype.names = newNames
        if isinstance(array, numpy.ma.MaskedArray):
            array.mask.dtype.names = newNames
            
        # just for consistancy
        return array
        
    def getPoints(self, colNames=None):
        """
        Returns the points for the extent/range of the current
        block as a structured array. The fields on this array
        are defined by the driver being used.
        
        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        if self.controls.spatialProcessing:
            points = self.driver.readPointsForExtent(colNames)
        else:
            points = self.driver.readPointsForRange(colNames)
        return points
        
    def getPulses(self, colNames=None, pulseIndex=None):
        """
        Returns the pulses for the extent/range of the current
        block as a structured array. The fields on this array
        are defined by the driver being used.

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        
        pulseIndex is an optional masked 3d array of indices to remap the
        1d pulse array to a 3D point by bin array. pulseIndex is returned from
        getPointsByBins with returnPulseIndex=True.
        """
        if self.controls.spatialProcessing:
            pulses = self.driver.readPulsesForExtent(colNames)
            if pulseIndex is not None:
                pulses = numpy.ma.array(pulses[pulseIndex], mask=pulseIndex.mask)
        else:
            pulses = self.driver.readPulsesForRange(colNames)              
        
        return pulses
        
    def getPulsesByBins(self, extent=None, colNames=None):
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

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        if self.controls.spatialProcessing:
            pulses = self.driver.readPulsesForExtentByBins(extent, colNames)
        else:
            msg = 'Call only valid when doing spatial processing'
            raise generic.LiDARNonSpatialProcessing(msg)
            
        return pulses
        
    def getPointsByBins(self, extent=None, colNames=None, indexByPulse=False,
                returnPulseIndex=False):
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

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        
        Set indexByPulse to True to bin points by the pulse index location rather
        than point location.
        
        Set returnPulseIndex to True to also return a 3 dimensional masked array
        containing the indexes into the 1d array returned by getPulses().
        """
        if self.controls.spatialProcessing:
            points = self.driver.readPointsForExtentByBins(extent, colNames, 
                        indexByPulse, returnPulseIndex)
        else:
            msg = 'Call only valid when doing spatial processing'
            raise generic.LiDARNonSpatialProcessing(msg)

        return points
    
    def rebinPtsByHeight(self, pointsByBin, bins, heightArray=None, heightField='Z'):
        """
        pointsByBin       3d ragged (masked) structured array of points. (nrows, ncols, npts)
        bins              Height bins into which to stratify points
        
        Set heightArray to a masked array of values used to vertically stratify the points. 
        This allows columns not in pointsByBin to be used.
        
        Set heightField to specify which pointsByBin column name to use for height values. 
        Only used if heightArray is None.
        
        Return:
            4d re-binned copy of pointsByBin
            
        """
        (maxpts, nrows, ncols) = pointsByBin.shape
        nbins = len(bins) - 1
        # Set up for first pass
        idxCount = numpy.zeros((nbins, nrows, ncols), dtype=numpy.uint16)
        if heightArray is None:
            heightArray = pointsByBin[heightField]
        
        # numba doesn't support None so create some empty arrays
        # for the outputs we don't need
        idxMask = numpy.ones((1, 1, 1, 1), dtype=numpy.bool)
        rebinnedPts = numpy.empty((1, 1, 1, 1), dtype=pointsByBin.data.dtype)
        
        # this first call we are just working out the sizes by letting
        # it populate idxCount and nothing else
        stratify3DArrayByValue(pointsByBin.data, heightArray.mask, rebinnedPts,
            idxMask, idxCount, bins, True, heightArray.data)
        ptsPerHgtBin = idxCount.max()
        
        # ok now we know the sizes we can create the arrays
        idxMask = numpy.ones((ptsPerHgtBin, nbins, nrows, ncols), dtype=numpy.bool)
        rebinnedPts = numpy.empty((ptsPerHgtBin, nbins, nrows, ncols), dtype=pointsByBin.data.dtype)
        # rezero the counts
        idxCount.fill(0)

        # now we can call the thing for real
        stratify3DArrayByValue(pointsByBin.data, heightArray.mask, rebinnedPts, 
            idxMask, idxCount, bins, False, heightArray.data)
            
        # create a masked array
        rebinnedPtsMasked = numpy.ma.array(rebinnedPts, mask=idxMask)
        return rebinnedPtsMasked
        
    def getPointsByPulse(self, colNames=None):
        """
        Returns the points as a 2d structured masked array. The first axis
        is the same length as the pulse array but the second axis contains the 
        points for each pulse. The mask will be set to True where no valid data
        since some pulses will have more points than others. 

        colNames can be a name or list of column names to return. By default
        all columns are returned.
        """
        return self.driver.readPointsByPulse(colNames)
        
    def getWaveformInfo(self):
        """
        Returns a 2d masked structured array with information about 
        the waveforms. First axis will be the waveform number,
        second will be same length as the pulses
        """
        return self.driver.readWaveformInfo()
        
    def getTransmitted(self):
        """
        Returns a masked 3d radiance array. 
        The first axis is the waveform bins, the second axis will 
        be the waveform number and the third axis axis will be the same length
        as the pulses.
        
        Because some pulses will have a longer waveform than others a masked
        array is returned.
        """
        return self.driver.readTransmitted()
        
    def getReceived(self):
        """
        Returns a masked 3d radiance array. 
        The first axis is the waveform bins, the second axis will 
        be the waveform number and the third axis axis will be the same length
        as the pulses.
        
        Because some pulses will have a longer waveform than others a masked
        array is returned.
        """
        return self.driver.readReceived()

    def getHeaderTranslationDict(self):
        """
        Return a dictionary keyed on HEADER_* values (above)
        that can be used to translate dictionary field names between the formats
        """
        return self.driver.getHeaderTranslationDict()

    def getHeader(self):
        """
        Returns the header as a dictionary of header key/value pairs.
        """
        return self.driver.getHeader()
        
    def setHeader(self, headerDict):
        """
        Sets header values as a dictionary of header key/value pairs.
        """
        self.driver.setHeader(headerDict)
        
    def getHeaderValue(self, name):
        """
        Gets a particular header value with the given name
        """
        return self.driver.getHeaderValue(name)
        
    def setHeaderValue(self, name, value):
        """
        Sets a particular header value with the given name
        """
        self.driver.setHeaderValue(name, value)
        
    def setHeaderValues(self, **kwargs):
        """
        Overloaded version to support key word args instead
        """
        for name in kwargs:
            self.driver.setHeaderValue(name, kwargs[name])
            
    def setScaling(self, colName, arrayType, gain, offset):
        """
        Set the scaling for the given column name
        
        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        """
        self.driver.setScaling(colName, arrayType, gain, offset)
        
    def getScaling(self, colName, arrayType):
        """
        Returns the scaling (gain, offset) for the given column name

        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        """
        return self.driver.getScaling(colName, arrayType)
        
    def setNativeDataType(self, colName, arrayType, dtype):
        """
        Set the native dtype (numpy.int16 etc)that a column is stored
        as internally after scaling (if any) is applied.
        
        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        
        generic.LiDARArrayColumnError is raised if information cannot be found.
        """
        self.driver.setNativeDataType(colName, arrayType, dtype)
        
    def getNativeDataType(self, colName, arrayType):
        """
        Return the native dtype (numpy.int16 etc)that a column is stored
        as internally after scaling (if any) is applied. Provided so scaling
        can be adjusted when translating between formats.
        
        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        
        generic.LiDARArrayColumnError is raised if information cannot be found.
        """
        return self.driver.getNativeDataType(colName, arrayType)

    def setNullValue(self, colName, arrayType, value, scaled=True):
        """
        Set the 'null' value for the given column.

        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants

        By default the value is treated as the scaled value, but this can
        be changed with the 'scaled' parameter.

        generic.LiDARArrayColumnError is raised if this cannot be set for the column.
        """
        self.driver.setNullValue(colName, arrayType, value, scaled)

    def getNullValue(self, colName, arrayType, scaled=True):
        """
        Get the 'null' value for the given column.

        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants

        By default the returned value is scaled, change this with the 'scaled'
        parameter.

        Raises generic.LiDARArrayColumnError if information cannot be
        found for the column.
        """
        return self.driver.getNullValue(colName, arrayType, scaled)    

    def getScalingColumns(self, arrayType):
        """
        Return a list of column names that require scaling to be 
        set on write. 

        arrayType is one of the lidarprocessor.ARRAY_TYPE_* constants
        """
        return self.driver.getScalingColumns(arrayType)
        
    def setWaveformInfo(self, info, colName=None):
        """
        Set the waveform info as a masked 2d array.

        If passed a structured array, the same
        field names are expected as those read with the same driver.
        
        If the array is non-structured (ie you passed a colNames as a string
        to getWaveformInfo()) you need to pass the same string as the colName 
        parameter.

        """
        self.waveformInfoToWrite = self.convertToStructIfNeeded(info, colName,
                                        self.waveformInfoToWrite)

    def setTransmitted(self, transmitted):
        """
        Set the transmitted waveform for each pulse as 
        a masked 3d integer array.
        """
        self.transmittedToWrite = transmitted
        
    def setReceived(self, received):
        """
        Set the received waveform for each pulse as 
        a masked 3d integer array.
        """
        self.receivedToWrite = received

    @staticmethod
    def convertToStructIfNeeded(data, colName, oldData=None):
        """
        Converts data to a structured array if it is not.
        If conversion is required it uses colName and data type of data.
        Raises exception if conversion not possible or does not make sense.

        if oldData is not None and data is non-structured, then the new
        data is appended onto oldData and returned. If data is structured,
        and error is raised.
        """
        isStruct = data.dtype.names is not None
        isMasked = isinstance(data, numpy.ma.MaskedArray)
        if isStruct and colName is not None:
            msg = 'if using structured arrays, leave colName as None'
            raise generic.LiDARArrayColumnError(msg)
        if not isStruct and colName is None:
            msg = 'if not using structured arrays, pass colName'
            raise generic.LiDARArrayColumnError(msg)

        if isStruct:
            if oldData is not None:
                msg = 'This data type has already been set for this block'
                raise generic.LiDARInvalidData(msg)

        else:
            if oldData is not None:
                # append new data to oldData
                data = arrayutils.addFieldToStructArray(oldData, colName, 
                        data.dtype, data)
            else:
                # turn back into structured array so keep consistent internally
                structdata = numpy.empty(data.shape, 
                                        dtype=[(colName, data.dtype)])
                structdata[colName] = data
                if isMasked:
                    structdata = numpy.ma.MaskedArray(structdata, mask=data.mask)

                data = structdata

        return data
        
    def setPoints(self, points, colName=None):
        """
        Write the points to a file. If passed a structured array, the same
        field names are expected as those read with the same driver.
        
        If the array is non-structured (ie you passed a colNames as a string
        to getPoints()) you need to pass the same string as the colName 
        parameter.

        Pass either a 1d array (like that read from getPoints()) or a
        3d masked array (like that read from getPointsByBins()).

        """
        self.pointsToWrite = self.convertToStructIfNeeded(points, colName,
                                    self.pointsToWrite)
            
    def setPulses(self, pulses, colName=None):
        """
        Write the pulses to a file. If passed a structured array, the same
        field names are expected as those read with the same driver.
        
        If the array is non-structured (ie you passed a colNames as a string
        to getPulses()) you need to pass the same string as the colName 
        parameter.

        Pass either a 1d array (like that read from getPulses()) or a
        3d masked array (like that read from getPulsesByBins()).

        """
        self.pulsesToWrite = self.convertToStructIfNeeded(pulses, colName,
                                    self.pulsesToWrite)
        
    def flush(self):
        """
        writes data to file set via the set*() functions
        """
        self.driver.writeData(self.pulsesToWrite, self.pointsToWrite, 
            self.transmittedToWrite, self.receivedToWrite, 
            self.waveformInfoToWrite)
        # reset for next time
        self.pointsToWrite = None
        self.pulsesToWrite = None
        self.receivedToWrite = None
        self.transmittedToWrite = None
        self.waveformInfoToWrite = None
        
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
        self.data = None
        
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
        self.data = data
        
    def flush(self):
        """
        Now actually do the write
        """
        self.driver.setData(self.data)
        self.data = None
        
