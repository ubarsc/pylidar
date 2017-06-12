
"""
Driver for ASCII files. Currently uncompressed files with a .dat
or .csv extension are supported. If the ZLIB_ROOT environment 
variable was set at build time, then gzip compressed files with a
.gz extension are also supported.

The user must specify the column names and types
via the COL_TYPES driver option.
For time-sequential files, the PULSE_COLS driver option needs to be
provided so the points can be grouped into pulses.
When PULSE_COLS isn't provided then the file is assumed to be 
non-time-sequential and a pulse is created for each point.

Driver Options
--------------

These are contained in the SUPPORTEDOPTIONS module level variable.

+-----------------------+-------------------------------------------+
| Name                  | Use                                       |
+=======================+===========================================+
| COL_TYPES             | A numpy style list of tuples defining     |
|                       | the data types of each column. Each tuple |
|                       | should have a name and a numpy dtype.     |
+-----------------------+-------------------------------------------+
| PULSE_COLS            | A list of fields which define the pulses. |
|                       | The values in the these columns will be   |
|                       | matched and where equal will be put into  |
|                       | one pulse and the other values into the   |
|                       | points for that pulse.                    |
+-----------------------+-------------------------------------------+
| CLASSIFICATION_CODES  | A list of tuples to translate the codes   |
|                       | used within the file to the               |
|                       | lidarprocessor.CLASSIFICATION_* ones.     |
|                       | Each tuple should have the internalCode   |
|                       | first, then the lidarprocessor code       |
|                       | Codes without a translation will be       |
|                       | copied through without change.            |
+-----------------------+-------------------------------------------+
| COMMENT_CHAR          | A single character that defines what is   |
|                       | used in the file to denote comments.      |
|                       | Lines that start with this character are  |
|                       | ignored. Defaults to '#'                  |
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
import sys
import gzip
import copy
import numpy

from . import generic
from . import gridindexutils

# Fail slightly less drastically when running from ReadTheDocs
if os.getenv('READTHEDOCS', default='False') != 'True':
    from . import _ascii
    HAVE_ZLIB = _ascii.HAVE_ZLIB
else:
    HAVE_ZLIB = False

SUPPORTEDOPTIONS = ('COL_TYPES', 'PULSE_COLS', 'CLASSIFICATION_CODES', 'COMMENT_CHAR')
"driver options"
COMPULSARYOPTIONS = ('COL_TYPES',)
"necessary driver options"

class ASCIIFile(generic.LiDARFile):
    """
    Driver for reading ASCII files. Uses the underlying _ascii C++ module.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)
        
        if mode != generic.READ:
            msg = 'ASCII driver is read only'
            raise generic.LiDARInvalidSetting(msg)

        try:
            self.typeCode = _ascii.getFileType(fname)
        except _ascii.error as e:
            msg = 'cannot open as ASCII file' + str(e)
            raise generic.LiDARFileException(msg)

        # check if the options are all valid. Good to check in case of typo.
        # hard to do this in C
        for key in userClass.lidarDriverOptions:
            if key not in SUPPORTEDOPTIONS:
                msg = '%s not a supported ASCII option' % repr(key)
                raise generic.LiDARInvalidSetting(msg)

        # also check they are all present - we can't read the file otherwise
        for key in COMPULSARYOPTIONS:
            if key not in userClass.lidarDriverOptions:
                msg = 'must provide %s driver option' % key
                raise generic.LiDARInvalidSetting(msg)


        self.pointDTypes = []
        self.pulseDTypes = []
        idx = 0
        pulseCols = []
        if 'PULSE_COLS' in userClass.lidarDriverOptions:
            pulseCols = userClass.lidarDriverOptions['PULSE_COLS']

        for name, dtype in userClass.lidarDriverOptions['COL_TYPES']:
            if name == "NUMBER_OF_RETURNS" or name == "PTS_START_IDX":
                msg = ("Can't use fields NUMBER_OF_RETURNS or PTS_START_IDX " +
                        "since they are generated")
                raise generic.LiDARInvalidSetting(msg)

            if name in pulseCols:
                self.pulseDTypes.append((name, dtype, idx))
            else:
                self.pointDTypes.append((name, dtype, idx))

            idx += 1
            

        bTimeSequential = len(self.pulseDTypes) > 0

        commentChar = '#'
        if 'COMMENT_CHAR' in userClass.lidarDriverOptions:
            commentChar = userClass.lidarDriverOptions['COMMENT_CHAR']

        # create reader
        self.reader = _ascii.Reader(fname, self.typeCode, self.pulseDTypes, 
                            self.pointDTypes, bTimeSequential, commentChar)

        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None

        # set our translation codes
        if 'CLASSIFICATION_CODES' in userClass.lidarDriverOptions:
            codes = userClass.lidarDriverOptions['CLASSIFICATION_CODES']
            for trans in codes:
                self.classificationTranslation.append(trans)
            
    @staticmethod        
    def getDriverName():
        return 'ASCII'

    def close(self):
        self.reader = None
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None

    def hasSpatialIndex(self):
        """
        ASCII files aren't spatially indexed
        """
        return False

    @staticmethod
    def getHeaderTranslationDict():
        """
        No header so not really supported. Return empty dict.
        """
        return {}

    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        """
        self.range = copy.copy(pulseRange)
        # return True if we can still read data
        # we just assume we can until we find out
        # after a read that we can't
        return not self.reader.finished

    def readData(self):
        """
        Internal method. Reads all the points and pulses
        for the current pulse range.
        """

        pulses, points = self.reader.readData(self.range.startPulse,
                            self.range.endPulse)

        # translate any classifications
        self.recodeClassification(points, generic.RECODE_TO_LAS)

        return pulses, points


    def readPointsForRange(self, colNames=None):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        if self.lastRange is None or self.range != self.lastRange:
            pulses, points = self.readData()
            self.lastRange = self.range        
            self.lastPoints = points
            self.lastPulses = pulses

        return self.subsetColumns(self.lastPoints, colNames)
        
    def readPulsesForRange(self, colNames=None):
        """
        Reads the pulses for the current range. Returns a 1d array.

        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        if self.lastRange is None or self.range != self.lastRange:
            pulses, points = self.readData()
            self.lastRange = self.range        
            self.lastPoints = points
            self.lastPulses = pulses
                            
        return self.subsetColumns(self.lastPulses, colNames)

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

    def getTotalNumberPulses(self):
        """
        No idea how to find this out...
        """
        raise generic.LiDARFunctionUnsupported()

    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        This driver does not support writing so ignore if reading,
        throw and error otherwise.
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return
        
        msg = 'ASCII driver does not support update/creating'
        raise generic.LiDARWritingNotSupported(msg)

    def getHeader(self):
        """
        ASCII files have no header
        """
        format = _ascii.FORMAT_NAMES[self.typeCode]
        return {'format' : format}
        
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.getHeader()[name]

    def readTransmitted(self):
        """
        ASCII (AFAIK) doesn't support transmitted
        """
        return None
        
    def readReceived(self):
        """
        ASCII (AFAIK) doesn't support received
        """
        return None

    def readWaveformInfo(self):
        """
        ASCII (AFAIK) doesn't support waveforms
        """
        return None

class ASCIIFileInfo(generic.LiDARFileInfo):
    """
    Class that gets information about a .las file
    and makes it available as fields.
    """
    def __init__(self, fname):
        generic.LiDARFileInfo.__init__(self, fname)

        try:
            typeCode = _ascii.getFileType(fname)
        except _ascii.error as e:
            msg = 'cannot open as ASCII file' + str(e)
            raise generic.LiDARFileException(msg)

        self.typeCode = typeCode
        self.format = _ascii.FORMAT_NAMES[typeCode]

        # I don't think there is any information we can add here??

    @staticmethod        
    def getDriverName():
        return 'ASCII'

    @staticmethod
    def getHeaderTranslationDict():
        """
        No header so not really supported. Return empty dict.
        """
        return {}
