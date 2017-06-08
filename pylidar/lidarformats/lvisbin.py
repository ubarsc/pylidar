"""
Driver for LVIS Binary files

Read Driver Options
-------------------

These are contained in the READSUPPORTEDOPTIONS module level variable.

+-----------------------+--------------------------------------------+
| Name                  | Use                                        |
+=======================+============================================+
| POINT_FROM            | an integer. Set to one of the POINT_FROM_* |
|                       | module level constants. Determines which   |
|                       | file the coordinates for the point is      |
|                       | created from. Defaults to POINT_FROM_LCE   |
+-----------------------+--------------------------------------------+
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

from . import generic
# Fail slightly less drastically when running from ReadTheDocs
if os.getenv('READTHEDOCS', default='False') != 'True':
    from . import _lvisbin
    # bring constants over
    POINT_FROM_LCE = _lvisbin.POINT_FROM_LCE
    POINT_FROM_LGE = _lvisbin.POINT_FROM_LGE
    POINT_FROM_LGW0 = _lvisbin.POINT_FROM_LGW0
    POINT_FROM_LGWEND = _lvisbin.POINT_FROM_LGWEND
else:
    POINT_FROM_LCE = None
    POINT_FROM_LGE = None
    POINT_FROM_LGW0 = None
    POINT_FROM_LGWEND = None
    "How the pulses are found"

READSUPPORTEDOPTIONS = ('POINT_FROM',)
"Supported read options"

def translateChars(input, old, new):
    """
    Translate any instances of old into new in string input.
    Assumes old and new are lowercase. Checks also for uppercase
    old and replaces with uppercase new.

    Use this to replace chars in the file extension while preserving the case.
    """
    output = input.replace(old, new)
    output = output.replace(old.upper(), new.upper())
    return output

def getFilenames(fname):
    """
    Given a filename, determines if it is one of the .lce, .lcw, .lgw 
    files and determines the other ones.
    Returns name of lce, lcw and lgw
    """
    root, ext = os.path.splitext(fname)
    ext_lwr = ext.lower()

    lcename = None
    lcwname = None
    lgwname = None

    # just do checks based on extension, not sure if we should
    # be calling detect_release_version() in the C++ code here...
    if ext_lwr == '.lce':
        lcename = fname
        lcwext = translateChars(ext, 'e', 'w')
        lcwname = root + lcwext
        if not os.path.exists(lcwname):
            lcwname = None
            
        lgwext = translateChars(lcwext, 'c', 'g')
        lgwname = root + lgwext
        if not os.path.exists(lgwname):
            lgwname = None
            
    elif ext_lwr == '.lcw':
        lcwname = fname
        lceext = translateChars(ext, 'w', 'e')
        lcename = root + lceext
        if not os.path.exists(lcename):
            lcename = None
            
        lgwext = translateChars(ext, 'c', 'g')
        lgwname = root + lgwext
        if not os.path.exists(lgwname):
            lgwname = None
            
    elif ext_lwr == '.lgw':
        lgwname = fname
        lcwext = translateChars(ext, 'g', 'c')
        lcename = root + lcwext
        if not os.path.exists(lcwname):
            lcwname = None
            
        lceext = translateChars(lcwext, 'w', 'e')
        lcename = root + lceext
        if not os.path.exists(lcename):
            lcename = None
            
    else:
        msg = 'not a lvis binary file'
        raise generic.LiDARFormatNotUnderstood(msg)
    
    return lcename, lcwname, lgwname

class LVISBinFile(generic.LiDARFile):
    """
    Reader for LVIS Binary files
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)    

        if mode != generic.READ:
            msg = 'LVIS Binary driver is read only'
            raise generic.LiDARInvalidSetting(msg)

        for key in userClass.lidarDriverOptions:
            if key not in READSUPPORTEDOPTIONS:
                msg = '%s not a supported lvis option' % repr(key)
                raise generic.LiDARInvalidSetting(msg)

        lcename, lcwname, lgwname = getFilenames(fname)
        print(lcename, lcwname, lgwname)

        point_from = POINT_FROM_LCE
        if 'POINT_FROM' in userClass.lidarDriverOptions:
            point_from = userClass.lidarDriverOptions['POINT_FROM']

        self.lvisFile = _lvisbin.File(lcename, lcwname, lgwname, point_from)
        self.range = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveformInfo = None
        self.lastReceived = None

    @staticmethod        
    def getDriverName():
        return 'LVIS Binary'

    def close(self):
        self.lvisFile = None
        self.range = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveformInfo = None
        self.lastReceived = None

    def readPointsByPulse(self, colNames=None):
        pass

    def hasSpatialIndex(self):
        "LVIS does not have a spatial index"
        return False

    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        """
        self.range = copy.copy(pulseRange)
        # return True if we can still read data
        # we just assume we can until we find out
        # after a read that we can't
        return not self.lvisFile.finished
    
    def readData(self, extent=None):
        """
        Internal method. Just reads into the self.last* fields
        """

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
        lvis (AFAIK) doesn't support transmitted
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
        "No way of knowing, maybe fseek/ftell?"
        raise generic.LiDARFunctionUnsupported()

    def getHeader(self):
        return {}

    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.getHeader()[name]
    
class LVISBinFileInfo(generic.LiDARFileInfo):
    """
    Class that gets information about a .las file
    and makes it available as fields.
    """
    def __init__(self, fname):
        generic.LiDARFileInfo.__init__(self, fname)
        
        self.lcename, self.lcwname, self.lgwname = getFilenames(fname)
            
    @staticmethod        
    def getDriverName():
        return 'LVIS Binary'
