
"""
SPD V3
"""
import copy
import numpy
import h5py
from rios import pixelgrid
from . import generic

class SPDV3File(generic.LiDARFile):
    def __init__(self, fname, mode):
        self.mode = mode
    
        # convert mode into h5py mode string
        if mode == generic.READ:
            h5py_mode = 'r'
        elif mode == generic.UPDATE:
            h5py_mode = 'r+'
        elif mode == generic.CREATE:
            h5py_mode == 'w'
        else:
            raise ValueError('Unknown value for mode parameter')
    
        # attempt to open the file
        try:
            self.fileHandle = h5py.File(fname, h5py_mode)
        except OSError as err:
            # always seems to through an OSError
            raise generic.LiDARFormatNotUnderstood(str(err))
            
        # check that it is indeed the right version
        headerKeys = self.fileHandle['HEADER'].keys()
        if (not 'VERSION_MAJOR_SPD' in headerKeys or 
                    not 'VERSION_MINOR_SPD' in headerKeys):
            msg = "File appears not to be SPD"
            raise generic.LiDARFormatNotUnderstood(msg)
        elif self.fileHandle['HEADER']['VERSION_MAJOR_SPD'][0] != 2:
            msg = "File seems to be wrong version for this driver"
            raise generic.LiDARFormatNotUnderstood(msg)

        # read in the bits I need            
        if mode == generic.READ:
            self.si_cnt = self.fileHandle['INDEX']['PLS_PER_BIN'][...]
            self.si_idx = self.fileHandle['INDEX']['BIN_OFFSETS'][...]
            self.si_binSize = self.fileHandle['HEADER']['BIN_SIZE'][0]
            self.si_xMin = self.fileHandle['HEADER']['X_MIN'][0]
            self.si_yMax = self.fileHandle['HEADER']['Y_MAX'][0]
            self.si_xMax = self.fileHandle['HEADER']['X_MAX'][0]
            self.si_yMin = self.fileHandle['HEADER']['Y_MIN'][0]
            self.wkt = self.fileHandle['HEADER']['SPATIAL_REFERENCE'][0].decode()
        else:
            # set on setPixelGrid
            self.si_cnt = None
            self.si_idx = None
            self.si_binSize = None
            self.si_xMin = None
            self.si_yMax = None
            self.si_xMax = None
            self.si_yMin = None
            self.wkt = None
            
        # so we can be clever about when to read from disk
        self.lastExtent = None
        self.lastPoints = None
        self.lastPulses = None

    def getPixelGrid(self):
        pixGrid = pixelgrid.PixelGridDefn(projection=self.wkt,
                    xMin=self.si_xMin, xMax=self.si_xMax,
                    yMin=self.si_yMin, yMax=self.si_yMax,
                    xRes=self.si_binSize, yRes=self.si_binSize)
        return pixGrid
    
    def setPixelGrid(self, pixGrid):
        assert self.mode != generic.READ
        self.si_binSize = pixGrid.xRes
        self.si_xMin = pixGrid.xMin
        self.si_yMax = pixGrid.yMax
        self.si_xMax = pixGrid.xMax
        self.si_yMin = pixGrid.yMin
        self.wkt = pixGrid.projection
        
        # create spatial index
        (nrows, ncols) = pixGrid.getDimensions()
        self.si_cnt = numpy.zeros((ncols, nrows), dtype=numpy.int)
        self.si_idx = numpy.zeros((ncols, nrows), dtype=numpy.int)
    
    @staticmethod
    def convertIdxCount(start_idx_array, count_array):
        """
        Convert SPD's default regular spatial index of pulse offsets and pulse counts
        per bin into a single array of pulse offsets for every pulse in the given
        subset of the spatial index. 

        """
        idx = start_idx_array.flatten()
        cnt = count_array.flatten()
        
        numBins = len(cnt)
        numPulses = cnt.sum()
        out = numpy.empty(numPulses, dtype=numpy.int64)
        nextPulse = 0
        for i in range(numBins):
            offsetsForBin = numpy.arange(idx[i], idx[i] + cnt[i])
            out[nextPulse:nextPulse+cnt[i]] = offsetsForBin
            nextPulse += cnt[i]
        # I believe that this calculation we have just done is also the perfect
        # place to assemble some sort of array showing which bin each pulse belongs with,
        # so it might be that we should return two things from this function. I don't yet 
        # know what such an array would look like, though....
        return out
    
    def readPointsForExtent(self, extent):
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == extent and 
                        not self.lastPoints is None):
            return self.lastPoints
            
        # this should also return anything cached
        pulses = self.readPulsesForExtent(extent)
        
        nReturns = pulses['NUMBER_OF_RETURNS']
        startIdxs = pulses['PTS_START_IDX']
        points_idx = numpy.empty(nReturns.sum(), dtype=numpy.int)
        
        pulse_idx = self.convertIdxCount(startIdxs, nReturns)

        # TODO: h5py doesn't seem to be able to use indices
        # directly from an array. Sigh.
        pulse_idx = list(pulse_idx)
                
        if len(pulse_idx) > 0:
            points = self.fileHandle['DATA']['POINTS'][pulse_idx]
        else:
            points = self.fileHandle['DATA']['POINTS'][0:0]
        
        self.lastExtent = copy.copy(extent)
        self.lastPoints = points
        return points
            
    def readPulsesForExtent(self, extent):
        """
        """
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == extent and 
                        not self.lastPulses is None):
            return self.lastPulses
        
        tlxbin = int((extent.xMin - self.si_xMin) / self.si_binSize)
        tlybin = int((extent.yMax - self.si_yMax) / self.si_binSize)
        brxbin = int(numpy.ceil((extent.xMax - self.si_xMin) / self.si_binSize))
        brybin = int(numpy.ceil((self.si_yMax - extent.yMin) / self.si_binSize))
        
        if tlxbin < 0:
            tlxbin = 0
        if tlybin < 0:
            tlybin = 0
        if brxbin > self.si_cnt.shape[0]:
            brxbin = self.si_cnt.shape[0]
        if brybin > self.si_cnt.shape[1]:
            brybin = self.si_cnt.shape[1]
        
        cnt_subset = self.si_cnt[tlxbin:brxbin+1, tlybin:brybin+1].flatten()
        idx_subset = self.si_idx[tlxbin:brxbin+1, tlybin:brybin+1].flatten()
        all_idx = self.convertIdxCount(idx_subset, cnt_subset)
        
        # TODO: h5py doesn't seem to be able to use indices
        # directly from an array. Sigh.
        all_idx = list(all_idx)
        
        if len(all_idx) > 0:
            pulses = self.fileHandle['DATA']['PULSES'][all_idx]
        else:
            # just an empty array with all the right fields
            pulses = self.fileHandle['DATA']['PULSES'][0:0]
        self.lastExtent = copy.copy(extent)
        self.lastPulses = pulses
        self.lastPoints = None # are now invalid
        return pulses
    
    
    def writePointsForExtent(self, extent, points):
        pass
    def writePulsesForExtent(self, extent, pulses):
        pass
    # see below for no spatial index
    def readPoints(self, n):
        pass
    def readPulses(self, n):
        pass
    def writePoints(self, points):
        pass
    def writePulses(self, pulses):
        pass

    def close(self, headerInfo=None):
        if self.mode != generic.READ:
            # write out to file
            self.fileHandle['INDEX']['PLS_PER_BIN'] = self.si_cnt
            self.fileHandle['INDEX']['BIN_OFFSETS'] = self.si_idx
            self.fileHandle['HEADER']['BIN_SIZE'] = self.si_binSize
            self.fileHandle['HEADER']['X_MIN'] = self.si_xmin
            self.fileHandle['HEADER']['Y_MAX'] = self.si_ymax
            self.fileHandle['HEADER']['X_MAX'] = self.si_xmax
            self.fileHandle['HEADER']['Y_MIN'] = self.si_ymin
            self.fileHandle['HEADER']['SPATIAL_REFERENCE'] = self.wkt
            
        # close
        self.fileHandle.close()
        self.fileHandle = None        
        self.lastExtent = None
        self.lastPoints = None
        self.lastPulses = None
