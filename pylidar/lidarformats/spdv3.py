
"""
SPD V3
"""
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
            raise LiDARFormatNotUnderstood(str(err))
            
        if mode == generic.READ:
            self.si_cnt = self.fileHandle['INDEX']['PLS_PER_BIN'][...]
            self.si_idx = self.fileHandle['INDEX']['BIN_OFFSETS'][...]
            self.si_binSize = self.fileHandle['HEADER']['BIN_SIZE']
            self.si_xmin = self.fileHandle['HEADER']['X_MIN']
            self.si_ymax = self.fileHandle['HEADER']['Y_MAX']
            self.si_xmax = self.fileHandle['HEADER']['X_MAX']
            self.si_ymin = self.fileHandle['HEADER']['Y_MIN']
            self.wkt = self.fileHandle['HEADER']['SPATIAL_REFERENCE']
        else:
            # set on setPixelGrid
            self.si_cnt = None
            self.si_idx = None
            self.si_binSize = None
            self.si_xmin = None
            self.si_ymax = None
            self.si_xmax = None
            self.si_ymin = None
            self.wkt = None
            
        # so we can be clever about when to read from disk
        self.lastExtent = None
        self.lastPoints = None
        self.lastPulses = None

    def getPixelGrid(self):
        pixGrid = pixelgrid.PixelGridDefn(projection=self.wkt,
                    xMin=self.si_xmin, xMax=self.si_xmax,
                    yMin=self.si_ymin, yMax=self.si_ymax,
                    xRes=self.si_binSize, yRes=self.si_binSize)
        return pixGrid
    
    def setPixelGrid(self, pixGrid):
        assert self.mode != generic.READ
        self.si_binSize = pixGrid.xRes
        self.si_xmin = pixGrid.xMin
        self.si_ymax = pixGrid.yMax
        self.si_xmax = pixGrid.xMax
        self.si_ymin = pixGrid.yMin
        self.wkt = pixGrid.projection
        
        # create spatial index
        (nrows, ncols) = pixGrid.getDimensions()
        self.si_cnt = numpy.zeros((ncols, nrows), dtype=numpy.int)
        self.si_idx = numpy.zeros((ncols, nrows), dtype=numpy.int)
    
    @staticmethod
    def convertIdxCount(start_idx_array, count_array):
        """
        Needs to be numpy magic??
        """
        out = numpy.empty(count_array.sum(), numpy.int)
        inidx = 0
        outidx = 0
        while indx < start_idx_array.shape[0]:
            cnt = count_array[indx]
            startidx = start_idx_array[inidx]
            for i in range(cnt):
                out[outidx] = startidx + x
                outidx += 1
            indx += 1
    
    def readPointsForExtent(self, extent):
        # returned cached if possible
        if self.lastExtent == extent and not self.lastPoints is None:
            return self.lastPoints
            
        # this should also return anything cached
        pulses = self.readPulsesForExtent(extent)
        
        nReturns = pulses['NUMBER_OF_RETURNS']
        startIdxs = pulses['PTS_START_IDX']
        points_idx = numpy.empty(nReturns.sum(), dtype=numpy.int)
        
        pulse_idx = self.convertIdxCount(startIdxs, nReturns)
                
        points = self.fileHandle['DATA']['POINTS'][pulse_idx]
        
        self.lastExtent = extent
        self.lastPoints = points
        return points
            
    def readPulsesForExtent(self, extent):
        """
        """
        # returned cached if possible
        if self.lastExtent == extent and not self.lastPulses is None:
            return self.lastPulses
        
        tlxbin = int((extent.minx - self.si_minx) / self.si_binSize)
        tlybin = int((extent.maxy - self.si_maxy) / self.si_binSize)
        brxbin = int(numpy.ceil((extent.maxx - self.si_minx) / self.si_binSize))
        brybin = int(numpy.ceil((extent.miny - self.si_miny) / self.si_binSize))
        
        cnt_subset = self.si_cnt[tlxbin:tlybin+1, brxbin:brybin+1]
        idx_subset = self.si_idx[tlxbin:tlybin+1, brxbin:brybin+1]
        
        all_idx = self.convertIdxCount(idx_subset, cnt_subset)
        
        pulses = self.fileHandle['DATA']['PULSES'][all_idx]
        self.lastExtent = extent
        self.lastPulses = pulses
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
