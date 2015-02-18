
"""
SPD V3
"""
import h5py
from pylidar import 
from . import generic

class SPDV3File(generic.LiDARFile):
    def __init__(self, fname, mode):
        # TODO: mode
        self.fileHandle = h5py.File(fname)
        if mode == generic.READ:
            self.si_cnt = self.fileHandle['INDEX']['PLS_PER_BIN'][...]
            self.si_idx = self.fileHandle['INDEX']['BIN_OFFSETS'][...]
            self.si_binSize = self.fileHandle['HEADER']['BIN_SIZE']
            self.si_tlx = self.fileHandle['HEADER']['X_MIN']
            self.si_tly = self.fileHandle['HEADER']['Y_MAX']
        else:
            # TODO: create blank arrays
            self.si_cnt = None
            self.si_idx = None
    
    def readPointsForExtent(self, extent):
        pass
    def readPulsesForExtent(self, extent):
        """
        """
        tlxbin = int((extent.tlx - self.si_tlx) / self.si_binSize)
        tlybin = int((extent.tly - self.si_tly) / self.si_binSize)
        # TODO:
        brxbin = int(ceil((
        
        cnt_subset = self.si_cnt[tlxbin:tlybin+1, brxbin:brybin+1]
        idx_subset = self.si_idx[tlxbin:tlybin+1, brxbin:brybin+1]
        
        # Magic numpy thing to get all indexes from idx and cnt
        all_indx = ????
        
        pulses = f['DATA']['PULSES'][all_idx]
        return pulses
    
    
    def writePointsForExtent(self, extent, data):
        pass
    def writePulsesForExtent(self, extent, data):
        pass
    # see below for no spatial index
    def readPoints(self, n):
        pass
    def readPulses(self, n):
        pass
    def writePoints(self, data):
        pass
    def writePulses(self, data):
        pass





