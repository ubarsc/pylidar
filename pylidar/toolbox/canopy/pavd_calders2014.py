"""
Functions for calculating vertical plant profiles (Calders et al., 2014)
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

import numpy
import collections
import statsmodels.api as sm
from numba import jit

from pylidar import lidarprocessor
    

def run_pavd_calders2014(dataFiles, controls, otherargs, outfile):
    """
    Main function for PAVD_CALDERS2014
    """  
    if otherargs.planecorrection:  
        print("Applying plane correction to point heights...")
        
        otherargs.xgrid = numpy.zeros(otherargs.gridsize**2, dtype=numpy.float64)
        otherargs.ygrid = numpy.zeros(otherargs.gridsize**2, dtype=numpy.float64)
        otherargs.zgrid = numpy.zeros(otherargs.gridsize**2, dtype=numpy.float64)
        otherargs.gridmask = numpy.ones(otherargs.gridsize**2, dtype=numpy.bool)
                   
        lidarprocessor.doProcessing(runXYMinGridding, dataFiles, controls=controls, otherArgs=otherargs)
        
        otherargs.planefit = planeFitHubers(otherargs.xgrid[~otherargs.gridmask], otherargs.ygrid[~otherargs.gridmask], 
            otherargs.zgrid[~otherargs.gridmask], reportfile=otherargs.rptfile)
    
    minZenithAll = min(otherargs.minzenith)
    maxZenithAll = max(otherargs.maxzenith)
    minHeightBin = min(0.0, otherargs.minheight)  
    
    otherargs.zenith = numpy.arange(minZenithAll+otherargs.zenithbinsize/2, maxZenithAll, otherargs.zenithbinsize)
    otherargs.height = numpy.arange(minHeightBin, otherargs.maxheight, otherargs.heightbinsize)
    otherargs.counts = numpy.zeros([otherargs.zenith.shape[0],otherargs.height.shape[0]])
    otherargs.pulses = numpy.zeros([otherargs.zenith.shape[0],1])     
    
    print("Calculating vertical plant profiles...")
    lidarprocessor.doProcessing(runZenithHeightStratification, dataFiles, controls=controls, otherArgs=otherargs)
    
    pgapz = numpy.where(otherargs.pulses > 0, 1 - numpy.cumsum(otherargs.counts, axis=1) / otherargs.pulses, numpy.nan)
    zenithRadians = numpy.radians(otherargs.zenith)
    zenithBinSizeRadians = numpy.radians(otherargs.zenithbinsize)
    
    lpp_pai,lpp_pavd,lpp_mla = calcLinearPlantProfiles(otherargs.height, otherargs.heightbinsize, 
        zenithRadians, pgapz)
    sapp_pai,sapp_pavd = calcSolidAnglePlantProfiles(zenithRadians, pgapz, otherargs.heightbinsize,
        zenithBinSizeRadians)
    
    writeProfiles(outfile, otherargs.zenith, otherargs.height, pgapz, 
                  lpp_pai, lpp_pavd, lpp_mla, sapp_pai, sapp_pavd)


@jit
def countPointsPulsesByZenithHeight(midZenithBins,minimumZenith,maximumZenith,zenithBinSize,pulseZenith,pulsesByPointZenith,
                                 pointHeight,heightBins,heightBinSize,pointCounts,pulseCounts,weights,minHeight):
    """
    Called by runZenithHeightStratification()
    
    Updates a 2D array of point counts (zenith, height) and 1D array of pulse counts (zenith)
    
    Parameters:
        midZenithBins           1D array of midpoints of zenith angle bins to use for the stratification
        minimumZenith           Minimum zenith angle value to consider for this block
        maximumZenith           Maximum zenith angle value to consider for this block
        zenithBinSize           Zenith angle bin size
        pulseZenith             1D array of pulse zenith angles for this block
        pulsesByPointZenith     1D array of pulse zenith angles for each point in this block
        pointHeight             1D array of point heights for this block
        heightBins              1D array of vertical height bin starts to use the stratification
        heightBinSize           Vertical height bin size
        weights                 1D array of points weights to use for calculating point intercept counts
        
    Returns:    
        pointCounts             2D array (zenith bins, height bins) of point intercept counts to update for this block
        pulseCounts             1D array (zenith bins) of laser shot counts to update for this block       
    
    """
    for i in range(midZenithBins.shape[0]):        
        if (midZenithBins[i] > minimumZenith) and (midZenithBins[i] <= maximumZenith):            
            lowerzenith = midZenithBins[i] - zenithBinSize / 2
            upperzenith = midZenithBins[i] + zenithBinSize / 2
            for j in range(pulseZenith.shape[0]):
                if (pulseZenith[j] > lowerzenith) and (pulseZenith[j] <= upperzenith):
                    pulseCounts[i,0] += 1.0            
            for j in range(pulsesByPointZenith.shape[0]):
                if weights[j] > 0:
                    if (pulsesByPointZenith[j] > lowerzenith) and (pulsesByPointZenith[j] <= upperzenith):
                        k = int( (pointHeight[j] - heightBins[0]) / heightBinSize )
                        if (k >= 0) and (k < heightBins.shape[0]) and (pointHeight[j] > minHeight):
                            pointCounts[i,k] += weights[j]

@jit
def minPointsByXYGrid(pointX, pointY, pointZ, gridX, gridY, gridZ, gridMask,
                           minX, maxX, minY, maxY, resolution, nbinsX):
    """
    Called by runXYMinGridding()
    
    Updates 1D arrays of point coordinates corresponding to a minimum Z grid
    
    Parameters:
        pointX          1D array of point X coordinates for this block
        pointY          1D array of point Y coordinates for this block
        pointZ          1D array of point Z coordinates for this block
        minX            Minimum X coordinate to consider
        maxX            Maximum X coordinate to consider
        minY            Minimum Y coordinate to consider
        maxY            Maximum Y coordinate to consider
        resolution      Spatial resolution of out XY grid
        nbinsX          Number of X bins in the output grid 
        
    Returns:
        gridX           A 1D array representation of a 2D grid of minumum Z point X coordinates
        gridY           A 1D array representation of a 2D grid of minumum Z point Y coordinates
        gridZ           A 1D array representation of a 2D grid of minumum Z point Z coordinates
        gridMask        A 1D array representation of a 2D bool grid of missing values
    
    """
    for i in range(pointZ.shape[0]):
        if (pointX[i] >= minX) and (pointX[i] <= maxX) and (pointY[i] >= minY) and (pointY[i] <= maxY):
            j = int( (pointY[i] - minY) / resolution ) * nbinsX + int( (pointX[i] - minX) / resolution )
            if (j >= 0) and (j < gridZ.shape[0]):
                if not gridMask[j]:
                    if pointZ[i] < gridZ[j]:
                        gridX[j] = pointX[i]
                        gridY[j] = pointY[i]
                        gridZ[j] = pointZ[i]
                else:
                    gridX[j] = pointX[i]
                    gridY[j] = pointY[i]
                    gridZ[j] = pointZ[i]
                    gridMask[j] = False
    
def runXYMinGridding(data, otherargs):
    """
    Derive a minimum Z surface following plane correction procedures outlined in Calders et al. (2014)
    """
    pointcolnames = ['X','Y','Z']

    halfextent = (otherargs.gridsize * otherargs.gridbinsize) / 2.0

    minX = otherargs.origin[0] - halfextent
    minY = otherargs.origin[1] - halfextent 
    maxX = otherargs.origin[0] + halfextent
    maxY = otherargs.origin[1] + halfextent 
            
    for indata in data.inList:
        
        points = indata.getPoints(colNames=pointcolnames)
        
        minPointsByXYGrid(points['X'], points['Y'], points['Z'], otherargs.xgrid, otherargs.ygrid, otherargs.zgrid, 
            otherargs.gridmask, minX, maxX, minY, maxY, otherargs.gridbinsize, otherargs.gridsize)

def runZenithHeightStratification(data, otherargs):
    """
    Derive Pgap(z) profiles following vertical profile procedures outlined in Calders et al. (2014)
    """
        
    for i,indata in enumerate(data.inList):
        
        pulsecolnames = ['NUMBER_OF_RETURNS','ZENITH']
        pulses = indata.getPulses(colNames=pulsecolnames)
        pulsesByPoint = numpy.ma.repeat(pulses, pulses['NUMBER_OF_RETURNS'])
        
        if indata.lidarDriver == "SPDV3":
            returnnumcol = 'RETURN_ID'
            pulses['ZENITH'] = numpy.degrees(pulses['ZENITH'])
            pulsesByPoint['ZENITH'] = numpy.degrees(pulsesByPoint['ZENITH'])
        else:
            returnnumcol = 'RETURN_NUMBER'
        
        if otherargs.planecorrection:
            pointcolnames = ['X','Y','Z','CLASSIFICATION',returnnumcol]
        else:
            pointcolnames = [otherargs.heightcol,'CLASSIFICATION',returnnumcol]        
        points = indata.getPoints(colNames=pointcolnames)
        
        if otherargs.weighted:
            weights = 1.0 / pulsesByPoint['NUMBER_OF_RETURNS']
        else:
            weights = numpy.array(points[otherargs.returnnumcol[i]] == 1, dtype=numpy.float32)
        
        if len(otherargs.excludedclasses) > 0:
            mask = numpy.in1d(points['CLASSIFICATION'], otherargs.excludedclasses)
            weights[mask] = 0.0
        
        if otherargs.planecorrection:
            pointHeights = points['Z'] - (otherargs.planefit["Parameters"][1] * points['X'] + 
                otherargs.planefit["Parameters"][2] * points['Y'] + otherargs.planefit["Parameters"][0])
        else:
            pointHeights = points[otherargs.heightcol]
        
        countPointsPulsesByZenithHeight(otherargs.zenith,otherargs.minzenith[i],otherargs.maxzenith[i],
            otherargs.zenithbinsize,pulses['ZENITH'],pulsesByPoint['ZENITH'],pointHeights,
            otherargs.height,otherargs.heightbinsize,otherargs.counts,otherargs.pulses,
            weights,otherargs.minheight)

def calcLinearPlantProfiles(height, heightbinsize, zenith, pgapz):
    """
    Calculate the linear model PAI/PAVD (see Jupp et al., 2009)
    """ 
    kthetal = -numpy.log(pgapz)
    xtheta = numpy.abs(2 * numpy.tan(zenith) / numpy.pi)
    paiv = numpy.zeros(pgapz.shape[1])
    paih = numpy.zeros(pgapz.shape[1])
    for i,h in enumerate(height):    
        a = numpy.vstack([xtheta, numpy.ones(xtheta.size)]).T
        y = kthetal[:,i]
        mask = ~numpy.isnan(y)
        if numpy.count_nonzero(mask) > 2:
            lv, lh = numpy.linalg.lstsq(a[mask,:], y[mask])[0]        
            paiv[i] = lv
            paih[i] = lh    
            if lv < 0:
                paih[i] = numpy.mean(y[mask])
                paiv[i] = 0.0
            if lh < 0:
                paiv[i] = numpy.mean(y[mask] / xtheta[mask])
                paih[i] = 0.0
    
    pai = paiv + paih
    pavd = numpy.gradient(pai, heightbinsize)
    
    mla = numpy.degrees( numpy.arctan2(paiv,paih) )
    
    return pai,pavd,mla

def calcHingePlantProfiles(heightbinsize, zenith, pgapz):
    """
    Calculate the hinge angle PAI/PAVD (see Jupp et al., 2009)
    """       
    hingeindex = numpy.argmin(numpy.abs(zenith - numpy.arctan(numpy.pi / 2)))
    pai = -1.1 * numpy.log(pgapz[hingeindex,:])
    pavd = numpy.gradient(pai, heightbinsize)
    
    return pai,pavd   
                      
def calcSolidAnglePlantProfiles(zenith, pgapz, heightbinsize, zenithbinsize, pai=None):
    """
    Calculate the Jupp et al. (2009) solid angle weighted PAI/PAVD
    """
    w = 2 * numpy.pi * numpy.sin(zenith) * zenithbinsize
    wn = w / numpy.sum(w[pgapz[:,-1] < 1])        
    ratio = numpy.zeros(pgapz.shape[1])
    
    for i in range(zenith.size):
        if (pgapz[i,-1] < 1):
            ratio += wn[i] * numpy.log(pgapz[i,:]) / numpy.log(pgapz[i,-1])
    
    if pai is None:
        hingeindex = numpy.argmin(numpy.abs(zenith - numpy.arctan(numpy.pi / 2)))
        pai = -1.1 * numpy.log(pgapz[hingeindex,-1])
    
    pai = pai * ratio
    pavd = numpy.gradient(pai, heightbinsize)
    
    return pai,pavd

def getProfileAsArray(zenith, height, pgapz, lpp_pai, lpp_pavd, lpp_mla, 
            sapp_pai, sapp_pavd):
    """
    Returns the vertical profile information as a single structured array
    """
    arrayDtype = [('height', 'f8')]
    vzNames = []
    for ring in zenith:
        name = "vz%05i"%(ring*100)
        vzNames.append(name)
        
        arrayDtype.append((name, 'f8'))

    for name in ["linearPAI", "linearPAVD", "linearMLA", "hingePAI", "juppPAVD"]:
        arrayDtype.append((name, 'f8'))

    profileArray = numpy.empty((height.shape[0], ), dtype=arrayDtype)

    profileArray['height'] = height
    for count, name in enumerate(vzNames):
        profileArray[name] = pgapz[count]

    profileArray["linearPAI"] = lpp_pai
    profileArray["linearPAVD"] = lpp_pavd
    profileArray["linearMLA"] = lpp_mla
    profileArray["hingePAI"] = sapp_pai
    profileArray["juppPAVD"] = sapp_pavd

    return profileArray
    
def writeProfiles(outfile, zenith, height, pgapz, lpp_pai, lpp_pavd, lpp_mla, 
            sapp_pai, sapp_pavd):
    """
    Write out the vertical profiles to file
    """  
    profileArray = getProfileAsArray(zenith, height, pgapz, lpp_pai, lpp_pavd, 
            lpp_mla, sapp_pai, sapp_pavd)
    
    numpy.savetxt(outfile, profileArray, fmt="%.4f", delimiter=',', 
            header=','.join(profileArray.dtype.names))

def planeFitHubers(x, y, z, reportfile=None):
    """
    Plane fitting (Huber's T norm with median absolute deviation scaling)
    Weighting by 1 / point range yet to be implemented in statsmodels.api
    """    
    xy = numpy.vstack((x,y)).T
    xy = sm.add_constant(xy)
    huber_t = sm.RLM(z, xy, M=sm.robust.norms.HuberT())
    huber_results = huber_t.fit()
            
    outdictn = collections.OrderedDict()
    outdictn["Parameters"] = huber_results.params        
    outdictn["Summary"] = huber_results.summary(yname='Z', xname=['Intercept','X','Y'])
    outdictn["Slope"] = numpy.degrees( numpy.arctan(numpy.sqrt(outdictn["Parameters"][1]**2 + outdictn["Parameters"][2]**2)) )
    outdictn["Aspect"] = numpy.degrees( numpy.arctan(outdictn["Parameters"][1] / outdictn["Parameters"][2]) )
    
    if reportfile is not None:
        f = open(reportfile,'w')
        for k,v in outdictn.items():
            f.write("%s:\n%s\n" % (k,v))
        f.close()
       
    return outdictn
