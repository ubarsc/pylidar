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
        otherargs.rgrid = numpy.zeros(otherargs.gridsize**2, dtype=numpy.float64)
        otherargs.gridmask = numpy.ones(otherargs.gridsize**2, dtype=numpy.bool)
                  
        lidarprocessor.doProcessing(runXYMinGridding, dataFiles, controls=controls, otherArgs=otherargs)
        
        otherargs.planefit = planeFitHubers(otherargs.xgrid[~otherargs.gridmask], otherargs.ygrid[~otherargs.gridmask], 
            otherargs.zgrid[~otherargs.gridmask], otherargs.rgrid[~otherargs.gridmask], reportfile=otherargs.rptfile)
    
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
    
    hpp_pai,hpp_pavd = calcHingePlantProfiles(otherargs.heightbinsize, zenithRadians, pgapz)
    
    lpp_pai,lpp_pavd,lpp_mla = calcLinearPlantProfiles(otherargs.height, otherargs.heightbinsize, 
        zenithRadians, pgapz)
        
    if otherargs.totalpaimethod == "HINGE":
        total_pai = numpy.max(hpp_pai)
    elif otherargs.totalpaimethod == "LINEAR":
        total_pai = numpy.max(lpp_pai)
    elif otherargs.totalpaimethod == "EXTERNAL":
        total_pai = otherargs.externalpai
    
    spp_pai,spp_pavd = calcSolidAnglePlantProfiles(zenithRadians, pgapz, otherargs.heightbinsize,
        zenithBinSizeRadians, total_pai)     
    
    writeProfiles(outfile, otherargs.zenith, otherargs.height, pgapz, 
                  lpp_pai, lpp_pavd, lpp_mla, hpp_pai, hpp_pavd, spp_pai, spp_pavd)


@jit
def countPointsPulsesByZenithHeight(midZenithBins,minimumAzimuth,maximumAzimuth,minimumZenith,maximumZenith,zenithBinSize,
    pulseAzimuth,pulsesByPointAzimuth,pulseZenith,pulsesByPointZenith,pointHeight,heightBins,heightBinSize,pointCounts,pulseCounts,weights,minHeight):
    """
    Called by runZenithHeightStratification()
    
    Updates a 2D array of point counts (zenith, height) and 1D array of pulse counts (zenith)
    
    Parameters:
        midZenithBins           1D array of midpoints of zenith angle bins to use for the stratification
        minimumAzimuth          Minimum azimuth angle value to consider for this block
        maximumAzimuth          Maximum azimuth angle value to consider for this block
        minimumZenith           Minimum zenith angle value to consider for this block
        maximumZenith           Maximum zenith angle value to consider for this block
        zenithBinSize           Zenith angle bin size
        pulseAzimuth            1D array of pulse azimuth angles for this block
        pulsesByPointAzimuth    1D array of pulse azimuth angles for each point in this block
        pulseZenith             1D array of pulse zenith angles for this block
        pulsesByPointZenith     1D array of pulse zenith angles for each point in this block
        pointHeight             1D array of point heights for this block
        heightBins              1D array of vertical height bin starts to use the stratification
        heightBinSize           Vertical height bin size
        weights                 1D array of points weights to use for calculating point intercept counts
        minHeight               Minimum height to include in the vertical profile
        
    Returns:    
        pointCounts             2D array (zenith bins, height bins) of point intercept counts to update for this block
        pulseCounts             1D array (zenith bins) of laser shot counts to update for this block       
    
    """
    for i in range(midZenithBins.shape[0]):        
        if (midZenithBins[i] > minimumZenith) and (midZenithBins[i] <= maximumZenith):            
            lowerzenith = midZenithBins[i] - zenithBinSize / 2
            upperzenith = midZenithBins[i] + zenithBinSize / 2
            for j in range(pulseZenith.shape[0]):
                if (pulseZenith[j] > lowerzenith) and (pulseZenith[j] <= upperzenith) and (pulseAzimuth[j] >= minimumAzimuth) and (pulseAzimuth[j] <= maximumAzimuth):
                    pulseCounts[i,0] += 1.0            
            for j in range(pulsesByPointZenith.shape[0]):
                if weights[j] > 0:
                    if (pulsesByPointZenith[j] > lowerzenith) and (pulsesByPointZenith[j] <= upperzenith) and (pulsesByPointAzimuth[j] >= minimumAzimuth) and (pulsesByPointAzimuth[j] <= maximumAzimuth):
                        k = int( (pointHeight[j] - heightBins[0]) / heightBinSize )
                        if (k >= 0) and (k < heightBins.shape[0]) and (pointHeight[j] > minHeight):
                            pointCounts[i,k] += weights[j]

@jit
def minPointsByXYGrid(pointX, pointY, pointZ, pointR, gridX, gridY, gridZ, gridR, gridMask,
                           minX, maxX, minY, maxY, resolution, nbinsX):
    """
    Called by runXYMinGridding()
    
    Updates 1D arrays of point coordinates corresponding to a minimum Z grid
    
    Parameters:
        pointX          1D array of point X coordinates for this block
        pointY          1D array of point Y coordinates for this block
        pointZ          1D array of point Z coordinates for this block
        pointR          1D array of point range coordinates for this block
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
        gridR           A 1D array representation of a 2D grid of minumum Z point range coordinates
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
                        gridR[j] = pointR[i]
                else:
                    gridX[j] = pointX[i]
                    gridY[j] = pointY[i]
                    gridZ[j] = pointZ[i]
                    gridR[j] = pointR[i]
                    gridMask[j] = False
    
def runXYMinGridding(data, otherargs):
    """
    Derive a minimum Z surface following plane correction procedures outlined in Calders et al. (2014)
    """
    pulsecolnames = ['X_ORIGIN','Y_ORIGIN','Z_ORIGIN','NUMBER_OF_RETURNS']    
    pointcolnames = ['X','Y','Z','RANGE']

    halfextent = (otherargs.gridsize * otherargs.gridbinsize) / 2.0

    minX = -halfextent
    minY = -halfextent 
    maxX = halfextent
    maxY = halfextent 
            
    for indata in data.inFiles:
        
        points = indata.getPoints(colNames=pointcolnames)
        pulses = indata.getPulses(colNames=pulsecolnames)
        pulsesByPoint = numpy.ma.repeat(pulses, pulses['NUMBER_OF_RETURNS'])
        
        x = points['X'] - pulsesByPoint['X_ORIGIN']
        y = points['Y'] - pulsesByPoint['Y_ORIGIN']
        z = points['Z'] - pulsesByPoint['Z_ORIGIN']
        
        minPointsByXYGrid(x, y, z, points['RANGE'], otherargs.xgrid, otherargs.ygrid, otherargs.zgrid, otherargs.rgrid, 
            otherargs.gridmask, minX, maxX, minY, maxY, otherargs.gridbinsize, otherargs.gridsize)

def runZenithHeightStratification(data, otherargs):
    """
    Derive Pgap(z) profiles following vertical profile procedures outlined in Calders et al. (2014)
    """
        
    for i,indata in enumerate(data.inFiles):
        
        pulsecolnames = ['X_ORIGIN','Y_ORIGIN','Z_ORIGIN','NUMBER_OF_RETURNS','ZENITH','AZIMUTH']
        pulses = indata.getPulses(colNames=pulsecolnames)
        pulsesByPoint = numpy.ma.repeat(pulses, pulses['NUMBER_OF_RETURNS'])
        
        if otherargs.lidardriver[i] == "SPDV3":
            returnnumcol = 'RETURN_ID'
            pulses['ZENITH'] = numpy.degrees(pulses['ZENITH'])
            pulsesByPoint['ZENITH'] = numpy.degrees(pulsesByPoint['ZENITH'])
            pulses['AZIMUTH'] = numpy.degrees(pulses['AZIMUTH'])
            pulsesByPoint['AZIMUTH'] = numpy.degrees(pulsesByPoint['AZIMUTH'])
        else:
            returnnumcol = 'RETURN_NUMBER'
        
        if otherargs.planecorrection or otherargs.externaldem is not None:
            pointcolnames = ['X','Y','Z','CLASSIFICATION',returnnumcol]
        else:
            pointcolnames = [otherargs.heightcol,'CLASSIFICATION',returnnumcol]        
        points = indata.getPoints(colNames=pointcolnames)
        
        if otherargs.weighted:
            weights = 1.0 / pulsesByPoint['NUMBER_OF_RETURNS']
        else:
            weights = numpy.array(points[returnnumcol] == 1, dtype=numpy.float32)
        
        if len(otherargs.excludedclasses) > 0:
            mask = numpy.in1d(points['CLASSIFICATION'], otherargs.excludedclasses)
            weights[mask] = 0.0
        
        if otherargs.planecorrection:
            x = points['X'] - pulsesByPoint['X_ORIGIN']
            y = points['Y'] - pulsesByPoint['Y_ORIGIN']
            z = points['Z'] - pulsesByPoint['Z_ORIGIN']            
            pointHeights = z - (otherargs.planefit["Parameters"][1] * x + 
                otherargs.planefit["Parameters"][2] * y + otherargs.planefit["Parameters"][0])
        elif otherargs.externaldem is not None:
            pointHeights = extractPointHeightsFromDEM(points['X'], points['Y'], points['Z'], otherargs)
        else:
            pointHeights = points[otherargs.heightcol]
        
        countPointsPulsesByZenithHeight(otherargs.zenith,otherargs.minazimuth[i],otherargs.maxazimuth[i],
            otherargs.minzenith[i],otherargs.maxzenith[i],otherargs.zenithbinsize,
            pulses['AZIMUTH'],pulsesByPoint['AZIMUTH'],pulses['ZENITH'],pulsesByPoint['ZENITH'],
            pointHeights,otherargs.height,otherargs.heightbinsize,otherargs.counts,otherargs.pulses,
            weights,otherargs.minheight)

def extractPointHeightsFromDEM(x, y, z, otherargs):
    """
    Extract point heights from an external DEM
    For points outside the DEM extent, we use nearest neighbour values
    TODO: Interpolate DEM elevations to actual point locations
    """
    col = ((x - otherargs.xMinDem) / otherargs.binSizeDem).astype(numpy.uint)
    row = ((otherargs.yMaxDem - y) / otherargs.binSizeDem).astype(numpy.uint)           
    pointHeights = numpy.empty(z.shape, dtype=numpy.float32)
    
    inside = (row >= 0) & (row < otherargs.dataDem.shape[0]) & \
             (col >= 0) & (col < otherargs.dataDem.shape[1])
    pointHeights[inside] = z[inside] - otherargs.dataDem[row[inside], col[inside]]
    
    left = (row >= 0) & (row < otherargs.dataDem.shape[0]) & (col < 0)
    pointHeights[left] = z[left] - otherargs.dataDem[row[left], 0]

    right = (row >= 0) & (row < otherargs.dataDem.shape[0]) & (col >= otherargs.dataDem.shape[1])
    pointHeights[right] = z[right] - otherargs.dataDem[row[right], -1]

    top = (row < 0)
    pointHeights[top] = z[top] - otherargs.dataDem[0, numpy.clip(col[top],0,otherargs.dataDem.shape[1]-1)]

    bottom = (row >= otherargs.dataDem.shape[0])
    pointHeights[bottom] = z[bottom] - otherargs.dataDem[-1, numpy.clip(col[bottom],0,otherargs.dataDem.shape[0]-1)]
    
    return pointHeights

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
                      
def calcSolidAnglePlantProfiles(zenith, pgapz, heightbinsize, zenithbinsize, totalpai):
    """
    Calculate the Jupp et al. (2009) solid angle weighted PAI/PAVD
    """
    w = 2 * numpy.pi * numpy.sin(zenith) * zenithbinsize
    wn = w / numpy.sum(w[pgapz[:,-1] < 1])        
    ratio = numpy.zeros(pgapz.shape[1])
    
    for i in range(zenith.size):
        if (pgapz[i,-1] < 1):
            ratio += wn[i] * numpy.log(pgapz[i,:]) / numpy.log(pgapz[i,-1])       
    
    pai = totalpai * ratio
    pavd = numpy.gradient(pai, heightbinsize)
    
    return pai,pavd

def getProfileAsArray(zenith, height, pgapz, lpp_pai, lpp_pavd, lpp_mla, 
            hpp_pai, hpp_pavd, sapp_pai, sapp_pavd):
    """
    Returns the vertical profile information as a single structured array
    """
    arrayDtype = [('height', 'f8')]
    vzNames = []
    for ring in zenith:
        name = "vz%05i"%(ring*100)
        vzNames.append(name)
        
        arrayDtype.append((name, 'f8'))

    for name in ["linearPAI", "linearPAVD", "linearMLA", "hingePAI", "hingePAVD", 
        "weightedPAI", "weightedPAVD"]:
        arrayDtype.append((name, 'f8'))

    profileArray = numpy.empty((height.shape[0], ), dtype=arrayDtype)

    profileArray['height'] = height
    for count, name in enumerate(vzNames):
        profileArray[name] = pgapz[count]

    profileArray["linearPAI"] = lpp_pai
    profileArray["linearPAVD"] = lpp_pavd
    profileArray["linearMLA"] = lpp_mla
    profileArray["hingePAI"] = hpp_pai
    profileArray["hingePAVD"] = hpp_pavd
    profileArray["weightedPAI"] = sapp_pai
    profileArray["weightedPAVD"] = sapp_pavd
    
    return profileArray
    
def writeProfiles(outfile, zenith, height, pgapz, lpp_pai, lpp_pavd, lpp_mla, 
            hpp_pai, hpp_pavd, sapp_pai, sapp_pavd):
    """
    Write out the vertical profiles to file
    """  
    profileArray = getProfileAsArray(zenith, height, pgapz, lpp_pai, lpp_pavd, 
            lpp_mla, hpp_pai, hpp_pavd, sapp_pai, sapp_pavd)
    
    numpy.savetxt(outfile, profileArray, fmt="%.4f", delimiter=',', 
            header=','.join(profileArray.dtype.names))

def planeFitHubers(x, y, z, r, reportfile=None):
    """
    Plane fitting (Huber's T norm with median absolute deviation scaling)
    Prior weights are set to 1 / point range.
    """
    xy = numpy.vstack((x/r,y/r)).T
    xy = sm.add_constant(xy)
    huber_t = sm.RLM(z/r, xy, M=sm.robust.norms.HuberT())
    huber_results = huber_t.fit()
            
    outdictn = collections.OrderedDict()
    outdictn["Parameters"] = huber_results.params        
    outdictn["Summary"] = huber_results.summary(yname='Z', xname=['Intercept','X','Y'])
    outdictn["Slope"] = numpy.degrees( numpy.arctan(numpy.sqrt(outdictn["Parameters"][1]**2 + outdictn["Parameters"][2]**2)) )
    outdictn["Aspect"] = numpy.degrees( numpy.arctan(outdictn["Parameters"][1] / outdictn["Parameters"][2]) )
    
    if reportfile is not None:
        f = open(reportfile,'w')
        for k,v in outdictn.items():
            # Parameters are in the summary output
            if k != "Parameters":
                f.write("%s:\n%s\n" % (k,v))
        f.close()
       
    return outdictn
