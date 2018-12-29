"""
Functions for multilayer crown delineation from ALS point cloud data (Duncanson et al., 2014)
Assumes a ground classification exists and is included in the input file format.
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
import pynninterp
from numba import jit
from scipy import ndimage

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.morphology import disk

from pylidar.toolbox import spatial
from pylidar.lidarformats import generic


def run_crown_duncanson2014(points, otherargs, outfiles):
    """
    Main function for CROWN_DUNCANSON2014
    """
    
    # Initialize the output files
    outImageFile = spatial.ImageWriter(outfiles[0], tlx=otherargs.bounds[0], tly=otherargs.bounds[3], 
            binSize=otherargs.resolution, driverName=otherargs.rasterdriver, epsg=None, 
            numBands=otherargs.maxlayers, nullVal=0)
    outMetricsFile = open(outfiles[1],'w')
    
    # Output grid size
    nX = (otherargs.bounds[1] - otherargs.bounds[0]) / otherargs.resolution
    nY = (otherargs.bounds[3] - otherargs.bounds[2]) / otherargs.resolution
    
    # Calculate height above ground of all points
    heights = calcHeights(points)
    
    # Run the process for a user defined number of layers
    for layer in range(otherargs.maxlayers):
    
        # Initialize the output grid
        outImage = numpy.zeros((nY,nX))
    
        if points.shape[0] > 0:
        
            # Derive the regions for the entire point cloud
            regions = processLayer(points, heights, None, outImage, otherargs)
            
            # Process vertical profiles for each region
            points_mask,region_ids,region_heights = processProfiles(regions, points, heights, otherargs)
            
            # Derive the regions for the upppermost layer point cloud
            regions = processLayer(points, heights, points_mask, outImage, otherargs)
            
            # Write to the output image stack
            outImageFile.setLayer(regions, layerNum=layer+1)
            
            # Get the crown metrics for the uppermost layer
            metrics_tmp = extractMetrics(regions, region_ids, region_heights, otherargs)
            if layer > 0:
                metrics = numpy.concatenate((metrics, metrics_tmp))
            else:
                metrics = metrics_tmp
            
            # Remove the upppermost layer from the point cloud
            points = points[~points_mask]
            heights = heights[~points_mask]
            
    outImageFile.close()
    numpy.savetxt(outfiles[1], metrics, fmt="%.4f", delimiter=',', 
            header=','.join(metrics.dtype.names))
        

def processLayer(points, heights, points_mask, outImage, x_min, y_max, otherargs):
    """
    Segment the uppermost canopy layer
    """        
    # Get the canopy height model
    if points_mask is None:
        findMaxHeights(points, heights, outImage, otherargs.bounds[0], otherargs.bounds[3], 
                otherargs.resolution)
    else:
        findMaxHeights(points[points_mask,:], heights[points_mask], outImage, 
                otherargs.bounds[0], otherargs.bounds[3], otherargs.resolution)
    
    # Fill pits in the CHM
    filled_chm = fillCHM(outImage, windowsize=otherargs.windowsize, minheight=otherargs.minheight)
    
    # Smooth the filled CHM
    smoothed_chm = smoothCHM(filled_chm, windowsize=otherargs.windowsize)
    
    # Run the watershed
    regions = runWatershed(smoothed_chm, windowsize=otherargs.windowsize, 
            minheight=otherargs.minheight)    
    
    return regions

    
def processProfiles(regions, points, heights, otherargs):
    """
    Extract the vertical profiles for the current set of segments
    and mask points that are not in the uppermost canopy
    """        
    # Initialize the profiles array
    region_ids,region_cnts = numpy.unique(regions, return_counts=True)
    z_max = numpy.max(heights)
    nz = int(z_max / otherargs.heightbinsize) + 1
    profile_data = numpy.zeros((region_ids.size, nz))
    profile_heights = numpy.zeros(region_ids.size)
    
    # Derive the vertical profile for each segment  
    binHeightsByRegion(points, heights, regions, region_ids, profile_data, profile_heights,  
            otherargs.bounds[0], otherargs.bounds[3], otherargs.resolution, otherargs.heightbinsize)  
    
    # Segment the profiles
    height_thresholds = numpy.zeros(region_ids.size)
    segmentProfiles(region_ids, profile_data, height_thresholds)
    
    # Update the point index
    points_mask = numpy.full(heights.size, False)
    maskPoints(data, heights, region_image, region_ids, height_thresholds, points_mask, 
            otherargs.bounds[0], otherargs.bounds[3])
    
    return points_mask,region_ids,profile_heights


@jit(nopython=True)
def binHeightsByRegion(data, heights, region_image, region_ids, profile_data, profile_heights,
    xMin, yMax, binSize, heightBinSize):
    """
    Create a maximum canopy height grid
    """
    for i in range(data.shape[0]):
        row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                xMin, yMax, binSize)
        if (row >= 0) & (col >= 0) & (row < region_image.shape[0]) & (col < region_image.shape[1]):
            if regionImage[row, col] != 0:
                for j in range(region_ids.shape[0]):
                    if region_ids[j] == regionImage[row, col]:
                        if heights[i] > profile_heights[j]:
                            profile_heights[j] = heights[i]
                        hbin = int(heights[i] / heightBinSize)
                        if (hbin >= 0) & (hbin < profile_data.shape[1]):
                            profile_data[j,hbin] += 1


@jit(nopython=True)
def segmentProfiles(profile_data, profile_heights, heightbinsize, vertical_buffer=0.1, relative_thres=0.15):
    """
    Smooth and split the vertical profile
    Smoothing width is dependent on height of the tree, 0.15*segment height in bins
    Buffer is 10% of the max height (so passing a truncated profile)
    """
    peak_idx = 0
    for i in range(profile_heights.shape[0]):
        smoothing_width = int(relative_thres * profile_heights[i])   
        start_bin = int(profile_heights[i] * vertical_buffer / heightbinsize)        
        end_bin = int(profile_heights[i] / heightbinsize) - start_bin    
        for_dip = profile_data[start_bin:end_bin,i]
        bins_dip = height_bins[start_bin:end_bin]       
        findProfilePeak(for_dip, bins_dip, peak_idx, smoothing_width=smoothing_width)
        if peak_idx[0] > 0:
            height_thresholds[i] = bins_dip[peak_idx[0]]
        else:
            height_thresholds[i] = 0
    

@jit(nopython=True)
def maskPoints(data, heights, region_image, region_ids, height_thresholds, data_mask, xMin, yMax):
    """
    Mask points that are not in the uppermost canopy
    """
    for i in range(data.shape[0]):
        row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                xMin, yMax, binSize)
        if (row >= 0) & (col >= 0) & (row < region_image.shape[0]) & (col < region_image.shape[1]):
            if regionImage[row, col] != 0:
                for j in range(region_ids.shape[0]):
                    if region_ids[j] == regionImage[row, col]:
                        if height[i] > height_thresholds[j]:
                            data_mask[i] = True                

                
def extractMetrics(regions, region_ids, region_heights, otherargs):
    """
    Extract metrics for each segment
    Implemented as done in Duncanson et al. (2014) but can be improved
    """    
    # Generate easting and northing values for each pixel
    xgrid,ygrid = numpy.meshgrid(numpy.arange(regions.shape[1]),numpy.arange(regions.shape[0]))
    xgrid = xgrid * otherargs.resolution + otherargs.bounds[0]
    ygrid = otherargs.bounds[3] - ygrid * otherargs.resolution
    
    # Initialize the output metrics structured array    
    arrayDtype = [("CrownID", 'i4')]
    for name in ["Easting", "Northing", "Height", "Area", "DiameterX", "DiameterY"]:
        arrayDtype.append((name, 'f8'))
    metrics = numpy.empty((region_ids.shape[0], ), dtype=arrayDtype)
    
    # Calculate the metrics
    metrics["CrownID"] = region_ids
    metrics["Height"] = region_heights
    for i,region_id in enumerate(region_ids):
        idx = regions == region_id
        metrics["Easting"][i] = numpy.mean(xgrid[idx])
        metrics["Northing"][i] = numpy.mean(ygrid[idx])       
        metrics["Area"][i] = numpy.sum(idx) * otherargs.resolution**2
        metrics["DiameterX"][i] = numpy.max(xgrid[idx]) - numpy.min(xgrid[idx])
        metrics["DiameterY"][i] = numpy.max(ygrid[idx]) - numpy.min(ygrid[idx])
    
    return metrics
    
    
def calcHeights(data):
    """
    Calculate canopy heights for all points
    Natural neighbour interpoliation is used to estimate ground elevation
    underneath nonground returns
    """
    # Interpolate the ground elevation of nonground points
    gndmask = data['CLASSIFICATION'] == generic.CLASSIFICATION_GROUND  
    gndz = pynninterp.NaturalNeighbourPts(data['X'][gndmask], data['Y'][gndmask], 
            data['Z'][gndmask], data[~gndmask,['X','Y']])
    
    # Calculate the heights    
    heights = numpy.zeros(data.shape[0], dtype=data['Z'].dtype)
    heights[gndmask] = data['Z'][~gndmask] - gndz
    
    return heights
    

@jit(nopython=True)
def findMaxHeights(data, heights, outImage, xMin, yMax, binSize):
    """
    Create a maximum canopy height grid
    """
    for i in range(data.shape[0]):
        row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                xMin, yMax, binSize)
        if (row >= 0) & (col >= 0) & (row < outImage.shape[0]) & (col < outImage.shape[1]):
            if outImage[row, col] != 0:
                if heights[i] < outImage[row, col]:
                    outImage[row, col] = heights[i]
            else:
                outImage[row, col] = heights[i]

                
def fillCHM(inImage, windowsize=3, minheight=2.0):
    """
    Fill within canopy gaps with the mean of surrounding canopy pixels heights
    Within canopy gaps are defined as pixels with a height less than a user defined
    threshold, but surrounded by at least 5 canopy heights greater than the threshold.
    The infill value is calculated as the mean value of canopy pixels. The neighborhood
    is defined by windowsize * 2 + 1
    """
    # Calculate the median heights
    offset = int((windowsize - 1) / 2)
    median_heights = ndimage.median_filter(inImage, size=windowsize, mode='mirror', 
            origin=[offset,offset])
    
    # Generate inputs to calculate the mean heights of canopy pixels
    weights = numpy.ones([windowsize,windowsize])
    cnt_heights = (inImage >= minheight).astype(inImage.dtype)
    ndimage.convolve(cnt_heights, weights=weights, mode='mirror', output=cnt_heights, 
            origin=[offset,offset])   
    sum_heights = numpy.where(inImage >= minheight, inImage, 0.0)
    ndimage.convolve(sum_heights, weights=weights, output=sum_heights, mode='mirror', 
            origin=[offset,offset])
    
    # Infill the CHM
    mask = (inImage < minheight) & (median_heights >= minheight)
    filled_chm = numpy.where(mask, sum_heights / cnt_heights, inImage)
    
    return filled_chm

    
def smoothCHM(inImage, windowsize=3):
    """
    Smooth the canopy height model. The neighborhood
    is defined by windowsize * 2 + 1
    """
    # Set the disk structuring element
    r = (windowsize - 1) / 2
    d = disk(r)
    
    # Smooth the CHM   
    offset = int(r)    
    smoothed_chm = numpy.where(inImage >= minheight, inImage, 0.0)
    ndimage.convolve(sum_heights, weights=d/numpy.sum(d), output=smoothed_chm, 
            mode='mirror', origin=[offset,offset])
    
    return smoothed_chm

    
def runWatershed(inImage, windowsize=3, minheight=2.0):
    """
    Run the scikit learn watershed algorithm
    Use local maximum peaks as input markers
    """
    # Generate the input markers
    d = numpy.ones((windowsize,windowsize))
    canopy_mask = inImage >= minheight
    local_maxi = peak_local_max(inImage, mask=canopy_mask,  
            footprint=d, indices=False)
    markers = ndimage.label(local_maxi)[0]
    
    # Run the watershed
    min_val = numpy.min(inImage)
    max_val = numpy.max(inImage)   
    tmpImage = max_val - ((255 + 0.9999) * (inImage - min_val)/(max_val - min_val))
    labels = watershed(tmpImage, markers, mask=canopy_mask)
    
    return labels

    
@jit(nopython=True)
def findProfilePeak(for_dip, bins_dip, peak_arr_x2, n_down=4, w_siz=2, second_peak_th_factor=0.06, smoothing_width=4):
    """
    Matthew Brolly's peak finding code
    for_dip = the profile
    bins_dip = the height bins
    peak_arr_x2 = the output array
    """
    normal_input = 1 - (for_dip / max(for_dip))    
    k = numpy.ones(smoothing_width) / smoothing_width
    temp_var = numpy.convol(normal_input, k)    
    max_peak = numpy.max(temp_var)
    max_peak_x, = numpy.where(temp_var == max_peak)
    peak_th = second_peak_th_factor * max_peak
    peak_arr = numpy.zeros(1)
    peak_arr_x = numpy.zeros(1,dtype=numpy.uint8)
    previous_w_peak = 0
    peak_assignment_flag = 0
    
    for x in range(0, temp_var.size-w_siz, w_siz):
        
        current_w_peak = numpy.max(temp_var[x:x+w_siz])
        
        if ((current_w_peak >= previous_w_peak) and (current_w_peak > peak_th)):
            previous_w_peak = current_w_peak
            previous_peak = current_w_peak
            previous_peak_x_tmp, = numpy.where(temp_var[x:x+w_siz] == current_w_peak)
            previous_peak_x = x + previous_peak_x_tmp
            peak_assignment_flag = 0
           
        if current_w_peak < previous_w_peak:
            previous_w_peak = current_w_peak
            peak_assignment_flag += 1

        if peak_assignment_flag == n_down:
            peak_arr = numpy.concatenate((peak_arr,previous_peak))
            peak_arr_x = numpy.concatenate((peak_arr_x,previous_peak_x))

        previous_w_peak = current_w_peak

    selected_i, = numpy.where(peak_arr > 0)
    
    if selected_i.size > 0:
        peak_arr2 = peak_arr[selected_i]
        peak_arr_x2 = peak_arr_x[selected_i]
    else:
        peak_arr_x2 = numpy.zeros(1)


def writeMetrics(metrics):
    """
    Write the crown metrics to a comma delimited ASCII file
    """
    pass

