/*
 *  cgalinterp.h
 *
 *
 * This file is part of PyLidar
 * Copyright (C) 2015 John Armston, Neil Flood, Sam Gillingham and Pete Bunting
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PYLIDAR_cgalinterp_H
#define PYLIDAR_cgalinterp_H

#include <vector>


namespace pylidar {

    /** A function to interpolate a grid of values using natural neighbour */
    DllExport double** interpGridNN(double *xVals, double *yVals, double *zVals, size_t nVals, double **xGrid, double **yGrid, size_t nXGrid, size_t nYGrid)throw(std::exception);

    /** A function to interpolate a grid of values using plane fitting */
    DllExport double** interpGridPlaneFit(double *xVals, double *yVals, double *zVals, size_t nVals, double **xGrid, double **yGrid, size_t nXGrid, size_t nYGrid)throw(std::exception);

    
}


#endif




