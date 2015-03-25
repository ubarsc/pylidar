/*
 *  cgalinterp.cpp
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

#include "cgalinterp.h"

#include <math>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/algorithm.h>
#include <CGAL/Origin.h>
#include <CGAL/squared_distance_2.h>

namespace pylidar {
    
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::FT                                         CGALCoordType;
    typedef K::Vector_2                                   CGALVector;
    typedef K::Point_2                                    CGALPoint;
    
    typedef CGAL::Delaunay_triangulation_2<K>             DelaunayTriangulation;
    typedef CGAL::Interpolation_traits_2<K>               InterpTraits;
    typedef CGAL::Delaunay_triangulation_2<K>::Vertex_handle    Vertex_handle;
    typedef CGAL::Delaunay_triangulation_2<K>::Face_handle    Face_handle;
    
    typedef std::vector< std::pair<CGALPoint, CGALCoordType> >   CoordinateVector;
    typedef std::map<CGALPoint, CGALCoordType, K::Less_xy_2>     PointValueMap;
    
    double** interpGridNN(double *xVals, double *yVals, double *zVals, size_t nVals, double **xGrid, double **yGrid, size_t nXGrid, size_t nYGrid)throw(std::exception)
    {
        double **outVals = NULL;
        try
        {
            if(nVals < 3)
            {
                throw std::exception("Not enough points, need at least 3.");
            }
            else if(nVals < 100)
            {
                double meanX = 0;
                double meanY = 0;
                
                double varX = 0;
                double varY = 0;
                
                for(size_t i = 0; i < nVals; ++i)
                {
                    meanX += xVals[i];
                    meanY += yVals[i];
                }
                
                meanX = meanX / nVals;
                meanY = meanY / nVals;
                
                //std::cout << "meanX = " << meanX << std::endl;
                //std::cout << "meanY = " << meanY << std::endl;
                
                for(size_t i = 0; i < nVals; ++i)
                {
                    varX += xVals[i] - meanX;
                    varY += yVals[i] - meanY;
                }
                
                varX = fabs(varX / nVals);
                varY = fabs(varY / nVals);
                
                //std::cout << "varX = " << varX << std::endl;
                //std::cout << "varY = " << varX << std::endl;
                
                if((varX < 4) | (varY < 4))
                {
                    throw std::exception("Points are all within a line.");
                }
            }
            
            DelaunayTriangulation *dt = new DelaunayTriangulation();
            PointValueMap *values = new PointValueMap();
            
            for(size_t i = 0; i < nVals; ++i)
            {
                K::Point_2 cgalPt(xVals[i], yVals[i]);
                dt->insert(cgalPt);
                CGALCoordType value = zVals[i];
                values->insert(std::make_pair(cgalPt, value));
            }
            
            outVals = new double*[nYGrid];
            for(size_t i  = 0; i < nYGrid; ++i)
            {
                outVals[i] = new double*[nXGrid];
                for(size_t j = 0; j < nXGrid; ++j)
                {
                    
                    K::Point_2 p(xGrid[i][j], yGrid[i][j]);
                    CoordinateVector coords;
                    CGAL::Triple<std::back_insert_iterator<CoordinateVector>, K::FT, bool> result = CGAL::natural_neighbor_coordinates_2(*dt, p, std::back_inserter(coords));
                    if(!result.third)
                    {
                        // Not within convex hull of dataset
                        outVals[i][j] = 0.0;
                    }
                    else
                    {
                        CGALCoordType norm = result.second;
                        CGALCoordType outValue = CGAL::linear_interpolation(coords.begin(), coords.end(), norm, CGAL::Data_access<PointValueMap>(*this->values));
                        outVals[i][j] = outValue;
                    }
                }
            }
            
            delete dt;
            delete values;
        }
        catch(std::exception &e)
        {
            throw e;
        }
        return outVals;
    }
    

    double** interpGridPlaneFit(double *xVals, double *yVals, double *zVals, size_t nVals, double **xGrid, double **yGrid, size_t nXGrid, size_t nYGrid)throw(std::exception)
    {
        double **outVals = NULL;
        try
        {
            /*
            if(nVals < 3)
            {
                throw std::exception("Not enough points, need at least 3.");
            }
            else if(nVals < 100)
            {
                double meanX = 0;
                double meanY = 0;
                
                double varX = 0;
                double varY = 0;
                
                for(size_t i = 0; i < nVals; ++i)
                {
                    meanX += xVals[i];
                    meanY += yVals[i];
                }
                
                meanX = meanX / nVals;
                meanY = meanY / nVals;
                
                //std::cout << "meanX = " << meanX << std::endl;
                //std::cout << "meanY = " << meanY << std::endl;
                
                for(size_t i = 0; i < nVals; ++i)
                {
                    varX += xVals[i] - meanX;
                    varY += yVals[i] - meanY;
                }
                
                varX = fabs(varX / nVals);
                varY = fabs(varY / nVals);
                
                //std::cout << "varX = " << varX << std::endl;
                //std::cout << "varY = " << varX << std::endl;
                
                if((varX < 4) | (varY < 4))
                {
                    throw std::exception("Points are all within a line.");
                }
            }
            
            DelaunayTriangulation *dt = new DelaunayTriangulation();
            PointValueMap *values = new PointValueMap();
            
            for(size_t i = 0; i < nVals; ++i)
            {
                K::Point_2 cgalPt(xVals[i], yVals[i]);
                dt->insert(cgalPt);
                CGALCoordType value = zVals[i];
                values->insert(std::make_pair(cgalPt, value));
            }
            
            outVals = new double*[nYGrid];
            for(size_t i  = 0; i < nYGrid; ++i)
            {
                outVals[i] = new double*[nXGrid];
                for(size_t j = 0; j < nXGrid; ++j)
                {
                    outVals[i][j] = 0.0;
                }
            }
            
            delete dt;
            delete values;
            */
        }
        catch(std::exception &e)
        {
            throw e;
        }
        return outVals;
    }
    
    
}


#endif




