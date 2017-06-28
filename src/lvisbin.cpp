/*
 * lvisbin.cpp
 *
 *
 * This file is part of PyLidar
 * Copyright (C) 2015 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
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

// adapted from http://lvis.gsfc.nasa.gov/utilities_home.html

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include "numpy/arrayobject.h"

#include "lvisbin.h"
#include "pylidar.h"
#include "pylvector.h"

#ifndef  GENLIB_LITTLE_ENDIAN
#define  GENLIB_LITTLE_ENDIAN 0x00
#endif

#ifndef  GENLIB_BIG_ENDIAN
#define  GENLIB_BIG_ENDIAN 0x01
#endif

// to determine how the pulses are found
#define POINT_FROM_LCE 0
#define POINT_FROM_LGE 1
#define POINT_FROM_LGW0 2
#define POINT_FROM_LGWEND 3

// for CVector
static const int nGrowBy = 10000;
static const int nInitSize = 256*256;

// For onld versions of VC
#if defined(_MSC_VER) && _MSC_VER < 1900
    #define isnan(x) _isnan(x)
#endif

/* define GENLIB_OUR_ENDIAN depending on endian setting
 from pyconfig.h */
#if WORDS_BIGENDIAN == 1
    #define GENLIB_OUR_ENDIAN GENLIB_BIG_ENDIAN
#else
    #define GENLIB_OUR_ENDIAN GENLIB_LITTLE_ENDIAN
#endif


#ifndef  LVIS_VERSION_COUNT
#define  LVIS_VERSION_COUNT 5  // 1.00 -> 1.04 is 5 total versions
#endif

#ifndef  VERSION_TESTBLOCK_LENGTH
#define  VERSION_TESTBLOCK_LENGTH (128 * 1024) // 128k should be enough to figure it out
#endif

// if you want a little more info to start...uncomment this
/* #define DEBUG_ON */

// prototypes we need
double host_double(double input_double,int host_endian);
float  host_float(float input_float,int host_endian);

// taken from lvis_release_reader.c. See https://lvis.gsfc.nasa.gov/
// self contained file version detection routine
int detect_release_version(char * filename, int * fileType, float * fileVersion, int myendian)
{
   // any misalignment of the data structure will result in HUGE double values
   // read in a block of data and sum up the abs() of some doubles and pick
   // the lowest value... that will match the version / type.
  
   struct lvis_lce_v1_00 * lce100; struct lvis_lge_v1_00 * lge100; struct lvis_lgw_v1_00 * lgw100;
   struct lvis_lce_v1_01 * lce101; struct lvis_lge_v1_01 * lge101; struct lvis_lgw_v1_01 * lgw101;
   struct lvis_lce_v1_02 * lce102; struct lvis_lge_v1_02 * lge102; struct lvis_lgw_v1_02 * lgw102;
   struct lvis_lce_v1_03 * lce103; struct lvis_lge_v1_03 * lge103; struct lvis_lgw_v1_03 * lgw103;
   struct lvis_lce_v1_04 * lce104; struct lvis_lge_v1_04 * lge104; struct lvis_lgw_v1_04 * lgw104;

   long          fileSize;
   int           i,mintype,type,minversion,version,status=0,maxpackets;
   size_t        maxsize;
   FILE          *fptest=NULL;
   double        testTotal;
   double        testTotalMin = (double) 0.0;
   double        testTotalsLat[3][LVIS_VERSION_COUNT];  // 3 file types and X versions to test
   double        testTotalsLon[3][LVIS_VERSION_COUNT];  // 3 file types and X versions to test
   double        testTotalsAlt[3][LVIS_VERSION_COUNT];  // 3 file types and X versions to test
   unsigned char testBlock[VERSION_TESTBLOCK_LENGTH];
   
   lce100 = (struct lvis_lce_v1_00 * ) testBlock;
   lce101 = (struct lvis_lce_v1_01 * ) testBlock;
   lce102 = (struct lvis_lce_v1_02 * ) testBlock;
   lce103 = (struct lvis_lce_v1_03 * ) testBlock;
   lce104 = (struct lvis_lce_v1_04 * ) testBlock;
   lge100 = (struct lvis_lge_v1_00 * ) testBlock;
   lge101 = (struct lvis_lge_v1_01 * ) testBlock;
   lge102 = (struct lvis_lge_v1_02 * ) testBlock;
   lge103 = (struct lvis_lge_v1_03 * ) testBlock;
   lge104 = (struct lvis_lge_v1_04 * ) testBlock;
   lgw100 = (struct lvis_lgw_v1_00 * ) testBlock;
   lgw101 = (struct lvis_lgw_v1_01 * ) testBlock;
   lgw102 = (struct lvis_lgw_v1_02 * ) testBlock;
   lgw103 = (struct lvis_lgw_v1_03 * ) testBlock;
   lgw104 = (struct lvis_lgw_v1_04 * ) testBlock;
   
   // determine the biggest packet, which limits how many to read (to be fair)
   maxsize = 1;
   if(sizeof(struct lvis_lce_v1_00) > maxsize) maxsize = sizeof(struct lvis_lce_v1_00);
   if(sizeof(struct lvis_lce_v1_01) > maxsize) maxsize = sizeof(struct lvis_lce_v1_01);
   if(sizeof(struct lvis_lce_v1_02) > maxsize) maxsize = sizeof(struct lvis_lce_v1_02);
   if(sizeof(struct lvis_lce_v1_03) > maxsize) maxsize = sizeof(struct lvis_lce_v1_03);
   if(sizeof(struct lvis_lce_v1_04) > maxsize) maxsize = sizeof(struct lvis_lce_v1_04);
   if(sizeof(struct lvis_lge_v1_00) > maxsize) maxsize = sizeof(struct lvis_lge_v1_00);
   if(sizeof(struct lvis_lge_v1_01) > maxsize) maxsize = sizeof(struct lvis_lge_v1_01);
   if(sizeof(struct lvis_lge_v1_02) > maxsize) maxsize = sizeof(struct lvis_lge_v1_02);
   if(sizeof(struct lvis_lge_v1_03) > maxsize) maxsize = sizeof(struct lvis_lge_v1_03);
   if(sizeof(struct lvis_lge_v1_04) > maxsize) maxsize = sizeof(struct lvis_lge_v1_04);
   if(sizeof(struct lvis_lgw_v1_00) > maxsize) maxsize = sizeof(struct lvis_lgw_v1_00);
   if(sizeof(struct lvis_lgw_v1_01) > maxsize) maxsize = sizeof(struct lvis_lgw_v1_01);
   if(sizeof(struct lvis_lgw_v1_02) > maxsize) maxsize = sizeof(struct lvis_lgw_v1_02);
   if(sizeof(struct lvis_lgw_v1_03) > maxsize) maxsize = sizeof(struct lvis_lgw_v1_03);
   if(sizeof(struct lvis_lgw_v1_04) > maxsize) maxsize = sizeof(struct lvis_lgw_v1_04);
      
   // open up the file for read access
   if((fptest = fopen((char *)filename,"rb"))==NULL)
     {
	fprintf(stderr,"Error opening the input file: %s\n",filename);
	exit(-1);
     }

   fileSize = VERSION_TESTBLOCK_LENGTH;
   status=fread(testBlock,1,sizeof(testBlock),fptest);
   // see if the file wasn't big enough
   if( (status!=sizeof(testBlock)) && (status>0)) fileSize = status;
   maxpackets = fileSize  / maxsize;

#ifdef DEBUG_ON  // uncomment the #define up top if you want to see these messages
   fprintf(stdout,"maxpackets = %d\n",maxpackets);
#endif
      
   // clear out the totals and our min variables
   mintype = 0; minversion = 0;
   for(type=0;type<3;type++)
     for( version=0; version<LVIS_VERSION_COUNT ; version++)
       {	  
	  testTotalsLat[type][version] = 0.0;
	  testTotalsLon[type][version] = 0.0;
	  testTotalsAlt[type][version] = 0.0;	  
       }
   
   if(status == fileSize)
     {     	
	for(i=0;i<maxpackets;i++)
	  for(type=0;type<3;type++)
	    for( version=0; version<LVIS_VERSION_COUNT ; version++)
	      {
		 if(type==0 && version==0)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lce100[i].tlat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lce100[i].tlon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lce100[i].zt,myendian));
		   }
		 if(type==0 && version==1)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lce101[i].tlat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lce101[i].tlon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lce101[i].zt,myendian));		      
		   }
		 if(type==0 && version==2)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lce102[i].tlat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lce102[i].tlon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lce102[i].zt,myendian));
		   }
		 if(type==0 && version==3)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lce103[i].tlat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lce103[i].tlon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lce103[i].zt,myendian));		      
		   }
		 if(type==0 && version==4)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lce104[i].tlat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lce104[i].tlon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lce104[i].zt,myendian));		      
		   }
		 if(type==1 && version==0)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lge100[i].glat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lge100[i].glon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lge100[i].zg,myendian));
		   }
		 if(type==1 && version==1)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lge101[i].glat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lge101[i].glon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lge101[i].zg,myendian));		      
		   }
		 if(type==1 && version==2)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lge102[i].glat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lge102[i].glon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lge102[i].zg,myendian));
		   }
		 if(type==1 && version==3)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lge103[i].glat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lge103[i].glon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lge103[i].zg,myendian));		      
		   }
		 if(type==1 && version==4)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lge104[i].glat,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lge104[i].glon,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lge104[i].zg,myendian));		      
		   }
		 if(type==2 && version==0)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lgw100[i].lat0,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lgw100[i].lon0,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lgw100[i].z0,myendian));
		   }
		 if(type==2 && version==1)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lgw101[i].lat0,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lgw101[i].lon0,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lgw101[i].z0,myendian));		      
		   }
		 if(type==2 && version==2)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lgw102[i].lat0,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lgw102[i].lon0,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lgw102[i].z0,myendian));
		   }
		 if(type==2 && version==3)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lgw103[i].lat0,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lgw103[i].lon0,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lgw103[i].z0,myendian));		      
		   }
		 if(type==2 && version==4)
		   {
		      testTotalsLat[type][version] += fabs(host_double(lgw104[i].lat0,myendian));
		      testTotalsLon[type][version] += fabs(host_double(lgw104[i].lon0,myendian));
		      testTotalsAlt[type][version] += fabsf(host_float(lgw104[i].z0,myendian));
		   }
	      }

	mintype = 0 ; minversion = 0;
	for(type=0;type<3;type++)
	  for( version=0; version<LVIS_VERSION_COUNT; version++ )
	      {
		 testTotal = testTotalsLat[type][version] +
		   testTotalsLon[type][version] +
		   testTotalsAlt[type][version];
		 // if we have not yet set a minimum and our new number is a real number
		 if( (testTotalMin == (double) 0.0) && isnan(testTotal)==0)
		   {     
		      testTotalMin = testTotal;
#ifdef DEBUG_ON  // uncomment the #define up top if you want to see these messages
		      fprintf(stdout,"SET A MIN! type = %d version = 1.0%1d = %f\n",type,version,testTotal);
#endif
		   }
		 else
		   if(testTotal < testTotalMin && isnan(testTotal)==0) 
		     {
			mintype = type;
			minversion = version;
			testTotalMin = testTotal;
#ifdef DEBUG_ON  // uncomment the #define up top if you want to see these messages
			fprintf(stdout,"SET A NEW MIN! type = %d version = 1.0%1d = %f\n",type,version,testTotal);
#endif
		     }
		 
#ifdef DEBUG_ON  // uncomment the #define up top if you want to see these messages
		 fprintf(stdout,"type = %d version = 1.0%1d = %f\n",type,version,testTotal);
#endif
	      }
	
#ifdef DEBUG_ON  // uncomment the #define up top if you want to see these messages
	fprintf(stdout,"SOLVED:  type = %d version = 1.0%1d\n",mintype,minversion);
#endif
     }
   
   if(fptest!=NULL) fclose(fptest);

   if(mintype == 0) *fileType = LVIS_RELEASE_FILETYPE_LCE;
   if(mintype == 1) *fileType = LVIS_RELEASE_FILETYPE_LGE;
   if(mintype == 2) *fileType = LVIS_RELEASE_FILETYPE_LGW;
   
   if(minversion == 0) *fileVersion = ((float)1.00);
   if(minversion == 1) *fileVersion = ((float)1.01);
   if(minversion == 2) *fileVersion = ((float)1.02);
   if(minversion == 3) *fileVersion = ((float)1.03);
   if(minversion == 4) *fileVersion = ((float)1.04);
   
   return status;
}

double host_double(double input_double,int host_endian)
{
   double return_value=0.0;
   unsigned char * inptr, * outptr;
   
   inptr = (unsigned char *) &input_double;
   outptr = (unsigned char *) &return_value;
   
   if(host_endian == GENLIB_LITTLE_ENDIAN)
     {
	outptr[0] = inptr[7];
	outptr[1] = inptr[6];
	outptr[2] = inptr[5];
	outptr[3] = inptr[4];
	outptr[4] = inptr[3];
	outptr[5] = inptr[2];
	outptr[6] = inptr[1];
	outptr[7] = inptr[0];
     }
   
   if(host_endian == GENLIB_BIG_ENDIAN)
     return input_double;
   else
     return return_value;
}

float host_float(float input_float,int host_endian)
{
   float return_value=0.0;
   unsigned char * inptr, * outptr;
   
   inptr = (unsigned char *) &input_float;
   outptr = (unsigned char *) &return_value;
   
   if(host_endian == GENLIB_LITTLE_ENDIAN)
     {
	outptr[0] = inptr[3];
	outptr[1] = inptr[2];
	outptr[2] = inptr[1];
	outptr[3] = inptr[0];
    }
   
   if(host_endian == GENLIB_BIG_ENDIAN)
     return input_float;
   else
     return return_value;
}

uint32_t host_dword(uint32_t input_dword,int host_endian)
{
   uint32_t return_value=0;
   unsigned char * inptr, * outptr;
   
   inptr = (unsigned char *) &input_dword;
   outptr = (unsigned char *) &return_value;
   
   if(host_endian == GENLIB_LITTLE_ENDIAN)
     {
	outptr[0] = inptr[3];
	outptr[1] = inptr[2];
	outptr[2] = inptr[1];
	outptr[3] = inptr[0];
    }
   
   if(host_endian == GENLIB_BIG_ENDIAN)
     return input_dword;
   else
     return return_value;
}

uint16_t host_word(uint16_t input_word,int host_endian)
{
   uint16_t return_value=0;
   unsigned char * inptr, * outptr;
   
   inptr = (unsigned char *) &input_word;
   outptr = (unsigned char *) &return_value;
   
   if(host_endian == GENLIB_LITTLE_ENDIAN)
     {
	outptr[0] = inptr[1];
	outptr[1] = inptr[0];
    }
   if(host_endian == GENLIB_BIG_ENDIAN)
     return input_word;
   else
     return return_value;
}

/* An exception object for this module */
/* created in the init function */
struct LVISBinState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct LVISBinState*)PyModule_GetState(m))
#define GETSTATE_FC GETSTATE(PyState_FindModule(&moduledef))
#else
#define GETSTATE(m) (&_state)
#define GETSTATE_FC (&_state)
static struct LVISBinState _state;
#endif

/* Pulses */
typedef struct {
    float lceVersion;
    float lgeVersion;
    float lgwVersion;

    // LCE - see lvis.h
    uint32_t lce_lfid; // >= v1.01
    uint32_t lce_shotnumber; // >= v1.01
    float lce_azimuth; // >= v1.03
    float lce_incidentangle; // >= v1.03
    float lce_range; // >= v1.03
    double lce_lvistime; // >= 1.02
    double lce_tlon;
    double lce_tlat;
    float  lce_zt;

    // LGE
    uint32_t lge_lfid; // >= v1.01
    uint32_t lge_shotnumber; // >= v1.01
    float lge_azimuth; // >= v1.03
    float lge_incidentangle; // >= v1.03
    float lge_range; // >= v1.03
    double lge_lvistime; // >= v1.02
    double lge_glon;
    double lge_glat;
    float  lge_zg;
    float  lge_rh25;
    float  lge_rh50;
    float  lge_rh75;
    float  lge_rh100;

    // LGW
    uint32_t lgw_lfid; // >= v1.01
    uint32_t lgw_shotnumber; // >= v1.01
    float lgw_azimuth; // >= v1.03
    float lgw_incidentangle; // >= v1.03
    float lgw_range; // >= v1.03
    double lgw_lvistime; // >= v1.02
    double lgw_lon0;
    double lgw_lat0;
    float  lgw_z0;
    double lgw_lonend;
    double lgw_latend;
    float  lgw_zend;
    float  lgw_sigmean;

    npy_uint32 wfm_start_idx;
    npy_uint8 number_of_waveform_samples;
    npy_uint8 number_of_returns;
    npy_uint64 pts_start_idx;
} SLVISPulse;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn LVISPulseFields[] = {
    CREATE_FIELD_DEFN(SLVISPulse, lceVersion, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgeVersion, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgwVersion, 'f'),

    CREATE_FIELD_DEFN(SLVISPulse, lce_lfid, 'u'),
    CREATE_FIELD_DEFN(SLVISPulse, lce_shotnumber, 'u'),
    CREATE_FIELD_DEFN(SLVISPulse, lce_azimuth, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lce_incidentangle, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lce_range, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lce_lvistime, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lce_tlon, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lce_tlat, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lce_zt, 'f'),

    CREATE_FIELD_DEFN(SLVISPulse, lge_lfid, 'u'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_shotnumber, 'u'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_azimuth, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_incidentangle, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_range, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_lvistime, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_glon, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_glat, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_zg, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_rh25, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_rh50, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_rh75, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lge_rh100, 'f'),

    // LGW
    CREATE_FIELD_DEFN(SLVISPulse, lgw_lfid, 'u'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_shotnumber, 'u'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_azimuth, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_incidentangle, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_range, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_lvistime, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_lon0, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_lat0, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_z0, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_lonend, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_latend, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_zend, 'f'),
    CREATE_FIELD_DEFN(SLVISPulse, lgw_sigmean, 'f'),

    CREATE_FIELD_DEFN(SLVISPulse, wfm_start_idx, 'u'),
    CREATE_FIELD_DEFN(SLVISPulse, number_of_waveform_samples, 'u'),    
    CREATE_FIELD_DEFN(SLVISPulse, number_of_returns, 'u'),
    CREATE_FIELD_DEFN(SLVISPulse, pts_start_idx, 'u'),

    {NULL} // Sentinel
};

/* Structure for waveform Info */
typedef struct {
    npy_uint16 number_of_waveform_received_bins;
    npy_uint64 received_start_idx;
    npy_uint16 number_of_waveform_transmitted_bins;
    npy_uint64 transmitted_start_idx;
    float receive_wave_gain;
    float receive_wave_offset;
    float trans_wave_gain;
    float trans_wave_offset;
} SLVISWaveformInfo;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn LVISWaveformInfoFields[] = {
    CREATE_FIELD_DEFN(SLVISWaveformInfo, number_of_waveform_received_bins, 'u'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, received_start_idx, 'u'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, number_of_waveform_transmitted_bins, 'u'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, transmitted_start_idx, 'u'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, receive_wave_gain, 'f'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, receive_wave_offset, 'f'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, trans_wave_gain, 'f'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, trans_wave_offset, 'f'),
    {NULL} // Sentinel
};

typedef struct {
    double x;
    double y;
    float z;
    npy_uint8 classification;
} SLVISPoint;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn LVISPointFields[] = {
    CREATE_FIELD_DEFN(SLVISPoint, x, 'f'),
    CREATE_FIELD_DEFN(SLVISPoint, y, 'f'),
    CREATE_FIELD_DEFN(SLVISPoint, z, 'f'),
    CREATE_FIELD_DEFN(SLVISPoint, classification, 'u'),
    {NULL} // Sentinel
};

/* Python object for reading LVIS LCE/LGE/LGW files */
typedef struct
{
    PyObject_HEAD
    FILE *fpLCE;
    FILE *fpLGE;
    FILE *fpLGW;
    float lceVersion;
    float lgeVersion;
    float lgwVersion;
    bool bFinished;
    bool bRecordsRead; // so we know whether to set bIgnore fields
    SpylidarFieldDefn *pPulseDefn; // we take a copy of LVISPulseFields so we can set bIgnore
    int nPointFrom; // POINT_FROM_LCE etc
} PyLVISFiles;

#if PY_MAJOR_VERSION >= 3
static int lvisbin_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int lvisbin_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_lvisbin",
        NULL,
        sizeof(struct LVISBinState),
        NULL,
        NULL,
        lvisbin_traverse,
        lvisbin_clear,
        NULL
};
#endif

/* destructor - close and delete tc */
static void 
PyLVISFiles_dealloc(PyLVISFiles *self)
{
    if( self->fpLCE != NULL )
    {
        fclose(self->fpLCE);
        self->fpLCE = NULL;
    }
    if( self->fpLGE != NULL )
    {
        fclose(self->fpLGE);
        self->fpLGE = NULL;
    }
    if( self->fpLGW != NULL )
    {
        fclose(self->fpLGW);
        self->fpLGW = NULL;
    }
    free(self->pPulseDefn);
    self->pPulseDefn = NULL;

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int 
PyLVISFiles_init(PyLVISFiles *self, PyObject *args, PyObject *kwds)
{
char *pszLCE_Fname = NULL, *pszLGE_Fname = NULL, *pszLGW_Fname = NULL;
int nPointFrom;

    if( !PyArg_ParseTuple(args, "zzzi", &pszLCE_Fname, &pszLGE_Fname, &pszLGW_Fname, &nPointFrom ) )
    {
        return -1;
    }

    if( ( nPointFrom < POINT_FROM_LCE) || ( nPointFrom > POINT_FROM_LGWEND ) )
    {
        PyErr_SetString(GETSTATE_FC->error, "nPointFrom out of range");
        return -1;
    }

    self->fpLCE = NULL;
    self->fpLGE = NULL;
    self->fpLGW = NULL;
    self->lceVersion = 0;
    self->lgeVersion = 0;
    self->lgwVersion = 0;
    self->bFinished = false;
    self->bRecordsRead = false;
    self->nPointFrom = nPointFrom;

    // take a copy of LVISPulseFields so we can fiddle about with 
    // the bIgnore field without breaking other instances of this class
    // Note: this is a shallow copy (does not copy pszName) but I think this is ok
    self->pPulseDefn = (SpylidarFieldDefn*)malloc(sizeof(LVISPulseFields));
    memcpy(self->pPulseDefn, LVISPulseFields, sizeof(LVISPulseFields));
    
    // set all to ignore and then we selectively turn them on
    // when we use them. 
    int i = 0;
    while( self->pPulseDefn[i].pszName != NULL )
    {
        self->pPulseDefn[i].bIgnore = 1;
        i++;
    }
    // these ones are always used
    pylidar_setIgnore(self->pPulseDefn, "lceVersion", 0);
    pylidar_setIgnore(self->pPulseDefn, "lgeVersion", 0);
    pylidar_setIgnore(self->pPulseDefn, "lgwVersion", 0);
    pylidar_setIgnore(self->pPulseDefn, "wfm_start_idx", 0);
    pylidar_setIgnore(self->pPulseDefn, "number_of_waveform_samples", 0);
    pylidar_setIgnore(self->pPulseDefn, "number_of_returns", 0);
    pylidar_setIgnore(self->pPulseDefn, "pts_start_idx", 0);

    int nFileType;

    if( pszLCE_Fname != NULL )
    {
        self->fpLCE = fopen(pszLCE_Fname, "rb");
        if( self->fpLCE == NULL )
        {
            PyErr_Format(GETSTATE_FC->error, "Cannot open %s", pszLCE_Fname);
            return -1;
        }

        nFileType = 999;
        detect_release_version(pszLCE_Fname, &nFileType, &self->lceVersion, GENLIB_OUR_ENDIAN);
        if( nFileType != LVIS_RELEASE_FILETYPE_LCE )
        {
            PyErr_Format(GETSTATE_FC->error, "File %s does not contain LCE data", pszLCE_Fname);
            fclose(self->fpLCE);
            self->fpLCE = NULL;
            return -1;
        }
    }

    if( pszLGE_Fname != NULL )
    {
        self->fpLGE = fopen(pszLGE_Fname, "rb");
        if( self->fpLGE == NULL )
        {
            if( self->fpLCE != NULL )
            {
                fclose(self->fpLCE);
                self->fpLCE = NULL;
            }
            PyErr_Format(GETSTATE_FC->error, "Cannot open %s", pszLGE_Fname);
            return -1;
        }

        nFileType = 999;
        detect_release_version(pszLGE_Fname, &nFileType, &self->lgeVersion, GENLIB_OUR_ENDIAN);
        if( nFileType != LVIS_RELEASE_FILETYPE_LGE )
        {
            PyErr_Format(GETSTATE_FC->error, "File %s does not contain LGE data", pszLGE_Fname);
            fclose(self->fpLGE);
            self->fpLGE = NULL;
            if( self->fpLCE != NULL )
            {
                fclose(self->fpLCE);
                self->fpLCE = NULL;
            }
            return -1;
        }
    }

    if( pszLGW_Fname != NULL )
    {
        self->fpLGW = fopen(pszLGW_Fname, "rb");
        if( self->fpLGW == NULL )
        {
            if( self->fpLCE != NULL )
            {
                fclose(self->fpLCE);
                self->fpLCE = NULL;
            }
            if( self->fpLGE != NULL )
            {
                fclose(self->fpLGE);
                self->fpLGE = NULL;
            }
            PyErr_Format(GETSTATE_FC->error, "Cannot open %s", pszLGW_Fname);
            return -1;
        }

        nFileType = 999;
        detect_release_version(pszLGW_Fname, &nFileType, &self->lgwVersion, GENLIB_OUR_ENDIAN);
        if( nFileType != LVIS_RELEASE_FILETYPE_LGW )
        {
            PyErr_Format(GETSTATE_FC->error, "File %s does not contain LGE data", pszLGE_Fname);
            fclose(self->fpLGW);
            self->fpLGW = NULL;
            if( self->fpLCE != NULL )
            {
                fclose(self->fpLCE);
                self->fpLCE = NULL;
            }
            if( self->fpLGE != NULL )
            {
                fclose(self->fpLGE);
                self->fpLGE = NULL;
            }
            return -1;
        }
    }

    return 0;
}

size_t getStructSize(PyLVISFiles *self, int fileType)
{
    if( fileType == LVIS_RELEASE_FILETYPE_LCE )
    {
        if(self->lceVersion == ((float)1.00))
            return sizeof(struct lvis_lce_v1_00);
        if(self->lceVersion == ((float)1.01))
            return sizeof(struct lvis_lce_v1_01);
        if(self->lceVersion == ((float)1.02))
            return sizeof(struct lvis_lce_v1_02);
        if(self->lceVersion == ((float)1.03))
            return sizeof(struct lvis_lce_v1_03);
        if(self->lceVersion == ((float)1.04))
            return sizeof(struct lvis_lce_v1_04);
    }

    if( fileType == LVIS_RELEASE_FILETYPE_LGE )
    {
        if(self->lgeVersion == ((float)1.00))
            return sizeof(struct lvis_lge_v1_00);
        if(self->lgeVersion == ((float)1.01))
            return sizeof(struct lvis_lge_v1_01);
        if(self->lgeVersion == ((float)1.02))
            return sizeof(struct lvis_lge_v1_02);
        if(self->lgeVersion == ((float)1.03))
            return sizeof(struct lvis_lge_v1_03);
        if(self->lgeVersion == ((float)1.04))
            return sizeof(struct lvis_lge_v1_04);
    }

    if( fileType == LVIS_RELEASE_FILETYPE_LGW )
    {
        if(self->lgwVersion == ((float)1.00))
            return sizeof(struct lvis_lgw_v1_00);
        if(self->lgwVersion == ((float)1.01))
            return sizeof(struct lvis_lgw_v1_01);
        if(self->lgwVersion == ((float)1.02))
            return sizeof(struct lvis_lgw_v1_02);
        if(self->lgwVersion == ((float)1.03))
            return sizeof(struct lvis_lgw_v1_03);
        if(self->lgwVersion == ((float)1.04))
            return sizeof(struct lvis_lgw_v1_04);
    }

    // should never get here...
    fprintf(stderr, "Failed to find size in getStructSize()\n");
    return 0;
}

void saveLCEToStruct(float lceVersion, char *pLCERecord, SLVISPulse *plvisPulse, SLVISPoint *plvisPoint, 
            SpylidarFieldDefn *pLVISPulseFields, bool bRecordsRead)
{
    if(lceVersion == ((float)1.00))
    {
        struct lvis_lce_v1_00 *p = reinterpret_cast<struct lvis_lce_v1_00*>(pLCERecord);
        plvisPulse->lce_tlon = p->tlon;
        plvisPulse->lce_tlat = p->tlat;
        plvisPulse->lce_zt = p->zt;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lce_tlon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_zt", 0);
        }
    }
    else if(lceVersion == ((float)1.01))
    {
        struct lvis_lce_v1_01 *p = reinterpret_cast<struct lvis_lce_v1_01*>(pLCERecord);
        plvisPulse->lce_lfid = p->lfid;
        plvisPulse->lce_shotnumber = p->shotnumber;
        plvisPulse->lce_tlon = p->tlon;
        plvisPulse->lce_tlat = p->tlat;
        plvisPulse->lce_zt = p->zt;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lce_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_zt", 0);
        }
    }
    else if(lceVersion == ((float)1.02))
    {
        struct lvis_lce_v1_02 *p = reinterpret_cast<struct lvis_lce_v1_02*>(pLCERecord);
        plvisPulse->lce_lfid = p->lfid;
        plvisPulse->lce_shotnumber = p->shotnumber;
        plvisPulse->lce_lvistime = p->lvistime;
        plvisPulse->lce_tlon = p->tlon;
        plvisPulse->lce_tlat = p->tlat;
        plvisPulse->lce_zt = p->zt;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lce_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_zt", 0);
        }
    }
    else if(lceVersion == ((float)1.03))
    {
        struct lvis_lce_v1_03 *p = reinterpret_cast<struct lvis_lce_v1_03*>(pLCERecord);
        plvisPulse->lce_lfid = p->lfid;
        plvisPulse->lce_shotnumber = p->shotnumber;
        plvisPulse->lce_lvistime = p->lvistime;
        plvisPulse->lce_azimuth = p->azimuth;
        plvisPulse->lce_incidentangle = p->incidentangle;
        plvisPulse->lce_range = p->range;
        plvisPulse->lce_tlon = p->tlon;
        plvisPulse->lce_tlat = p->tlat;
        plvisPulse->lce_zt = p->zt;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lce_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_azimuth", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_incidentangle", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_range", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_zt", 0);
        }
    }
    else if(lceVersion == ((float)1.04))
    {
        // Note: same fields as v1.03
        struct lvis_lce_v1_04 *p = reinterpret_cast<struct lvis_lce_v1_04*>(pLCERecord);
        plvisPulse->lce_lfid = p->lfid;
        plvisPulse->lce_shotnumber = p->shotnumber;
        plvisPulse->lce_lvistime = p->lvistime;
        plvisPulse->lce_azimuth = p->azimuth;
        plvisPulse->lce_incidentangle = p->incidentangle;
        plvisPulse->lce_range = p->range;
        plvisPulse->lce_tlon = p->tlon;
        plvisPulse->lce_tlat = p->tlat;
        plvisPulse->lce_zt = p->zt;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lce_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_azimuth", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_incidentangle", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_range", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_tlat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lce_zt", 0);
        }
    }

    if( plvisPoint != NULL )
    {
        // set the point from LCE
        plvisPoint->x = plvisPulse->lce_tlon;
        plvisPoint->y = plvisPulse->lce_tlat;
        plvisPoint->z = plvisPulse->lce_zt;
    }
}

void saveLGEToStruct(float lgeVersion, char *pLGERecord, SLVISPulse *plvisPulse, SLVISPoint *plvisPoint, 
            SpylidarFieldDefn *pLVISPulseFields, bool bRecordsRead)
{
    if(lgeVersion == ((float)1.00))
    {
        struct lvis_lge_v1_00 *p = reinterpret_cast<struct lvis_lge_v1_00 *>(pLGERecord);
        plvisPulse->lge_glon = p->glon;
        plvisPulse->lge_glat = p->glat;
        plvisPulse->lge_zg = p->zg;
        plvisPulse->lge_rh25 = p->rh25;
        plvisPulse->lge_rh50 = p->rh50;
        plvisPulse->lge_rh75 = p->rh75;
        plvisPulse->lge_rh100 = p->rh100;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lge_glon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_zg", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh25", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh50", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh75", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh100", 0);
        }
    }
    else if(lgeVersion == ((float)1.01))
    {
        struct lvis_lge_v1_01 *p = reinterpret_cast<struct lvis_lge_v1_01 *>(pLGERecord);
        plvisPulse->lge_lfid = p->lfid;
        plvisPulse->lge_shotnumber = p->shotnumber;
        plvisPulse->lge_glon = p->glon;
        plvisPulse->lge_glat = p->glat;
        plvisPulse->lge_zg = p->zg;
        plvisPulse->lge_rh25 = p->rh25;
        plvisPulse->lge_rh50 = p->rh50;
        plvisPulse->lge_rh75 = p->rh75;
        plvisPulse->lge_rh100 = p->rh100;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lge_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_zg", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh25", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh50", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh75", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh100", 0);
        }
    }
    else if(lgeVersion == ((float)1.02))
    {
        struct lvis_lge_v1_02 *p = reinterpret_cast<struct lvis_lge_v1_02 *>(pLGERecord);
        plvisPulse->lge_lfid = p->lfid;
        plvisPulse->lge_shotnumber = p->shotnumber;
        plvisPulse->lge_lvistime = p->lvistime;
        plvisPulse->lge_glon = p->glon;
        plvisPulse->lge_glat = p->glat;
        plvisPulse->lge_zg = p->zg;
        plvisPulse->lge_rh25 = p->rh25;
        plvisPulse->lge_rh50 = p->rh50;
        plvisPulse->lge_rh75 = p->rh75;
        plvisPulse->lge_rh100 = p->rh100;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lge_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_zg", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh25", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh50", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh75", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh100", 0);
        }
    }
    else if(lgeVersion == ((float)1.03))
    {
        struct lvis_lge_v1_03 *p = reinterpret_cast<struct lvis_lge_v1_03 *>(pLGERecord);
        plvisPulse->lge_lfid = p->lfid;
        plvisPulse->lge_shotnumber = p->shotnumber;
        plvisPulse->lge_azimuth = p->azimuth;
        plvisPulse->lge_incidentangle = p->incidentangle;
        plvisPulse->lge_range = p->range;
        plvisPulse->lge_lvistime = p->lvistime;
        plvisPulse->lge_glon = p->glon;
        plvisPulse->lge_glat = p->glat;
        plvisPulse->lge_zg = p->zg;
        plvisPulse->lge_rh25 = p->rh25;
        plvisPulse->lge_rh50 = p->rh50;
        plvisPulse->lge_rh75 = p->rh75;
        plvisPulse->lge_rh100 = p->rh100;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lge_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_azimuth", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_incidentangle", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_range", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_zg", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh25", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh50", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh75", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh100", 0);
        }
    }
    else if(lgeVersion == ((float)1.04))
    {
        // Note: fields are the same as v1.03
        struct lvis_lge_v1_04 *p = reinterpret_cast<struct lvis_lge_v1_04 *>(pLGERecord);
        plvisPulse->lge_lfid = p->lfid;
        plvisPulse->lge_shotnumber = p->shotnumber;
        plvisPulse->lge_azimuth = p->azimuth;
        plvisPulse->lge_incidentangle = p->incidentangle;
        plvisPulse->lge_range = p->range;
        plvisPulse->lge_lvistime = p->lvistime;
        plvisPulse->lge_glon = p->glon;
        plvisPulse->lge_glat = p->glat;
        plvisPulse->lge_zg = p->zg;
        plvisPulse->lge_rh25 = p->rh25;
        plvisPulse->lge_rh50 = p->rh50;
        plvisPulse->lge_rh75 = p->rh75;
        plvisPulse->lge_rh100 = p->rh100;
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lge_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_azimuth", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_incidentangle", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_range", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glon", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_glat", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_zg", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh25", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh50", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh75", 0);
            pylidar_setIgnore(pLVISPulseFields, "lge_rh100", 0);
        }
    }

    if( plvisPoint != NULL )
    {
        // set the point from LGE
        plvisPoint->x = plvisPulse->lge_glon;
        plvisPoint->y = plvisPulse->lge_glat;
        plvisPoint->z = plvisPulse->lge_zg;
    }
}

void saveLGWToStruct(float lgwVersion, char *pLGWRecord, SLVISWaveformInfo *plvisWaveformInfo, 
        SLVISPulse *plvisPulse, SLVISPoint *plvisPoint, pylidar::CVector<uint16_t> *pTransmitted, 
        pylidar::CVector<uint16_t> *pReceived, 
        SpylidarFieldDefn *pLVISPulseFields, bool bRecordsRead, int nPointFrom)
{
    if(lgwVersion == ((float)1.00))
    {
        struct lvis_lgw_v1_00 *p = reinterpret_cast<struct lvis_lgw_v1_00*>(pLGWRecord);
        plvisPulse->lgw_lon0 = p->lon0;
        plvisPulse->lgw_lat0 = p->lat0;
        plvisPulse->lgw_z0 = p->z0;
        plvisPulse->lgw_lonend = p->lon431;
        plvisPulse->lgw_latend = p->lat431;
        plvisPulse->lgw_zend = p->z431;
        plvisPulse->lgw_sigmean = p->sigmean;
        plvisWaveformInfo->number_of_waveform_received_bins = sizeof(p->wave) / sizeof(p->wave[0]);
        plvisWaveformInfo->received_start_idx = pReceived->getNumElems();
        plvisWaveformInfo->number_of_waveform_transmitted_bins = 0;
        plvisWaveformInfo->transmitted_start_idx = 0;
        uint16_t data;
        for( npy_uint16 i = 0; i < plvisWaveformInfo->number_of_waveform_received_bins; i++ )
        {
            data = static_cast<uint16_t>(p->wave[i]);
            pReceived->push(&data);
        }
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lgw_lon0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lat0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_z0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lonend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_latend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_zend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_sigmean", 0);
        }
    }
    else if(lgwVersion == ((float)1.01))
    {
        struct lvis_lgw_v1_01 *p = reinterpret_cast<struct lvis_lgw_v1_01*>(pLGWRecord);
        plvisPulse->lgw_lfid = p->lfid;
        plvisPulse->lgw_shotnumber = p->shotnumber;
        plvisPulse->lgw_lon0 = p->lon0;
        plvisPulse->lgw_lat0 = p->lat0;
        plvisPulse->lgw_z0 = p->z0;
        plvisPulse->lgw_lonend = p->lon431;
        plvisPulse->lgw_latend = p->lat431;
        plvisPulse->lgw_zend = p->z431;
        plvisPulse->lgw_sigmean = p->sigmean;
        plvisWaveformInfo->number_of_waveform_received_bins = sizeof(p->wave) / sizeof(p->wave[0]);
        plvisWaveformInfo->received_start_idx = pReceived->getNumElems();
        plvisWaveformInfo->number_of_waveform_transmitted_bins = 0;
        plvisWaveformInfo->transmitted_start_idx = 0;
        uint16_t data;
        for( npy_uint16 i = 0; i < plvisWaveformInfo->number_of_waveform_received_bins; i++ )
        {
            data = static_cast<uint16_t>(p->wave[i]);
            pReceived->push(&data);
        }
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lgw_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lon0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lat0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_z0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lonend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_latend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_zend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_sigmean", 0);
        }
    }
    else if(lgwVersion == ((float)1.02))
    {
        struct lvis_lgw_v1_02 *p = reinterpret_cast<struct lvis_lgw_v1_02*>(pLGWRecord);
        plvisPulse->lgw_lfid = p->lfid;
        plvisPulse->lgw_shotnumber = p->shotnumber;
        plvisPulse->lgw_lvistime = p->lvistime;
        plvisPulse->lgw_lon0 = p->lon0;
        plvisPulse->lgw_lat0 = p->lat0;
        plvisPulse->lgw_z0 = p->z0;
        plvisPulse->lgw_lonend = p->lon431;
        plvisPulse->lgw_latend = p->lat431;
        plvisPulse->lgw_zend = p->z431;
        plvisPulse->lgw_sigmean = p->sigmean;
        plvisWaveformInfo->number_of_waveform_received_bins = sizeof(p->wave) / sizeof(p->wave[0]);
        plvisWaveformInfo->received_start_idx = pReceived->getNumElems();
        plvisWaveformInfo->number_of_waveform_transmitted_bins = 0;
        plvisWaveformInfo->transmitted_start_idx = 0;
        uint16_t data;
        for( npy_uint16 i = 0; i < plvisWaveformInfo->number_of_waveform_received_bins; i++ )
        {
            data = static_cast<uint16_t>(p->wave[i]);
            pReceived->push(&data);
        }
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lgw_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lon0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lat0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_z0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lonend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_latend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_zend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_sigmean", 0);
        }
    }
    else if(lgwVersion == ((float)1.03))
    {
        struct lvis_lgw_v1_03 *p = reinterpret_cast<struct lvis_lgw_v1_03*>(pLGWRecord);
        plvisPulse->lgw_lfid = p->lfid;
        plvisPulse->lgw_shotnumber = p->shotnumber;
        plvisPulse->lgw_azimuth = p->azimuth;
        plvisPulse->lgw_incidentangle = p->incidentangle;
        plvisPulse->lgw_range = p->range;
        plvisPulse->lgw_lvistime = p->lvistime;
        plvisPulse->lgw_lon0 = p->lon0;
        plvisPulse->lgw_lat0 = p->lat0;
        plvisPulse->lgw_z0 = p->z0;
        plvisPulse->lgw_lonend = p->lon431;
        plvisPulse->lgw_latend = p->lat431;
        plvisPulse->lgw_zend = p->z431;
        plvisPulse->lgw_sigmean = p->sigmean;
        plvisWaveformInfo->number_of_waveform_received_bins = sizeof(p->rxwave) / sizeof(p->rxwave[0]);
        plvisWaveformInfo->received_start_idx = pReceived->getNumElems();
        plvisWaveformInfo->number_of_waveform_transmitted_bins = sizeof(p->txwave) / sizeof(p->txwave[0]);
        plvisWaveformInfo->transmitted_start_idx = pTransmitted->getNumElems();
        uint16_t data;
        for( npy_uint16 i = 0; i < plvisWaveformInfo->number_of_waveform_received_bins; i++ )
        {
            data = static_cast<uint16_t>(p->rxwave[i]);
            pReceived->push(&data);
        }
        for( npy_uint16 i = 0; i < plvisWaveformInfo->number_of_waveform_transmitted_bins; i++ )
        {
            data = static_cast<uint16_t>(p->txwave[i]);
            pTransmitted->push(&data);
        }
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lgw_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_azimuth", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_incidentangle", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_range", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lon0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lat0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_z0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lonend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_latend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_zend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_sigmean", 0);
        }
    }
    else if(lgwVersion == ((float)1.04))
    {
        struct lvis_lgw_v1_04 *p = reinterpret_cast<struct lvis_lgw_v1_04*>(pLGWRecord);
        plvisPulse->lgw_lfid = p->lfid;
        plvisPulse->lgw_shotnumber = p->shotnumber;
        plvisPulse->lgw_azimuth = p->azimuth;
        plvisPulse->lgw_incidentangle = p->incidentangle;
        plvisPulse->lgw_range = p->range;
        plvisPulse->lgw_lvistime = p->lvistime;
        plvisPulse->lgw_lon0 = p->lon0;
        plvisPulse->lgw_lat0 = p->lat0;
        plvisPulse->lgw_z0 = p->z0;
        plvisPulse->lgw_lonend = p->lon527;
        plvisPulse->lgw_latend = p->lat527;
        plvisPulse->lgw_zend = p->z527;
        plvisPulse->lgw_sigmean = p->sigmean;
        plvisWaveformInfo->number_of_waveform_received_bins = sizeof(p->rxwave) / sizeof(p->rxwave[0]);
        plvisWaveformInfo->received_start_idx = pReceived->getNumElems();
        plvisWaveformInfo->number_of_waveform_transmitted_bins = sizeof(p->txwave) / sizeof(p->txwave[0]);
        plvisWaveformInfo->transmitted_start_idx = pTransmitted->getNumElems();
        for( npy_uint16 i = 0; i < plvisWaveformInfo->number_of_waveform_received_bins; i++ )
        {
            pReceived->push(&p->rxwave[i]);
        }
        for( npy_uint16 i = 0; i < plvisWaveformInfo->number_of_waveform_transmitted_bins; i++ )
        {
            pTransmitted->push(&p->txwave[i]);
        }
        // these fields are used on this version - only need to do 
        // this first time
        if( !bRecordsRead )
        {
            pylidar_setIgnore(pLVISPulseFields, "lgw_lfid", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_shotnumber", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_azimuth", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_incidentangle", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_range", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lvistime", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lon0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lat0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_z0", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_lonend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_latend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_zend", 0);
            pylidar_setIgnore(pLVISPulseFields, "lgw_sigmean", 0);
        }
    }

    if( plvisPoint != NULL )
    {
        // set the point from LGW
        if( nPointFrom == POINT_FROM_LGW0 )
        {
            plvisPoint->x = plvisPulse->lgw_lon0;
            plvisPoint->y = plvisPulse->lgw_lat0;
            plvisPoint->z = plvisPulse->lgw_z0;
        }
        else
        {
            plvisPoint->x = plvisPulse->lgw_lonend;
            plvisPoint->y = plvisPulse->lgw_latend;
            plvisPoint->z = plvisPulse->lgw_zend;
        }
    }
}

static PyObject *PyLVISFiles_readData(PyLVISFiles *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;

    pylidar::CVector<SLVISPulse> pulses(nPulses, nGrowBy);
    pylidar::CVector<SLVISPoint> points(nPulses, nGrowBy);
    pylidar::CVector<SLVISWaveformInfo> waveforms(nPulses, nGrowBy);

    // TODO: better estimate of initial size based on nPulses and size of 
    // waveform array for the LGW version.
    pylidar::CVector<uint16_t> transmitted(nInitSize, nGrowBy);
    pylidar::CVector<uint16_t> received(nInitSize, nGrowBy);

    SLVISPulse lvisPulse;
    lvisPulse.lceVersion = self->lceVersion;
    lvisPulse.lgeVersion = self->lgeVersion;
    lvisPulse.lgwVersion = self->lgwVersion;
    lvisPulse.wfm_start_idx = 0;
    lvisPulse.number_of_waveform_samples = 0;
    lvisPulse.number_of_returns = 1; // always just one return
    lvisPulse.pts_start_idx = 0;

    SLVISPoint lvisPoint;
    // these not really supported (but needed by SPDv4) to init to 0
    lvisPoint.classification = 0; // created

    SLVISWaveformInfo lvisWaveformInfo;

    // we don't really support these fields but they are needed
    // by SPDV4 and in theory make sense to have anyway. 
    // Set them so they don't do anything. Data comes as int from LVIS
    lvisWaveformInfo.receive_wave_gain = 1.0;
    lvisWaveformInfo.receive_wave_offset = 0.0;
    lvisWaveformInfo.trans_wave_gain = 1.0;
    lvisWaveformInfo.trans_wave_offset = 0.0;

    // seek to the right spot in the files
    size_t lceSize = 0;
    size_t lgeSize = 0;
    size_t lgwSize = 0;

    if( self->fpLCE != NULL )
    {
        lceSize = getStructSize(self, LVIS_RELEASE_FILETYPE_LCE);
        if( fseek(self->fpLCE, lceSize * nPulseStart, SEEK_SET) == -1 )
        {
            self->bFinished = true;            
        }
    }

    if( self->fpLGE != NULL )
    {
        lgeSize = getStructSize(self, LVIS_RELEASE_FILETYPE_LGE);
        if( fseek(self->fpLGE, lgeSize * nPulseStart, SEEK_SET) == -1 )
        {
            self->bFinished = true;            
        }
    }

    if( self->fpLGW != NULL )
    {
        lgwSize = getStructSize(self, LVIS_RELEASE_FILETYPE_LGW);
        if( fseek(self->fpLGW, lgwSize * nPulseStart, SEEK_SET) == -1 )
        {
            self->bFinished = true;            
        }
    }

    // read the data out
    if( !self->bFinished )
    {
        char *pLCERecord = NULL, *pLGERecord = NULL, *pLGWRecord = NULL;
        for( Py_ssize_t nPulseCount = 0; nPulseCount < nPulses; nPulseCount++ )
        {
            lvisPulse.pts_start_idx = points.getNumElems();

            // allocate some space for each record first time around
            if( self->fpLCE != NULL )
            { 
                if(pLCERecord == NULL)
                    pLCERecord = (char*)malloc(lceSize);

                if( fread(pLCERecord, lceSize, 1, self->fpLCE) < 1 )
                {
                    self->bFinished = true;
                    break;
                }

                SLVISPoint *p = NULL;
                if( self->nPointFrom == POINT_FROM_LCE )
                    p = &lvisPoint;
                saveLCEToStruct(self->lceVersion, pLCERecord, &lvisPulse, p, 
                        self->pPulseDefn, self->bRecordsRead);
            }
            if( self->fpLGE != NULL )
            {
                if(pLGERecord == NULL )
                    pLGERecord = (char*)malloc(lgeSize);

                if( fread(pLGERecord, lgeSize, 1, self->fpLGE) < 1 )
                {
                    self->bFinished = true;
                    break;
                }

                SLVISPoint *p = NULL;
                if( self->nPointFrom == POINT_FROM_LGE )
                    p = &lvisPoint;
                saveLGEToStruct(self->lgeVersion, pLGERecord, &lvisPulse, p, 
                        self->pPulseDefn, self->bRecordsRead);
            }
            if( self->fpLGW != NULL )
            {
                if(pLGWRecord == NULL )
                    pLGWRecord = (char*)malloc(lgwSize);

                if( fread(pLGWRecord, lgwSize, 1, self->fpLGW) < 1 )
                {
                    self->bFinished = true;
                    break;
                }

                SLVISPoint *p = NULL;
                if( ( self->nPointFrom == POINT_FROM_LGW0 ) || ( self->nPointFrom == POINT_FROM_LGWEND) )
                    p = &lvisPoint;
                saveLGWToStruct(self->lgwVersion, pLGWRecord, &lvisWaveformInfo, 
                        &lvisPulse, p, &transmitted, &received, self->pPulseDefn, 
                        self->bRecordsRead, self->nPointFrom);

                lvisPulse.wfm_start_idx = waveforms.getNumElems();
                lvisPulse.number_of_waveform_samples = 1; // only one per pulse
            }

            self->bRecordsRead = true;

            pulses.push(&lvisPulse);
            points.push(&lvisPoint);
            if( self->fpLGW != NULL )
                waveforms.push(&lvisWaveformInfo);
        }
        if( pLCERecord != NULL )
            free(pLCERecord);
        if( pLGERecord != NULL )
            free(pLGERecord);
        if( pLGWRecord != NULL )
            free(pLGWRecord);
    }

    PyArrayObject *pNumpyPoints = points.getNumpyArray(LVISPointFields);
    PyArrayObject *pNumpyPulses = pulses.getNumpyArray(self->pPulseDefn);
    PyArrayObject *pNumpyInfos = waveforms.getNumpyArray(LVISWaveformInfoFields);
    PyArrayObject *pNumpyReceived = received.getNumpyArray(NPY_UINT16);
    PyArrayObject *pNumpyTransmitted = transmitted.getNumpyArray(NPY_UINT16);

    // build tuple
    PyObject *pTuple = PyTuple_Pack(5, pNumpyPulses, pNumpyPoints, pNumpyInfos, pNumpyReceived, pNumpyTransmitted);

    // decref the objects since we have finished with them (PyTuple_Pack increfs)
    Py_DECREF(pNumpyPulses);
    Py_DECREF(pNumpyPoints);
    Py_DECREF(pNumpyInfos);
    Py_DECREF(pNumpyReceived);
    Py_DECREF(pNumpyTransmitted);

    return pTuple;
}

// helper method to find the number of records in a file
// given a structure size
size_t getNumRecords(FILE *fp, size_t structSize)
{
    size_t currpos = ftell(fp);

    // seek to end
    fseek(fp, 0, SEEK_END);
    size_t endpos = ftell(fp);

    // back to where we where
    fseek(fp, currpos, SEEK_SET);
    
    // this should round down
    return endpos / structSize;
}

// return an estimate of the number of pulses based on the size
// of the file and the size of the structs for the file version 
// we are using. Assumes the size for the different file types 
// (lce, lge, lgw) are the same since it just uses the first type
// that is available.
static PyObject *PyLVISFiles_getNumPulses(PyLVISFiles *self, PyObject *args)
{
    size_t structSize = 0;
    size_t numPulses = 0;
    if( self->fpLCE != NULL )
    {
        structSize = getStructSize(self, LVIS_RELEASE_FILETYPE_LCE);
        numPulses = getNumRecords(self->fpLCE, structSize);
    }

    if( self->fpLGE != NULL )
    {
        structSize = getStructSize(self, LVIS_RELEASE_FILETYPE_LGE);
        numPulses = getNumRecords(self->fpLGE, structSize);
    }

    if( self->fpLGW != NULL )
    {
        structSize = getStructSize(self, LVIS_RELEASE_FILETYPE_LGW);
        numPulses = getNumRecords(self->fpLGW, structSize);
    }

    return PyLong_FromSize_t(numPulses);
}

/* Table of methods */
static PyMethodDef PyLVISFiles_methods[] = {
    {"readData", (PyCFunction)PyLVISFiles_readData, METH_VARARGS, NULL},
    {"getNumPulses", (PyCFunction)PyLVISFiles_getNumPulses, METH_NOARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyObject *PyLVISFiles_getFinished(PyLVISFiles* self, void *closure)
{
    if( self->bFinished )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyGetSetDef PyLVISFiles_getseters[] = {
    {(char*)"finished", (getter)PyLVISFiles_getFinished, NULL, (char*)"Get Finished reading state", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyLVISFilesType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_lvisbin.File",         /*tp_name*/
    sizeof(PyLVISFiles),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyLVISFiles_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "LVIS Binary File object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyLVISFiles_methods,             /* tp_methods */
    0,             /* tp_members */
    PyLVISFiles_getseters,     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyLVISFiles_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};
#if PY_MAJOR_VERSION >= 3

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__lvisbin(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_lvisbin(void)
#endif
{
    PyObject *pModule;
    struct LVISBinState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_lvisbin", NULL);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_lvisbin.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    PyModule_AddObject(pModule, "error", state->error);

    /* Scan file type */
    PyLVISFilesType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyLVISFilesType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyLVISFilesType);
    PyModule_AddObject(pModule, "File", (PyObject *)&PyLVISFilesType);

    // module constants
    PyModule_AddIntConstant(pModule, "POINT_FROM_LCE", POINT_FROM_LCE);
    PyModule_AddIntConstant(pModule, "POINT_FROM_LGE", POINT_FROM_LGE);
    PyModule_AddIntConstant(pModule, "POINT_FROM_LGW0", POINT_FROM_LGW0);
    PyModule_AddIntConstant(pModule, "POINT_FROM_LGWEND", POINT_FROM_LGWEND);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
