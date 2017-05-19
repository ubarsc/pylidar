
// adapted from http://lvis.gsfc.nasa.gov/utilities_home.html

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include "numpy/arrayobject.h"

#include "lvis.h"
#include "pylidar.h"

#ifndef  GENLIB_LITTLE_ENDIAN
#define  GENLIB_LITTLE_ENDIAN 0x00
#endif

#ifndef  GENLIB_BIG_ENDIAN
#define  GENLIB_BIG_ENDIAN 0x01
#endif

/* define WKB_BYTE_ORDER depending on endian setting
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
struct LVISState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct LVISState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct LVISState _state;
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
} SLVISWaveformInfo;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn LVISWaveformInfoFields[] = {
    CREATE_FIELD_DEFN(SLVISWaveformInfo, number_of_waveform_received_bins, 'u'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, received_start_idx, 'u'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, number_of_waveform_transmitted_bins, 'u'),
    CREATE_FIELD_DEFN(SLVISWaveformInfo, transmitted_start_idx, 'u'),
    {NULL} // Sentinel
};

typedef struct {
    double x;
    double y;
    float z;
} LVISPoint;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn LVISPointFields[] = {
    CREATE_FIELD_DEFN(LVISPoint, x, 'f'),
    CREATE_FIELD_DEFN(LVISPoint, y, 'f'),
    CREATE_FIELD_DEFN(LVISPoint, z, 'f'),
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
} PyLVISFiles;

#if PY_MAJOR_VERSION >= 3
static int lvis_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int lvis_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_lvis",
        NULL,
        sizeof(struct LVISState),
        NULL,
        NULL,
        lvis_traverse,
        lvis_clear,
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

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int 
PyLVISFiles_init(PyLVISFiles *self, PyObject *args, PyObject *kwds)
{
char *pszLCE_Fname = NULL, *pszLGE_Fname = NULL, *pszLGW_Fname = NULL;

    if( !PyArg_ParseTuple(args, "zzz", &pszLCE_Fname, &pszLGE_Fname, &pszLGW_Fname ) )
    {
        return -1;
    }

    self->fpLCE = NULL;
    self->fpLGE = NULL;
    self->fpLGW = NULL;
    self->lceVersion = 0;
    self->lgeVersion = 0;
    self->lgwVersion = 0;

    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    // best way I could find for obtaining module reference
    // from inside a class method. Not needed for Python < 3.
    m = PyState_FindModule(&moduledef);
#endif
    int nFileType;

    if( pszLCE_Fname != NULL )
    {
        self->fpLCE = fopen(pszLCE_Fname, "rb");
        if( self->fpLCE == NULL )
        {
            PyErr_Format(GETSTATE(m)->error, "Cannot open %s", pszLCE_Fname);
            return -1;
        }

        nFileType = 999;
        detect_release_version(pszLCE_Fname, &nFileType, &self->lceVersion, GENLIB_OUR_ENDIAN);
        if( nFileType != LVIS_RELEASE_FILETYPE_LCE )
        {
            PyErr_Format(GETSTATE(m)->error, "File %s does not contain LCE data", pszLCE_Fname);
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
            PyErr_Format(GETSTATE(m)->error, "Cannot open %s", pszLGE_Fname);
            return -1;
        }

        nFileType = 999;
        detect_release_version(pszLGE_Fname, &nFileType, &self->lgeVersion, GENLIB_OUR_ENDIAN);
        if( nFileType != LVIS_RELEASE_FILETYPE_LGE )
        {
            PyErr_Format(GETSTATE(m)->error, "File %s does not contain LGE data", pszLGE_Fname);
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
            PyErr_Format(GETSTATE(m)->error, "Cannot open %s", pszLGW_Fname);
            return -1;
        }

        nFileType = 999;
        detect_release_version(pszLGW_Fname, &nFileType, &self->lgwVersion, GENLIB_OUR_ENDIAN);
        if( nFileType != LVIS_RELEASE_FILETYPE_LGW )
        {
            PyErr_Format(GETSTATE(m)->error, "File %s does not contain LGE data", pszLGE_Fname);
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

static PyObject *PyLVISFiles_readData(PyLVISFiles *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;

    Py_RETURN_NONE;
}

/* Table of methods */
static PyMethodDef PyLVISFiles_methods[] = {
    {"readData", (PyCFunction)PyLVISFiles_readData, METH_VARARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyLVISFilesType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_lvis.File",         /*tp_name*/
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
    "LVIS File object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyLVISFiles_methods,             /* tp_methods */
    0,             /* tp_members */
    0,           /* tp_getset */
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
PyInit__lvis(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_lvis(void)
#endif
{
    PyObject *pModule;
    struct LVISState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_lvis", module_methods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_lvis.error", NULL, NULL);
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
    PyModule_AddObject(pModule, "LVISFile", (PyObject *)&PyLVISFilesType);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
