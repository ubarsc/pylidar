
// adapted from http://lvis.gsfc.nasa.gov/utilities_home.html

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include "numpy/arrayobject.h"

#include "lvis.h"

#ifndef  DEFAULT_DATA_RELEASE_VERSION
#define  DEFAULT_DATA_RELEASE_VERSION ((float) 1.03)
#endif

#ifndef  LVIS_RELEASE_READER_VERSION
#define  LVIS_RELEASE_READER_VERSION  ((float) 1.04)
#endif

#ifndef  LVIS_RELEASE_READER_VERSION_DATE
#define  LVIS_RELEASE_READER_VERSION_DATE 20120131
#endif

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

// prototypes we need
double host_double(double input_double,int host_endian);
float  host_float(float input_float,int host_endian);

// if you want a little more info to start...uncomment this
/* #define DEBUG_ON */

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
   int           i,mintype,type,minversion,version,status=0,maxpackets,maxsize;
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

dword host_dword(dword input_dword,int host_endian)
{
   dword return_value=0;
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

word host_word(word input_word,int host_endian)
{
   word return_value=0;
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
    float lcwVersion;

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
   uint32_t lge_shotnumber;  
   float lge_azimuth;
   float lge_incidentangle;
   float lge_range;
   double lge_lvistime;
   double lge_glon;
   double lge_glat;
   float  lge_zg;
   float  lge_rh25;
   float  lge_rh50;
   float  lge_rh75;
   float  lge_rh100;
    
} SLVISPulse;
