#ifndef __LVIS_RELEASE_STRUCTURE_H
#define __LVIS_RELEASE_STRUCTURE_H

// Begun: 2004/08/19
// David Lloyd Rabine
// NASA GSFC Code 694.0 Laser Remote Sensing Branch
// david@ltpmail.gsfc.nasa.gov
// 301-614-6771
// http://lvis.gsfc.nasa.gov
// 

// Revision History:
// 2005/10/28 - dlr
//              added the lfid and shotnumber to all structures
// 2006/04/18 - dlr
//              added the time field to all structures (seconds of the day)
// 2008/11/20 - dlr
//              adding all structure versions to the code
//              additionally, adding header defintion strings
//              moved the LCE from 01 to 00 to match the detection routine return
// 2011/12/13 - dlr
//              modifying this code to use stdint.h so sizes are consistent on any platform

#include <stdint.h>

#define LVIS_RELEASE_FILETYPE_LCE 0x00
#define LVIS_RELEASE_FILETYPE_LGE 0x01
#define LVIS_RELEASE_FILETYPE_LGW 0x02

#define LVIS_RELEASE_STRUCTURE_DATE 20111213
#define LVIS_RELEASE_STRUCTURE_VERSION  1.04
//
// Data Notes: 
// All latitude, longitude and elevations are relative to 
// the WGS-84 reference system
//
// Digitizer is a Signatec PDA500 recording at 8bits and 500MHz sampling rate
// disciplined by an rubidium oscillator

// LFID description  IIMMMMMNNNN
// =====================================
// II    = Instrument identifier (0->32)
// MMMMM = modified julian day
// NNNN  = file number for that day

// pack the bytes (no padding allowed)
//

#pragma pack(1)
struct lvis_lce_v1_00  // LVIS Canopy Elevation v1.00
{
   double tlon;  // longitude of the highest detected return (degrees east)
   double tlat;  // latitude of the highest detected return (degrees north)
   float  zt;    // elevation of the highest detected return (m) (i.e., rh100, see below)
};

typedef struct lce_v1_00 * ptr_lce_v1_00;

#pragma pack(1)
struct lvis_lce_v1_01  // LVIS Canopy Elevation v1.01
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   double tlon;  // longitude of the highest detected return (degrees east)
   double tlat;  // latitude of the highest detected return (degrees north)
   float  zt;    // elevation of the highest detected return (m) (i.e., rh100, see below)
};

typedef struct lce_v1_01 * ptr_lce_v1_01;

#pragma pack(1)
struct lvis_lce_v1_02  // LVIS Canopy Elevation v1.02
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double tlon;  // longitude of the highest detected return (degrees east)
   double tlat;  // latitude of the highest detected return (degrees north)
   float  zt;    // elevation of the highest detected return (m) (i.e., rh100, see below)
};

typedef struct lce_v1_02 * ptr_lce_v1_02;

#pragma pack(1)
struct lvis_lce_v1_03  // LVIS Canopy Elevation v1.03
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   float azimuth;            // true heading from the aircraft to the ground (degrees)
   float incidentangle;      // off nadir angle (degrees)
   float range;              // range from the aircraft to the ground (meters)
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double tlon;  // longitude of the highest detected return (degrees east)
   double tlat;  // latitude of the highest detected return (degrees north)
   float  zt;    // elevation of the highest detected return (m) (i.e., rh100, see below)
};

typedef struct lce_v1_03 * ptr_lce_v1_03;

#pragma pack(1)
struct lvis_lce_v1_04  // LVIS Canopy Elevation v1.04
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   float azimuth;            // true heading from the aircraft to the ground (degrees)
   float incidentangle;      // off nadir angle (degrees)
   float range;              // range from the aircraft to the ground (meters)
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double tlon;  // longitude of the highest detected return (degrees east)
   double tlat;  // latitude of the highest detected return (degrees north)
   float  zt;    // elevation of the highest detected return (m) (i.e., rh100, see below)
};

typedef struct lce_v1_04 * ptr_lce_v1_04;

#pragma pack(1)
struct lvis_lge_v1_00 // LVIS Ground Elevation v1.00
{
   double glon;   // longitude of the lowest detected mode within the waveform (degrees east)
   double glat;   // latitude of the lowest detected mode within the waveform (degrees north)
   float  zg;     // mean elevation of the lowest detected mode within the waveform (m)
   float  rh25;   // height (relative to zg) at which 25% of the waveform energy occurs (m)
   float  rh50;   // height (relative to zg) at which 50% of the waveform energy occurs (m)
   float  rh75;   // height (relative to zg) at which 75% of the waveform energy occurs (m)
   float  rh100;  // height (relative to zg) at which 100% of the waveform energy occurs (m)
};

typedef struct lge_v1_00 * ptr_lge_v1_00;

#pragma pack(1)
struct lvis_lge_v1_01 // LVIS Ground Elevation v1.01
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   double glon;   // longitude of the lowest detected mode within the waveform (degrees east)
   double glat;   // latitude of the lowest detected mode within the waveform (degrees north)
   float  zg;     // mean elevation of the lowest detected mode within the waveform (m)
   float  rh25;   // height (relative to zg) at which 25% of the waveform energy occurs (m)
   float  rh50;   // height (relative to zg) at which 50% of the waveform energy occurs (m)
   float  rh75;   // height (relative to zg) at which 75% of the waveform energy occurs (m)
   float  rh100;  // height (relative to zg) at which 100% of the waveform energy occurs (m)
};

typedef struct lge_v1_01 * ptr_lge_v1_01;

#pragma pack(1)
struct lvis_lge_v1_02 // LVIS Ground Elevation v1.02
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double glon;   // longitude of the lowest detected mode within the waveform (degrees east)
   double glat;   // latitude of the lowest detected mode within the waveform (degrees north)
   float  zg;     // mean elevation of the lowest detected mode within the waveform (m)
   float  rh25;   // height (relative to zg) at which 25% of the waveform energy occurs (m)
   float  rh50;   // height (relative to zg) at which 50% of the waveform energy occurs (m)
   float  rh75;   // height (relative to zg) at which 75% of the waveform energy occurs (m)
   float  rh100;  // height (relative to zg) at which 100% of the waveform energy occurs (m)
};

typedef struct lge_v1_02 * ptr_lge_v1_02;

#pragma pack(1)
struct lvis_lge_v1_03 // LVIS Ground Elevation v1.03
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   float azimuth;            // true heading from the aircraft to the ground (degrees)
   float incidentangle;      // off nadir angle (degrees)
   float range;              // range from the aircraft to the ground (meters)
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double glon;   // longitude of the lowest detected mode within the waveform (degrees east)
   double glat;   // latitude of the lowest detected mode within the waveform (degrees north)
   float  zg;     // mean elevation of the lowest detected mode within the waveform (m)
   float  rh25;   // height (relative to zg) at which 25% of the waveform energy occurs (m)
   float  rh50;   // height (relative to zg) at which 50% of the waveform energy occurs (m)
   float  rh75;   // height (relative to zg) at which 75% of the waveform energy occurs (m)
   float  rh100;  // height (relative to zg) at which 100% of the waveform energy occurs (m)
};

typedef struct lge_v1_03 * ptr_lge_v1_03;

#pragma pack(1)
struct lvis_lge_v1_04 // LVIS Ground Elevation v1.04
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   float azimuth;            // true heading from the aircraft to the ground (degrees)
   float incidentangle;      // off nadir angle (degrees)
   float range;              // range from the aircraft to the ground (meters)
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double glon;   // longitude of the lowest detected mode within the waveform (degrees east)
   double glat;   // latitude of the lowest detected mode within the waveform (degrees north)
   float  zg;     // mean elevation of the lowest detected mode within the waveform (m)
   float  rh25;   // height (relative to zg) at which 25% of the waveform energy occurs (m)
   float  rh50;   // height (relative to zg) at which 50% of the waveform energy occurs (m)
   float  rh75;   // height (relative to zg) at which 75% of the waveform energy occurs (m)
   float  rh100;  // height (relative to zg) at which 100% of the waveform energy occurs (m)
};

typedef struct lge_v1_04 * ptr_lge_v1_04;

#pragma pack(1)
struct lvis_lgw_v1_00 // LVIS Geolocated Waveforms v1.00
{
   double lon0;    // longitude of the highest sample of the waveform (degrees east)
   double lat0;    // latitude of the highest sample of the waveform (degrees north)
   float  z0;      // elevation of the highest sample of the waveform (m)
   double lon431;  // longitude of the lowest sample of the waveform (degrees east)
   double lat431;  // latitude of the lowest sample of the waveform (degrees north)
   float  z431;    // elevation of the lowest sample of the waveform (m)
   float  sigmean; // signal mean noise level, calculated in-flight (counts)
   unsigned char wave[432]; // return waveform, recorded in-flight (counts)
};

typedef struct lgw_v1_00 * ptr_lgw_v1_00;

#pragma pack(1)
struct lvis_lgw_v1_01 // LVIS Geolocated Waveforms v1.01
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   double lon0;    // longitude of the highest sample of the waveform (degrees east)
   double lat0;    // latitude of the highest sample of the waveform (degrees north)
   float  z0;      // elevation of the highest sample of the waveform (m)
   double lon431;  // longitude of the lowest sample of the waveform (degrees east)
   double lat431;  // latitude of the lowest sample of the waveform (degrees north)
   float  z431;    // elevation of the lowest sample of the waveform (m)
   float  sigmean; // signal mean noise level, calculated in-flight (counts)
   unsigned char wave[432]; // return waveform, recorded in-flight (counts)
};

typedef struct lgw_v1_01 * ptr_lgw_v1_01;

#pragma pack(1)
struct lvis_lgw_v1_02 // LVIS Geolocated Waveforms v1.02
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double lon0;    // longitude of the highest sample of the waveform (degrees east)
   double lat0;    // latitude of the highest sample of the waveform (degrees north)
   float  z0;      // elevation of the highest sample of the waveform (m)
   double lon431;  // longitude of the lowest sample of the waveform (degrees east)
   double lat431;  // latitude of the lowest sample of the waveform (degrees north)
   float  z431;    // elevation of the lowest sample of the waveform (m)
   float  sigmean; // signal mean noise level, calculated in-flight (counts)
   unsigned char wave[432]; // return waveform, recorded in-flight (counts)
};

typedef struct lgw_v1_02 * ptr_lgw_v1_02;

#pragma pack(1)
struct lvis_lgw_v1_03 // LVIS Geolocated Waveforms v1.03
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   float azimuth;            // true heading from the aircraft to the ground (degrees)
   float incidentangle;      // off nadir angle (degrees)
   float range;              // range from the aircraft to the ground (meters)
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double lon0;    // longitude of the highest sample of the waveform (degrees east)
   double lat0;    // latitude of the highest sample of the waveform (degrees north)
   float  z0;      // elevation of the highest sample of the waveform (m)
   double lon431;  // longitude of the lowest sample of the waveform (degrees east)
   double lat431;  // latitude of the lowest sample of the waveform (degrees north)
   float  z431;    // elevation of the lowest sample of the waveform (m)
   float  sigmean; // signal mean noise level, calculated in-flight (counts)
   unsigned char txwave[80];  // transmit waveform, recorded in-flight (counts)
   unsigned char rxwave[432]; // return   waveform, recorded in-flight (counts)
};

typedef struct lgw_v1_03 * ptr_lgw_v1_03;

#pragma pack(1)
struct lvis_lgw_v1_04 // LVIS Geolocated Waveforms v1.04
{
   uint32_t lfid;       // unique LVIS file identifier
   uint32_t shotnumber; // unique LVIS shotnumber (in a file the shotnumber is always unique) 
   float azimuth;            // true heading from the aircraft to the ground (degrees)
   float incidentangle;      // off nadir angle (degrees)
   float range;              // range from the aircraft to the ground (meters)
   double lvistime;          // LVIS recorded UTC time (seconds of the day) when the shot was acquired
   double lon0;    // longitude of the highest sample of the waveform (degrees east)
   double lat0;    // latitude of the highest sample of the waveform (degrees north)
   float  z0;      // elevation of the highest sample of the waveform (m)
   double lon527;  // longitude of the lowest sample of the waveform (degrees east)
   double lat527;  // latitude of the lowest sample of the waveform (degrees north)
   float  z527;    // elevation of the lowest sample of the waveform (m)
   float  sigmean; // signal mean noise level, calculated in-flight (counts)
   uint16_t txwave[120];  // transmit waveform, recorded in-flight (counts)
   uint16_t rxwave[528]; // return   waveform, recorded in-flight (counts)
};

typedef struct lgw_v1_04 * ptr_lgw_v1_04;

#pragma pack()

#endif
