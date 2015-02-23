
#ifndef PYLIDAR_H
#define PYLIDAR_H

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/* mark all exported classes/functions with DllExport to have
 them exported by Visual Studio */
#ifndef DllExport
    #ifdef _MSC_VER
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport
    #endif
#endif

/* MSVC 2008 uses different names.... */
#ifdef _MSC_VER
    #if _MSC_VER >= 1600
        #include <stdint.h>
    #else        
        typedef __int8              int8_t;
        typedef __int16             int16_t;
        typedef __int32             int32_t;
        typedef __int64             int64_t;
        typedef unsigned __int8     uint8_t;
        typedef unsigned __int16    uint16_t;
        typedef unsigned __int32    uint32_t;
        typedef unsigned __int64    uint64_t;
    #endif
#else
    #include <stdint.h>
#endif

/* call first - preferably in the init() of your module
 this sets up the connection to numpy */
DllExport void pylidar_init();

/* print error - used internally */
DllExport void pylidar_error(char *errstr, ...);

/* Helper function to get information about a named field within an array
 pass null for params you not interested in */
DllExport int pylidar_getFieldDescr(PyObject *pArray, const char *pszName, int *pnOffset, char *pcKind, int *pnSize, int *pnLength);

/* return a field as a int64_t array. Caller to delete */
DllExport int64_t *pylidar_getFieldAsInt64(PyObject *pArray, const char *pszName);

/* return a field as a double array. Caller to delete */
DllExport double *pylidar_getFieldAsFloat64(PyObject *pArray, const char *pszName);


/* structure for defining a numpy structured array from a C one
 create using CREATE_FIELD_DEFN below */
typedef struct
{
    const char *pszName;
    char cKind; // 'i' for signed int, 'u' for unsigned int, 'f' for float
    int nSize;
    int nOffset;
    int nStructTotalSize;
} SpylidarFieldDefn;

#define CREATE_FIELD_DEFN(STRUCT, FIELD, KIND) \
    {#FIELD, KIND, sizeof(STRUCT.FIELD), offsetof(STRUCT, FIELD), sizeof(STRUCT)}

/* 
Here is an example of use:
//Say you had a structure like this:
typedef struct {
    double x,
    double y,
    int count
}

//Create an array of structures defining the fields like this:
static SpylidarFieldDefn fields[] = {
    CREATE_FIELD_DEFN(SMyStruct, x, 'f'),
    CREATE_FIELD_DEFN(SMyStruct, y, 'f'),
    CREATE_FIELD_DEFNS(MyStruct, count, 'i'),
    {NULL} // Sentinel 
};

*/
/* Wrap an existing C array of structures and return as a numpy array */
/* Python will free data when finished */
DLLExport PyObject *pylidar_structArrayToNumpy(void *pStructArray, npy_intp nElems, SpylidarFieldDefn *pDefn);

#ifdef __cplusplus
}
#endif

#endif /*PYLIDAR_H*/

