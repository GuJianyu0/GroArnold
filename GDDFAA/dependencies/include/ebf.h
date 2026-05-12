/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Author: Sanjib Sharma                                                     *
 * Copyright (c) 2012 Sanjib Sharma                                          *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of EBF.  The full EBF copyright notice, including       *
 * terms governing use, modification, and redistribution, is contained in    *
 * the files COPYING and COPYRIGHT, which can be found at the root           *
 * of the source code distribution tree.                                     *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef EBF_H_
#define EBF_H_



#include "ebftable.h"

#ifdef EBFLANGOPTCPP
#include<cstdlib>
#include<cstring>
using namespace  ebf;
using namespace  ebf::ebfc;
#else
#include<stdlib.h>
#include<string.h>
#ifdef __cplusplus
 extern "C" {
#endif
#endif

#define EBF_FILE_NAME_MAXSIZE 1000




/**
 * An integer type code used by EBF to define data types.
 * @param typestring The name of data type. Valid values are \n
 * char, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64
 * @return an integer giving the type code
 */
int Ebf_TypeS(const char* typestring);









/**
 * The main structure using which  input output operations are performed
 */

typedef union
{
	char bufchar[8000];
	int8_t bufint8[8000];
	int16_t bufint16[4000];
	int32_t bufint32[2000];
	int64_t bufint64[1000];
	uint8_t bufuint8[8000];
	uint16_t bufuint16[4000];
	uint32_t bufuint32[2000];
	uint64_t bufuint64[1000];
	efloat32 buffloat32[2000];
	efloat64 buffloat64[1000];
} EbfFileBuf;

typedef struct
{
	char filename[EBF_FILE_NAME_MAXSIZE];
	EbfFileBuf tbuf;
	int debug,magic,mode,status,ecode,givenmode;
	int patternData_size;
	FILE* fp;
	EbfHeader ebfh;
} EbfFile;

/**
 * A constructor for EbfFile
 * To be used as EbfFile efile=EbfFile_Create();
 * @return an initilaized EbfFile
 */
EbfFile EbfFile_Create();

/**
 *   open an efile object for I/O operations.
 *   @param  efile    a pointer to efile object
 *   @param filename1: (String) The name of file on which to perform I/O
 *   @param dataname1: (String) The name of the data object, to be read or written.
 *   @param mode1    : (String) "w" for writing a fresh file, "a"  for appending to an existing file.
 *               In general "r" is for reading.
 *   @param datatype: an integer type code specifying the intended destination datatype.
 *               See Ebf_TypeS for generating from a string.
 *   @param dataunit:(char *)  units of data e.g. 100 m/s.
 *   @param rank The number of dimensions.
 *   @param dim  A pointer to an array specifying the dimensions. Note, the first
 *     dimension can be set to 0. When Ebf_close is invoked dim[0] adjusted
 *     to match the number of items written.
 *
 */
void EbfFile_Open(EbfFile* efile,const char* filename1,const char* dataname1,const char* mode1,int datatype,const char* dataunit,int rank, int64_t *dim);

/**
 *   open an efile object for read operation and position the file pointer to the location of the first element.
 *   @param  efile    a pointer to efile object
 *   @param filename1: (String) The name of file on which to perform I/O
 *   @param dataname1: (String) The name of the data object, to be read or written.
 */
void EbfFile_OpenR(EbfFile* efile,const char* filename1,const char* dataname1);

/**
 *   open an efile object for write operations.
 *   @param  efile    a pointer to efile object
 *   @param filename1: (String) The name of file on which to perform I/O
 *   @param dataname1: (String) The name of the data object, to be read or written.
 *   @param mode1    : (String) "w" for writing a fresh file, "a"  for appending to an existing file.
 *               In general "r" is for reading.
 *   @param datatype: an integer type code specifying the intended destination datatype.
 *               See Ebf_TypeS for generating from a string.
 *   @param dataunit:(char *)  units of data e.g. 100 m/s.
 */
void EbfFile_OpenW(EbfFile* efile,const char* filename1,const char* dataname1,const char* mode1,int datatype,const char* dataunit);
void EbfFile_SaveTo(EbfFile *efile,const char* filename1,const char* mode1);
void EbfFile_Close(EbfFile *efile);


/**
 * enquire header properties
 */
int64_t EbfFile_Elements(EbfFile *efile);

/**
 * get a linear index to a multidimensional data
 */
int64_t EbfFile_Index3(EbfFile *efile, int64_t i1,int64_t i2,int64_t i3);
int64_t EbfFile_Index2(EbfFile *efile, int64_t i1,int64_t i2);

/**
 * set get position	in units of datasize i.e index of elements
 */
void EbfFile_Seek(EbfFile *efile, int64_t offset);
int64_t EbfFile_Tell(EbfFile *efile);



/**
 * Type safe functions to write data to an already open file
 * Writes data_size elements from memeory pointed to by x, to an open ebf file and increments the file pointer.
 * @param efile pointer to an efile structure which should be initialized
 * @param data the pointer to data being written
 * @param data_size the number of elements to be written
 */
void EbfFile_wChar(EbfFile *efile,const char* data, int64_t data_size);
void EbfFile_wInt32(EbfFile *efile,const int32_t* data, int64_t data_size);
void EbfFile_wInt64(EbfFile *efile,const int64_t* data, int64_t data_size);
void EbfFile_wFloat32(EbfFile *efile,const efloat32* data, int64_t data_size);
void EbfFile_wFloat64(EbfFile *efile,const efloat64* data, int64_t data_size);
void EbfFile_wInt16(EbfFile *efile,const int16_t * data, int64_t data_size);
void EbfFile_wInt8(EbfFile *efile,const int8_t* data, int64_t data_size);
void EbfFile_wUInt8(EbfFile *efile,const uint8_t* data, int64_t data_size);
void EbfFile_wUInt16(EbfFile *efile,const uint16_t * data, int64_t data_size);
void EbfFile_wUInt32(EbfFile *efile,const uint32_t* data, int64_t data_size);
void EbfFile_wUInt64(EbfFile *efile,const uint64_t* data, int64_t data_size);

/**
 * Type safe functions to read data from an already open file
 * @param efile pointer to an efile structure which should be initialized
 * @param data the pointer to data to read to
 * @param data_size the number of elements to be read
 * @param offset1 the index of the element to be read. If -1 then successive read.
 */
void EbfFile_rChar(EbfFile *efile,char* data, int64_t data_size);
void EbfFile_rInt32(EbfFile *efile,int32_t* data, int64_t data_size);
void EbfFile_rInt64(EbfFile *efile,int64_t* data, int64_t data_size);
void EbfFile_rFloat32(EbfFile *efile,efloat32* data, int64_t data_size);
void EbfFile_rFloat64(EbfFile *efile,efloat64* data, int64_t data_size);
void EbfFile_rInt16(EbfFile *efile,int16_t * data, int64_t data_size);
void EbfFile_rInt8(EbfFile *efile,int8_t* data, int64_t data_size);
void EbfFile_rUInt8(EbfFile *efile,uint8_t* data, int64_t data_size);
void EbfFile_rUInt16(EbfFile *efile,uint16_t * data, int64_t data_size);
void EbfFile_rUInt32(EbfFile *efile,uint32_t* data, int64_t data_size);
void EbfFile_rUInt64(EbfFile *efile,uint64_t* data, int64_t data_size);
/**
 * read raw bytes with endian conversion, it is not typesafe.
 * Note: size and offset are in units of bytes for readVoid
 */
void EbfFile_rVoid(EbfFile *efile,void* data, int64_t data_size);





/**
 *   Reads an array of count elements from an ebf file into memory pointed to by x.
 *   There are 11 versions of this function depending upon the data type of data.
 *   The chosen function should match the data type of data.
 *   @param filename1: (char *) The name of file from which to read data e.g. "test.ebf"
 *   @param dataname1: (char *) A valid name of a data object that exists in the file, e.g. "/x"
 *   @param data       : A pointer of any data type into which the data will be read.
 *   @param dim1   : (int64_t) a 64 bit integer specifying the number of elements to be read
 *               Use \n EbfDataInfo dinfo = Ebf_getDataInfo() \n to get number of elements
 *               as dinfo.elements.
 */
int Ebf_ReadChar(const char *filename1,const char *dataname1,char* data, int64_t dim1);
int Ebf_ReadInt32(const char *filename1,const char *dataname1,int32_t* data, int64_t dim1);
int Ebf_ReadInt64(const char *filename1,const char *dataname1,int64_t* data, int64_t dim1);
int Ebf_ReadFloat32(const char *filename1,const char *dataname1,efloat32* data, int64_t dim1);
int Ebf_ReadFloat64(const char *filename1,const char *dataname1,efloat64* data, int64_t dim1);
int Ebf_ReadInt16(const char *filename1,const char *dataname1,int16_t* data, int64_t dim1);
int Ebf_ReadInt8(const char *filename1,const char *dataname1,int8_t * data, int64_t dim1);
int Ebf_ReadUInt8(const char *filename1,const char *dataname1,uint8_t* data, int64_t dim1);
int Ebf_ReadUInt16(const char *filename1,const char *dataname1,uint16_t * data, int64_t dim1);
int Ebf_ReadUInt32(const char *filename1,const char *dataname1,uint32_t * data, int64_t dim1);
int Ebf_ReadUInt64(const char *filename1,const char *dataname1,uint64_t* data, int64_t dim1);

/**
 * Read c style null terminated string. If size of supplied arrays is small
 * reads only part of it and issues a warning.
 * @param filename1
 * @param dataname1
 * @param data
 * @param maxsize maximum size of data array including null char
 */
int Ebf_ReadCString(const char *filename1,const char *dataname1,char* data, size_t maxsize);





/**
 *   Writes an array of count elements pointed to by pointer data.
 *   There are 11 versions of this function depending upon the data type of data e.g Ebf_WriteInt32(), Ebf_writeDouble() etc.
 *   The chosen function should match the data type of data. For appropriate suffixes see table above.
 *   @param filename1  The name of file in which to write data e.g. "test.ebf".
 *   @param dataname1  The name of the data object, this should begin with "/" , e.g. "/x".
 *   @param mode       The writing mode, "w" for fresh file and "a" to append to an existing file.
 *   @param data       An array or a scalar  of any one of the primitive data types.
 *   @param count      A 64 bit integer specifying the number of elements to be written.
 *   @param dataunit: Units of data e.g. 100 m/s.
 */
int Ebf_WriteChar(const char *filename1,const char * dataname1, const char* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteInt32(const char *filename1,const char * dataname1, const int32_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteInt64(const char *filename1,const char * dataname1, const int64_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteFloat32(const char *filename1,const char * dataname1, const efloat32* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteFloat64(const char *filename1,const char * dataname1, const efloat64* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteInt16(const char *filename1,const char * dataname1, const int16_t * data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteInt8(const char *filename1,const char * dataname1, const int8_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteUInt8(const char *filename1,const char * dataname1, const uint8_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteUInt16(const char *filename1,const char * dataname1, const uint16_t * data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteUInt32(const char *filename1,const char * dataname1, const  uint32_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteUInt64(const char *filename1,const char * dataname1, const uint64_t* data, const char* mode1, const char* dataunit, int64_t dim1);

int Ebf_WriteCharAs(const int32_t typecode, const char *filename1,const char * dataname1, const char* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteInt32As(const int32_t typecode,const char *filename1,const char * dataname1, const int32_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteInt64As(const int32_t typecode,const char *filename1,const char * dataname1, const int64_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteFloat32As(const int32_t typecode,const char *filename1,const char * dataname1, const efloat32* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteFloat64As(const int32_t typecode,const char *filename1,const char * dataname1, const efloat64* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteInt16As(const int32_t typecode,const char *filename1,const char * dataname1, const int16_t * data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteInt8As(const int32_t typecode,const char *filename1,const char * dataname1, const int8_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteUInt8As(const int32_t typecode,const char *filename1,const char * dataname1, const uint8_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteUInt16As(const int32_t typecode,const char *filename1,const char * dataname1, const uint16_t * data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteUInt32As(const int32_t typecode,const char *filename1,const char * dataname1, const  uint32_t* data, const char* mode1, const char* dataunit, int64_t dim1);
int Ebf_WriteUInt64As(const int32_t typecode,const char *filename1,const char * dataname1, const uint64_t* data, const char* mode1, const char* dataunit, int64_t dim1);


/**
 * Write C-style null terminated string
 * @param filename1
 * @param dataname1
 * @param data
 * @param mode1
 */
int Ebf_WriteCString(const char *filename1,const char *dataname1,char* data, const char* mode1);



#ifdef __cplusplus
}
#endif



#endif /* EBF_H_ */





