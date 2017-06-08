// $Id: util.h 363 2010-02-22 15:02:45Z mfederico $

#ifndef IRSTLM_UTIL_H
#define IRSTLM_UTIL_H


#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <assert.h>

using namespace std;

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

//random values between -1 and +1
#define MY_RAND (((float)random()/RAND_MAX)* 2.0 - 1.0)

#define UNUSED(x) { (void) x; }

#define LMTMAXLEV  20
#define MAX_LINE  100000
#define IRSTLM_DUB_DEFAULT  10000000
#define IRSTLM_REQUIREDMAXLEV_DEFAULT  1000

//0.000001 = 10^(-6)
//0.000000000001 = 10^(-12)
//1.000001 = 1+10^(-6)
//1.000000000001 = 1+10^(-12)
//0.999999 = 1-10^(-6)
//0.999999999999 = 1-10^(-12)
#define LOWER_SINGLE_PRECISION_OF_0 -0.000001
#define UPPER_SINGLE_PRECISION_OF_0 0.000001
#define LOWER_DOUBLE_PRECISION_OF_0 -0.000000000001
#define UPPER_DOUBLE_PRECISION_OF_0 0.000000000001
#define UPPER_SINGLE_PRECISION_OF_1 1.000001
#define LOWER_SINGLE_PRECISION_OF_1 0.999999
#define UPPER_DOUBLE_PRECISION_OF_1 1.000000000001
#define LOWER_DOUBLE_PRECISION_OF_1 0.999999999999

#define	IRSTLM_NO_ERROR		0
#define	IRSTLM_ERROR_GENERIC	1
#define	IRSTLM_ERROR_IO		2
#define	IRSTLM_ERROR_MEMORY	3
#define	IRSTLM_ERROR_DATA	4
#define	IRSTLM_ERROR_MODEL	5

#define BUCKET 10000
#define SSEED 50

typedef std::map< std::string, float > topic_map_t;

class ngram;
typedef unsigned int  ngram_state_t; //type for pointing to a full ngram in the table

class mfstream;

std::string gettempfolder();
std::string createtempName();
void createtempfile(mfstream  &fileStream, std::string &filePath, std::ios_base::openmode flags);

void removefile(const std::string &filePath);

void *MMap(int	fd, int	access, off_t	offset, size_t	len, off_t	*gap);
int Munmap(void	*p,size_t	len,int	sync);


// A couple of utilities to measure access time
void ResetUserTime();
void PrintUserTime(const std::string &message);
double GetUserTime();

void ShowProgress(long long current,long long total);

int parseWords(char *, const char **, int);
int parseline(istream& inp, int Order,ngram& ng,float& prob,float& bow);

void exit_error(int err, const std::string &msg="");

namespace irstlm{
	void* reallocf(void *ptr, size_t size);
}

//extern int tracelevel;
extern const int tracelevel;

#define TRACE_ERR(str) { std::cerr << str; }
#define VERBOSE(level,str) { if (tracelevel > level) { TRACE_ERR("DEBUG_LEVEL:" << level << "/" << tracelevel << " "); TRACE_ERR(str); } }
#define IFVERBOSE(level) if (tracelevel > level)

/*
#define _DEBUG_LEVEL TRACE_LEVEL

#define TRACE_ERR(str) { std::cerr << str; }
#define VERBOSE(level,str) { if (_DEBUG_LEVEL > level) { TRACE_ERR("DEBUG_LEVEL:" <<_DEBUG_LEVEL << " "); TRACE_ERR(str); } }
#define IFVERBOSE(level) if (_DEBUG_LEVEL > level)
*/

void MY_ASSERT(bool x);

#endif

