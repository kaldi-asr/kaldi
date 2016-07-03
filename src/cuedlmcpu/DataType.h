#include <time.h>
#include <iostream>
#ifndef _DATATYPE_DEF_H__
#define _DATATYPE_DEF_H__
typedef  short int          Int16;
typedef unsigned short int  Uint16;
typedef long int            Int32;
typedef unsigned long int   Ulint32;

#define  MAX_CUED_STRINGLEN       1000
// if two clusters includes the speakers in the same session, then never merge them.
#define NOTMERGESPEAKERSINSAMESESSION
#define MIN_CUED_BICVALUE  -1e10

#define MAX_CUED_STRING 1000
#define INVALID_CUED_INT 1e8
// #define CACHESIZE 1000
#define CUED_CUDA
// #define CUDADEBUG
// #define CPUSIDE   // This macro runs GPU and CPU in the mean time when GPU is set
// #define TIMEINFO
// #define TIMEINFO1
#define CUED_DUMPENTROPY
// #define VARYLEARNRATE  // apply different learnning rate in layer1 and layer0
// #define USEMOMENTUM
#define CUED_WEIGHTDECAY_MB  1   // the number of minibatch when the weight matrix decays
#define CUED_RESETVALUE 0.1
// typedef double Real;
typedef float Real;
// #define NOSENTSPLICE      // rnnlm training without sentence splice, just for speed comparison
#define isdouble(Real)  (sizeof(Real) == sizeof(double))
typedef double direct_t;


#define foreach_row(_i,_m)    for (size_t _i = 0; _i < (_m)->rows(); _i++)
#define foreach_column(_j,_m) for (size_t _j = 0; _j < (_m)->cols(); _j++)
#define foreach_coord(_i,_j,_m) for (size_t _j = 0; _j < (_m)->cols(); _j++) for (size_t _i = 0; _i < (_m)->rows(); _i++)


class auto_timer
{
    timespec time_start, time_end;
    Real sec;
    Real nsec;
    Real acctime;
public:
    void start ()
    {
        clock_gettime (CLOCK_REALTIME, &time_start);
    }
    void end()
    {
         clock_gettime (CLOCK_REALTIME, &time_end);
    }
    void add()
    {
        end();
        if (time_end.tv_nsec - time_start.tv_nsec < 0)
        {
            nsec = 1000000000 + time_end.tv_nsec - time_start.tv_nsec;
            sec = time_end.tv_sec - time_start.tv_sec - 1;
        }
        else
        {
            nsec = time_end.tv_nsec - time_start.tv_nsec;
            sec = time_end.tv_sec - time_start.tv_sec;
        }
        acctime += sec + nsec * 1.0 / 1000000000;
    }
    Real stop()
    {
        end();
        if (time_end.tv_nsec - time_start.tv_nsec < 0)
        {
            nsec = 1000000000 + time_end.tv_nsec - time_start.tv_nsec;
            sec = time_end.tv_sec - time_start.tv_sec - 1;
        }
        else
        {
            nsec = time_end.tv_nsec - time_start.tv_nsec;
            sec = time_end.tv_sec - time_start.tv_sec;
        }
        return sec + nsec * 1.0/1000000000;
    }
    Real getacctime ()
    {
        return acctime;
    }
    void clear()
    {
        sec = 0.0;
        nsec = 0.0;
        acctime = 0.0;
    }
    auto_timer ()
    {
        sec = 0.0;
        nsec = 0.0;
        acctime = 0.0;
        time_start.tv_sec = 0;
        time_end.tv_sec = 0;
        time_start.tv_nsec = 0;
        time_end.tv_nsec = 0;
    }
};
#endif
