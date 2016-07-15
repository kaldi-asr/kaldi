#ifndef __HEAD_H__
#define __HEAD_H__

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <time.h>
#include <assert.h>
#include <string.h>
#include "math.h"

using namespace std;

#define     SUCCESS             0
#define     FILEREADERROR       1
#define     ARGSPARSEERROR      2
#define     EOL                 -2
#define     FILLEDNULL          1e8
#define     CHECKNUM            9999999
#define     MINRANDINITVALUE    -0.1
#define     MAXRANDINITVALUE    0.1
// #define     CONSTLOGNORM        9
#define     NOTSTOREMODEL
// #define     NUM_THREAD          8
#define     MAX_WORD_LINE       4008
// #define     INPUTEMBEDDINGCPU
#define     FUDGE_FACTOR        1e-8
#define     MBUPDATEOUTPUTLAYER
// #define     RMSPROP
#define     RMSPROPCOFF         0.9
#define     LOGADD
typedef     map<string, int>    WORDMAP;
typedef     float               Real;

// #define     DEBUG
#endif
