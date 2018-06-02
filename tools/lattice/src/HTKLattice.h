/*
 * HTKLattice.h --
 *	Information associated with HTK Standard Lattice Format
 *
 * Copyright (c) 2003-2010 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lattice/src/HTKLattice.h,v 1.19 2010/06/02 07:06:23 stolcke Exp $
 *
 */

#ifndef _HTKLattice_h_
#define _HTKLattice_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <math.h>

#include "Prob.h"
#include "Vocab.h"

const float HTK_undef_float = HUGE_VAL;
const unsigned HTK_undef_uint = (unsigned)-1;

extern const char *HTK_null_word;

extern double HTK_LogP_Zero;

/*
 * Options for score mapping
 */
typedef enum {
	mapHTKnone,
	mapHTKacoustic,
	mapHTKngram,
	mapHTKlanguage
} HTKScoreMapping;

/*
 * Lattice header information (plus some external parameters)
 */
class HTKHeader
{
public:
    HTKHeader();
    HTKHeader(double acscale, double lmscale, double ngscale,
	      double prscale, double duscale, double wdpenalty,
	      double x1scale, double x2scale, double x3scale,
	      double x4scale, double x5scale, double x6scale,
	      double x7scale, double x8scale, double x9scale);
    ~HTKHeader();
    HTKHeader &operator= (const HTKHeader &other);

    double logbase;
    double tscale;
    double acscale;
    double ngscale;
    double lmscale;
    double wdpenalty;
    double prscale;
    double duscale;
    double amscale;
    double x1scale;
    double x2scale;
    double x3scale;
    double x4scale;
    double x5scale;
    double x6scale;
    double x7scale;
    double x8scale;
    double x9scale;

    char *vocab;
    char *lmname;
    char *ngname;
    char *hmms;

    Boolean wordsOnNodes;
    Boolean scoresOnNodes;
    Boolean useQuotes;
};

/*
 * Word/Link-related information
 */
class HTKWordInfo
{
public:
    HTKWordInfo();
    HTKWordInfo(const HTKWordInfo &other);
    ~HTKWordInfo();
    HTKWordInfo &operator= (const HTKWordInfo &other);

    float time;				// start or end, depending on direction
    VocabIndex word;			// word ID
    VocabString wordLabel;		// word label string for OOVs
    unsigned var;			// pronunciation variant
    char *div;				// segmentation info
    char *states;			// state alignment

    LogP acoustic;			// acoustic model log score
    LogP ngram;				// ngram model log score
    LogP language;			// language model log score
    LogP pron;				// pronunciation log score
    LogP duration;			// duration log score
    LogP xscore1;			// extra score #1
    LogP xscore2;			// extra score #2
    LogP xscore3;			// extra score #3
    LogP xscore4;			// extra score #4
    LogP xscore5;			// extra score #5
    LogP xscore6;			// extra score #6
    LogP xscore7;			// extra score #7
    LogP xscore8;			// extra score #8
    LogP xscore9;			// extra score #9
    Prob posterior;			// posterior probability
};

ostream &operator<< (ostream &stream, HTKWordInfo &link);

#endif /* _HTKLattice_h_ */

