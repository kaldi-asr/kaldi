/*
 * NBest.h --
 *	N-best lists
 *
 * Copyright (c) 1995-2012 SRI International, 2012-2013 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/NBest.h,v 1.49 2016/06/17 00:11:06 victor Exp $
 *
 */

#ifndef _NBest_h_
#define _NBest_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif

#include "Boolean.h"
#include "Prob.h"
#include "Counts.h"
#include "File.h"
#include "Map.h"
#include "Vocab.h"
#include "Array.h"
#include "LM.h"
#include "MemStats.h"
#include "Debug.h"
#include "Bleu.h"

#undef valid		/* avoids conflict with class member on some systems */

/* 
 * Magic string headers identifying Decipher N-best lists
 */
const char nbest1Magic[] = "NBestList1.0";
const char nbest2Magic[] = "NBestList2.0";

typedef float NBestScore;			/* same as LogP */

typedef float NBestTimestamp;

/*
 * For Bleu computation
 */
struct BleuCount {
    unsigned short correct[MAX_BLEU_NGRAM];
};

/*
 * Optional detailed information associated with words in N-best lists
 */
class NBestWordInfo {
public:
    NBestWordInfo();
    ~NBestWordInfo();
    NBestWordInfo &operator= (const NBestWordInfo &other);

    void write(File &file);			// write info to file
    Boolean parse(const char *s);		// parse info from string
    void invalidate();				// invalidate info
    Boolean valid() const;			// check that info is valid
    void merge(const NBestWordInfo &other, Prob otherPosterior = 0.0);
						// combine two pieces of info

    VocabIndex word;
    NBestTimestamp start;
    NBestTimestamp duration;
    LogP acousticScore;
    LogP languageScore;
    LogP confidenceScore;
    LogP confidenceScore2;
    LogP confidenceScore3;
    char *phones;
    char *phoneDurs;
    /*
     * The following two are used optionally when used as input to 
     * WordMesh::wordAlign() to encode case where the word/transition
     * posteriors differ from the overall hyp posteriors.
     */
    Prob wordPosterior;				// word posterior probability
    Prob transPosterior;			// transition to next word p.p.

    /*
     * Utility functions
     */
    static unsigned length(const NBestWordInfo *words);
    static NBestWordInfo *copy(NBestWordInfo *to, const NBestWordInfo *from);
    static VocabIndex *copy(VocabIndex *to, const NBestWordInfo *from);
};

extern const char *phoneSeparator;	// used for phones & phoneDurs strings
extern const NBestTimestamp frameLength; // quantization unit of word timemarks

/*
 * Support for maps with (const NBestWordInfo *) as keys
 */

size_t LHash_hashKey(const NBestWordInfo *key, unsigned maxBits);
const NBestWordInfo *Map_copyKey(const NBestWordInfo *key);
void Map_freeKey(const NBestWordInfo *key);
Boolean LHash_equalKey(const NBestWordInfo *key1, const NBestWordInfo *key2);
int SArray_compareKey(const NBestWordInfo *key1, const NBestWordInfo *key2);


/*
 * A hypothesis in an N-best list with associated info
 */
class NBestHyp {
public:
    NBestHyp();
    ~NBestHyp();
    NBestHyp &operator= (const NBestHyp &other);

    void rescore(LM &lm, double lmScale, double wtScale);
    void decipherFix(LM &lm, double lmScale, double wtScale);
    void reweight(double lmScale, double wtScale, double amScale = 1.0);

    Boolean parse(char *line, Vocab &vocab, unsigned decipherFormat = 0,
			LogP2 acousticOffset = 0.0,
			const char *multiChar = 0, Boolean backtrace = false);

    Boolean parseSRInterpFormat(char *line, Vocab &vocab, LHash<VocabString, NBestScore>& scores);

    void write(File &file, Vocab &vocab, Boolean decipherFormat = true,
						    LogP2 acousticOffset = 0.0);

    Count getNumWords() { return numWords; }
    static void freeThread();

    VocabIndex *words;
    NBestWordInfo *wordInfo;
    LogP2 acousticScore;
    LogP2 languageScore;
    Count numWords;
    LogP totalScore;
    Prob posterior;
    FloatCount numErrors;
    unsigned rank;
    BleuCount *bleuCount;
    unsigned closestRefLeng;
};

class NBestList: public Debug
{
public:
    NBestList(Vocab &vocab, unsigned maxSize = 0,
			Boolean multiwords = false, Boolean backtrace = false);
    NBestList(Vocab &vocab, unsigned maxSize,
			const char *multiChar, Boolean backtrace = false);
    virtual ~NBestList() {};

    static const unsigned initialSize;

    unsigned numHyps() { return _numHyps; };
    NBestHyp &getHyp(unsigned number) { return hypList[number]; };
    unsigned addHyp(NBestHyp &hyp) {
    	memcpy(&hypList[_numHyps++], &hyp, sizeof(hyp));
	memset(&hyp, 0, sizeof(hyp));
	return _numHyps; }
      
    void sortHyps();
    void sortHypsBySentenceBleu(unsigned order);
    float sortHypsByErrorRate();

    void rescoreHyps(LM &lm, double lmScale, double wtScale);
    void decipherFix(LM &lm, double lmScale, double wtScale);
    void reweightHyps(double lmScale, double wtScale, double amScale = 1.0);
    void computePosteriors(double lmScale, double wtScale,
					double postScale, double amScale = 1.0);
    void removeNoise(LM &lm);

    unsigned wordError(const VocabIndex *words,
				unsigned &sub, unsigned &ins, unsigned &del,
				float weight = 0.0);

    double minimizeWordError(VocabIndex *words, unsigned length,
				double &subs, double &inss, double &dels,
				unsigned maxRescore = 0, Prob postPrune = 0.0);

    void acousticNorm();
    void acousticDenorm();

    Boolean read(File &file);
    Boolean readSRInterpFormat(File & file, LHash<VocabString,Array<NBestScore>* >& nbestScores);

    Boolean write(File &file, Boolean decipherFormat = true,
						unsigned numHyps = 0);
    void memStats(MemStats &stats);

    Vocab &vocab;
    LogP2 acousticOffset;

private:
    Array<NBestHyp> hypList;
    unsigned _numHyps;
    unsigned maxSize;
    const char *multiChar;	// multiword delimiter char (0 = no splitting)
    Boolean backtrace;		// keep backtrace information (if available)
};

#endif /* _NBest_h_ */
