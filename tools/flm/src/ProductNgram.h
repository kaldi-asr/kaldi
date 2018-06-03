/*
 * ProductNgram.h --
 *	Product N-gram backoff language models
 *      Jeff Bilmes <bilmes@ee.washington.edu>
 *
 * Copyright (c) 1995-2007 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/flm/src/ProductNgram.h,v 1.10 2012/10/29 17:25:00 mcintyre Exp $
 *
 */

#ifndef _ProductNgram_h_
#define _ProductNgram_h_


#include "Ngram.h"
#include "ProductVocab.h"

#include "FNgram.h"

class ProductNgram: public Ngram
{
public:
    ProductVocab &vocab;			/* vocabulary */

    ProductNgram(ProductVocab &vocab, unsigned order);
    ~ProductNgram();

    virtual void memStats(MemStats &stats);
    virtual Boolean read(File &file, Boolean limitVocab = false);

    virtual Prob wordProbSum(const VocabIndex *context);

    virtual void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    virtual LogP contextBOW(const VocabIndex *context, unsigned length);

    // tie model parameters to underlying FNgram
    virtual Boolean &skipOOVs()
	{ return (fngramLM == 0) ? _skipOOVs : fngramLM->skipOOVs; };	
    virtual Boolean &trustTotals()
	{ return (fngramLM == 0) ? _trustTotals : fngramLM->trustTotals; };	

    // not implemented yet -- 
    // dummy functions to prevent inapproriate use of Ngram versions
    virtual Boolean writeWithOrder(File &file, unsigned int order)
	{ sorry("writeWithOrder"); return false; };
    virtual Boolean estimate(NgramStats &stats,
                        Count *mincount = 0,
                        Count *maxcounts = 0)
	{ sorry("estimate"); return false; };
    virtual Boolean estimate(NgramStats &stats, Discount **discounts)
	{ sorry("estimate"); return false; };
    virtual Boolean estimate(NgramCounts<FloatCount> &stats,
                                                        Discount **discounts)
	{ sorry("estimate"); return false; };
    virtual void mixProbs(Ngram &lm2, double lambda)
	{ sorry("mixProbs"); };
    virtual void mixProbs(Ngram &lm1, Ngram &lm2, double lambda)
	{ sorry("mixProbs"); };
    virtual void recomputeBOWs()
	{ sorry("recomputeBOWs"); };
    virtual void pruneProbs(double threshold, unsigned minorder = 2)
	{ sorry("pruneProbs"); };
    virtual void pruneLowProbs(unsigned minorder = 2)
	{ sorry("pruneLowProbs"); };
    virtual void rescoreProbs(LM &lm)
	{ sorry("rescoreProbs"); };

    // low-level functions that don't make sense here
    virtual unsigned int numNgrams(unsigned int n) { return 0; };
    LogP *findBOW(const VocabIndex *context) { return 0; };
    LogP *insertBOW(const VocabIndex *context) { return 0; };
    LogP *findProb(VocabIndex word, const VocabIndex *context) { return 0; };
    LogP *insertProb(VocabIndex word, const VocabIndex *context) { return 0; };
    void removeBOW(const VocabIndex *context) { return; };
    void removeProb(VocabIndex word, const VocabIndex *context) { return; };

    static void freeThread();
protected:
    virtual LogP wordProbBO(VocabIndex word, const VocabIndex *context,
			    unsigned int clen);

    // underlying FNgram objects used for defining and evaluating factors
    FNgramSpecs<FNgramCount> *fnSpecs;
    FNgramStats *factoredStats;
    FNgram *fngramLM;

    void sorry(const char *what) {
	cerr << "ProductNgram::" << what << " not implemented yet\n";
	exit(1);
    };
};

#endif /* _ProductNgram_h_ */
