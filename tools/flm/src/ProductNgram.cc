/*
 * ProductNgram.cc --
 *	Product N-gram backoff language models
 *      Jeff Bilmes <bilmes@ee.washington.edu>
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2012 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/flm/src/ProductNgram.cc,v 1.15 2014-08-29 21:35:47 frandsen Exp $";
#endif

#include "ProductNgram.h"
#include "TLSWrapper.h"

/*
 * Debug levels used
 */
#define DEBUG_PRINT_SENT_PROBS		1	/* from LM.cc */
#define DEBUG_WORD_PROB_SUMS		4

ProductNgram::ProductNgram(ProductVocab &vocab, unsigned order)
    : Ngram(vocab, order), vocab(vocab),
      fnSpecs(0), factoredStats(0), fngramLM(0)
{
}

ProductNgram::~ProductNgram()
{
    delete fngramLM;
    delete factoredStats;
    delete fnSpecs;
}

void
ProductNgram::memStats(MemStats &stats)
{
    stats.total += sizeof(*this);
    if (factoredStats != 0) factoredStats->memStats(stats);
    if (fngramLM != 0) fngramLM->memStats(stats);
}

Boolean
ProductNgram::read(File &file, Boolean limitVocab)
				// limitVocab is ignored for now
{
    delete fngramLM;		fngramLM = 0;
    delete fnSpecs;		fnSpecs = 0;
    delete factoredStats;	factoredStats = 0;

    // create and initialize FNgramSpecs object

    fnSpecs = new FNgramSpecs<FNgramCount>(file, vocab.fvocab, debuglevel());
    if (!fnSpecs) {
	//cerr << "Error creating fnspecs object\n";
	return false;
    }

    // create and initialize FNgramStats object

    FNgramStats *factoredStats = new FNgramStats(vocab.fvocab, *fnSpecs);
    assert(factoredStats != 0);
      
    factoredStats->debugme(debuglevel());

    // read in the counts, we need to do this for now.
    // TODO: change so that counts are not needed for ppl/rescoring.
    if (!factoredStats->read()) {
        //cerr << "error reading in counts in factor file\n";
	// @kw false positive: MLK.MIGHT (factoredStats)
	return false;
    }

    factoredStats->estimateDiscounts();
    factoredStats->computeCardinalityFunctions();
    factoredStats->sumCounts();

    // create and initialize FNgram object

    fngramLM = new FNgram(vocab.fvocab, *fnSpecs);
    assert(fngramLM != 0);

    // Don't enable debug levels >= 2 in FNgram since they just duplicate
    // debugging output in LM, violating the common format.
    fngramLM->debugme(debuglevel() > DEBUG_PRINT_SENT_PROBS ?
					    DEBUG_PRINT_SENT_PROBS :
					    debuglevel());

    // For now, set to values to get backwards compat with ngram.cc
    fngramLM->virtualBeginSentence = false;
    fngramLM->virtualEndSentence = false;
    fngramLM->noScoreSentenceBoundaryMarks = true;

    // Once the FNgram object is allocated, skipOOVs() and trustTotals()
    // return referenecs to its parameters, but we need to make sure that
    // values set before allocation are inherited by the new FNgram.
    fngramLM->skipOOVs = _skipOOVs;
    fngramLM->trustTotals = _trustTotals;

    if (!fngramLM->read()) {
	//cerr << "error reading in factored LM files\n";
	return false;
    }

    return true;
}


/*
 * The product LM forms its probability by looking at the current
 * factored LM and then multiplying (forming the product) of the various
 * factors currently loaded.
 */

static TLSWC(WidMatrix, wordProbBOwidMatrixTLS);

LogP
ProductNgram::wordProbBO(VocabIndex word, const VocabIndex *context,
							unsigned int clen)
{
    WidMatrix &widMatrix = TLSW_GET(wordProbBOwidMatrixTLS);

    assert(fngramLM != 0);
    if (fngramLM == 0) {
	return LogP_Zero;
    }

    // word is w[t]
    // context[0] is w[t-1]
    // context[1] is w[t-2]
    // and so on.
    const unsigned childPos = clen;

    // load up the word and the context.
    
    vocab.loadWidFactors(word,widMatrix[clen]);
    for (unsigned pos=1;pos<=clen;pos++) {
	vocab.loadWidFactors(context[pos-1],widMatrix[clen-pos]);
    }

    return fngramLM->wordProb(widMatrix,childPos,clen+1);
}

/*
 * Returns unique identifier for the context used by the LM (and its length).
 * We just create the hash code for all the context words within the
 * N-gram order given by the model.  This ignores the possibilities of hashing
 * collisions, but should work ok in practice.
 */
void *
ProductNgram::contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length)
{
    // ProductNgram uses the full context given to it,
    // up to the maximal order specified
    // (n-gram model uses at most n-1 words of context)
    length = Vocab::length(context);
    if (order - 1 < length) {
	length = order - 1;
    }

    // truncate context to used length and compute a hash code
    TruncatedContext usedContext(context, length);

    return (void *)LHash_hashKey(usedContext, 30);
}

LogP
ProductNgram::contextBOW(const VocabIndex *context, unsigned length)
{
    return LogP_One;
}

Prob
ProductNgram::wordProbSum(const VocabIndex *context)
{
    double total = 0.0;
    VocabIter iter(vocab);
    VocabIndex wid;

    /*
     * prob summing interrupts sequential processing mode
     */
    Boolean wasRunning = running(false);

    while (iter.next(wid)) {
	if (!isNonWord(wid)) {
	    Prob p = LogPtoProb(wordProb(wid, context));
	    total += p;
	    if (debug(DEBUG_WORD_PROB_SUMS)) {
		cerr << "summing: " << vocab.getWord(wid) << " " << p
		     << " total " << total << endl;
	    }
	}
    }

    running(wasRunning);
    return total;
}

void
ProductNgram::freeThread()
{
    TLSW_FREE(wordProbBOwidMatrixTLS);
}
