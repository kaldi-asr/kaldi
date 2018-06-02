/*
 * BayesMix.h --
 *	Bayesian Mixture language model
 *
 * A Bayesian mixture LM interpolates two LMs M_1 and M_2 according to a
 * local estimate of the posterior model probabilities.
 *
 *	p(w | context)  = p(M_1 | context) p(w | context, M_1) +
 *	                  p(M_2 | context) p(w | context, M_2)
 *
 * where p(M_i | context) is proportional to p(M_i) p(context | M_i)
 * and p(context | M_i) is either the full sentence prefix probability,
 * or a marginal probability of a truncated context.  For example, for
 * mixtures of bigram models, the p(context | M_i) would simply be the unigram 
 * probability of the last word according to M_i.
 *
 * Copyright (c) 1995-2002 SRI International, 2012-2013 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/BayesMix.h,v 1.19 2014-04-22 06:57:45 stolcke Exp $
 *
 */

#ifndef _BayesMix_h_
#define _BayesMix_h_

#include "LM.h"
#include "Ngram.h"
#include "Array.h"
#include "NgramProbArrayTrie.h"

class BayesMix: public LM
{
public:
    BayesMix(Vocab &vocab,
		    unsigned clength = 0, double llscale = 1.0);
    BayesMix(Vocab &vocab, LM &lm1, LM &lm2,
		    unsigned clength = 0, Prob prior = 0.5,
		    double llscale = 1.0);
    BayesMix(Vocab &vocab, Array<LM *> &subLMs, Array<Prob> &priors,
		    unsigned clength = 0, double llscale = 1.0);
    ~BayesMix();

    unsigned numLMs;		/* number of component LMs */
    Array<Prob> priors;		/* prior weights */
    Prob &prior;		/* backward compatibility: prior of lm1 */

    unsigned clength;		/* context length used for posteriors */
    double llscale;		/* log likelihood scaling factor */

    virtual Boolean read(File &file, Boolean limitVocab = false)
	{ return readMixLMs(file, limitVocab, false); };
				/* read list of mixture LMs and weights */

    Boolean readContextPriors(File &file, Boolean limitVocab = false);
				/* read and use context-dependent priors */
    Array<Prob> &findPriors(const VocabIndex *context);
				/* retrieve context-dependent priors */

    LM *subLM(unsigned i)
	{ return (i < numLMs)? subLMs[i] : 0; };

    /*
     * LM interface
     */
    virtual LogP wordProb(VocabIndex word, const VocabIndex *context);
    virtual void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    virtual Boolean isNonWord(VocabIndex word);
    virtual void setState(const char *state);
    virtual Boolean addUnkWords();

    virtual Boolean running() const { return _running; }
    virtual Boolean running(Boolean newstate);

    void debugme(unsigned level);
    ostream &dout() const { return Debug::dout(); };
    ostream &dout(ostream &stream);

    unsigned prefetchingNgrams();
    Boolean prefetchNgrams(NgramCounts<Count> &ngrams);
    Boolean prefetchNgrams(NgramCounts<XCount> &ngrams);
    Boolean prefetchNgrams(NgramCounts<FloatCount> &ngrams);

protected:
    Array<LM *> subLMs;				/* component models */
    Boolean deleteSubLMs;			/* need to delete sub lms */
    Boolean useContextPriors;
    NgramProbArrayTrie contextPriors;

    Boolean readMixLMs(File &file, Boolean limitVocab, Boolean ngramOnly);
};

/*
 * Specialize subclass mixing only Ngram models
 */
class NgramBayesMix: public BayesMix
{
public:
    NgramBayesMix(Vocab &vocab,
		    unsigned clength = 0, double llscale = 1.0)
	: BayesMix(vocab, clength, llscale) {};
    NgramBayesMix(Vocab &vocab, Ngram &lm1, Ngram &lm2,
		    unsigned clength = 0, Prob prior = 0.5,
		    double llscale = 1.0)
	: BayesMix(vocab, lm1, lm2, clength, prior, llscale) {};
    NgramBayesMix(Vocab &vocab, Array<Ngram *> &subLMs, Array<Prob> &priors,
		    unsigned clength = 0, double llscale = 1.0)
	: BayesMix(vocab, *(Array<LM *> *)&subLMs, priors, clength, llscale) {};

    Ngram *subLM(unsigned i)
	{ return (i < BayesMix::numLMs)? (Ngram *)BayesMix::subLMs[i] : 0; };

    virtual Boolean read(File &file, Boolean limitVocab = false)
	{ return BayesMix::readMixLMs(file, limitVocab, true); };
};

#endif /* _BayesMix_h_ */
