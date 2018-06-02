/*
 * AdaptiveMix.h --
 *	Adaptive Mixture language model
 *
 * Copyright (c) 1998-2003 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/AdaptiveMix.h,v 1.8 2014-04-22 06:57:45 stolcke Exp $
 *
 */

#ifndef _AdaptiveMix_h_
#define _AdaptiveMix_h_

#include "LM.h"
#include "Array.h"

class AdaptiveMix: public LM
{
public:
    AdaptiveMix(Vocab &vocab, double decay = 1.0, double llscale = 1.0,
						    unsigned maxIters = 2);

    double decay;		/* history likelihood decay factor */
    double llscale;		/* log likelihood scaling factor */
    unsigned maxIters;		/* maximum number of iterations of EM */

    /*
     * LM interface
     */
    virtual LogP wordProb(VocabIndex word, const VocabIndex *context);
    virtual void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    virtual void setState(const char *state);

    virtual Boolean read(File &file, Boolean limitVocab = false);
    virtual Boolean write(File &file);

    /*
     * Propagate changes to running state to component models
     */
    virtual Boolean running() const { return _running; }
    virtual Boolean running(Boolean newstate) {
        Boolean old = _running; _running = newstate; 
	for (unsigned i = 0; i < numComps; i++) {
		compLMs[i]->running(newstate);
	}
	return old;
    };

    /*
     * Propagate changes to Debug state to component models
     */
    void debugme(unsigned level) {
	for (unsigned i = 0; i < numComps; i++) {
		compLMs[i]->debugme(level);
	}
	Debug::debugme(level);
    };
    ostream &dout() const { return Debug::dout(); };
    ostream &dout(ostream &stream) { /* propagate dout changes to sub-lms */
	for (unsigned i = 0; i < numComps; i++) {
		compLMs[i]->dout(stream);
	}
	return Debug::dout(stream);
    };

protected:
    unsigned numComps;			/* number of components */
    Array<LM *> compLMs;		/* components models */
    Array<Prob> priors;			/* priors for components */
    Array< Array<Prob> > histProbs;	/* history component probabilities */
    unsigned endOfHistory;		
    unsigned endOfSentence;
    Array<Prob> posteriors;		/* posteriors of components */

    Boolean accumulating;		/*  accumulate history likelihoods */

    void initPosteriors();		/* initialize posteriors */
    void computePosteriors();		/* update posteriors */
};

#endif /* _AdaptiveMix_h_ */
