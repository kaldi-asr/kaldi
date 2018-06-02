/*
 * DynamicLM.h
 *	Dynamically loaded language model
 *
 * This model interprets global state change requests to load new LMs
 * on demand.  It can be used to implement simple adaptation schemes.
 * (Currently only ngram models are supported.)
 *
 * Copyright (c) 1995, SRI International, 2012 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/DynamicLM.h,v 1.5 2014-04-22 06:57:45 stolcke Exp $
 *
 */

#ifndef _DynamicLM_h_
#define _DynamicLM_h_

#include "LM.h"

class DynamicLM: public LM
{
public:
    DynamicLM(Vocab &vocab);

    virtual LogP wordProb(VocabIndex word, const VocabIndex *context);
    virtual void setState(const char *state);

    /*
     * Propagate changes to running state to component model
     */
    virtual Boolean running() const { return _running; }
    virtual Boolean running(Boolean newstate)
      { Boolean old = _running; _running = newstate; 
	if (myLM) myLM->running(newstate); return old; };

    /*
     * Propagate changes to Debug state to component model
     */
    void debugme(unsigned level)
	{ if (myLM) myLM->debugme(level); Debug::debugme(level); };
    ostream &dout() const { return Debug::dout(); };
    ostream &dout(ostream &stream)  /* propagate dout changes to sub-lms */
	{ if (myLM) myLM->dout(stream); return Debug::dout(stream); };

    /*
     * Propagate prefetching protocol to component model
     */
    unsigned prefetchingNgrams()
	{ if (myLM) return myLM->prefetchingNgrams(); else return 0; };
    Boolean prefetchNgrams(NgramCounts<Count> &ngrams)
	{ if (myLM) return myLM->prefetchNgrams(ngrams); else return true; };
    Boolean prefetchNgrams(NgramCounts<XCount> &ngrams)
	{ if (myLM) return myLM->prefetchNgrams(ngrams); else return true; };
    Boolean prefetchNgrams(NgramCounts<FloatCount> &ngrams)
	{ if (myLM) return myLM->prefetchNgrams(ngrams); else return true; };

protected:
    LM *myLM;			/* the current model */
    const char *currentState;	/* last state info */
};


#endif /* _DynamicLM_h_ */
