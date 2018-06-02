/*
 * LoglinearMix.h --
 *	Log-linear Mixture language model
 *
 * Copyright (c) 2005 SRI International, 2013 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/LoglinearMix.h,v 1.5 2014-04-22 06:57:46 stolcke Exp $
 *
 */

#ifndef _LoglinearMix_h_
#define _LoglinearMix_h_

#include "LM.h"
#include "Array.h"
#include "Trie.h"

class LoglinearMix: public LM
{
public:
    LoglinearMix(Vocab &vocab, LM &lm1, LM &lm2, Prob prior = 0.5);
    LoglinearMix(Vocab &vocab, Array<LM *> &subLMs, Array<Prob> &priors);

    unsigned numLMs;		/* number of component LMs */
    Array<Prob> priors;		/* prior weights */
    Prob &prior;		/* backward compatibility: prior of lm1 */

    /*
     * LM interface
     */
    virtual LogP wordProb(VocabIndex word, const VocabIndex *context);
    virtual void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    virtual Boolean isNonWord(VocabIndex word);
    virtual void setState(const char *state);
    virtual Boolean addUnkWords();

    virtual Boolean const running() { return _running; }
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

    Trie<VocabIndex, LogP> denomProbs;		/* cached normalizers */
};

#endif /* _LoglinearMix_h_ */

