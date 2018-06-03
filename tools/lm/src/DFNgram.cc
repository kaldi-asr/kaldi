/*
 * DFNgram.cc --
 *	N-gram backoff language model for disfluencies
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2006 SRI International, 2015 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/DFNgram.cc,v 1.26 2015-06-26 08:06:36 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>

#include "DFNgram.h"
#include "Trellis.cc"
#include "Array.cc"

/*
 * Define null key for DF state trellis
 */
inline void
Map_noKey(DFstate &state)
{
    state = DFMAX;
}

inline Boolean
Map_noKeyP(const DFstate &state)
{
    return state == DFMAX;
}

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_TRELLIS(DFstate);
#endif

#define DEBUG_PRINT_WORD_PROBS          2	/* from LM.cc */
#define DEBUG_PRINT_VITERBI		2
#define DEBUG_STATE_PROBS		4

/*
 * Compile-time flags to disable certain types of disfluency models
 */
//#define NO_SDEL
//#define NO_DEL
//#define NO_REP
#define NO_FP		/* we know FP modeling is bad idea */

DFNgram::DFNgram(Vocab &vocab, unsigned int order)
    : Ngram(vocab, order), trellis(maxWordsPerLine + 2 + 1), savedLength(0)
{
    /*
     * Initialize special disfluency tokens
     */
    UHindex = vocab.addWord(UHstring);
    UMindex = vocab.addWord(UMstring);
    SDELindex = vocab.addWord(SDELstring);
    DEL1index = vocab.addWord(DEL1string);
    DEL2index = vocab.addWord(DEL2string);
    REP1index = vocab.addWord(REP1string);
    REP2index = vocab.addWord(REP2string);
}

DFNgram::~DFNgram()
{
}

void *
DFNgram::contextID(VocabIndex word, const VocabIndex *context, unsigned &length)
{
    /*
     * Due to the DP algorithm, we alway use the full context
     * (don't inherit Ngram::contextID()).
     */
    return LM::contextID(word, context, length);
}

LogP
DFNgram::contextBOW(const VocabIndex *context, unsigned length)
{
    /*
     * Due to the DP algorithm, we alway use the full context
     * (don't inherit Ngram::contextBOW()).
     */
    return LM::contextBOW(context, length);
}

Boolean
DFNgram::isNonWord(VocabIndex word)
{
    return LM::isNonWord(word) ||
	   word == SDELindex ||
	   word == DEL1index ||
	   word == DEL2index ||
	   word == REP1index ||
	   word == REP2index;
}

/*
 * Compute the prefix probability of word string (in reversed order)
 * taking hidden disfluency events into account.
 * This is done by dynamic programming, filling in the DPtrellis[]
 * array from left to right.
 * Entry DPtrellis[i][state].prob is the probability that of the first
 * i words while being in DFstate state.
 * The total prefix probability is the column sum in DPtrellis[],
 * summing over all states.
 * For efficiency reasons, we specify the last word separately.
 * If context == 0, reuse the results from the previous call.
 */
LogP
DFNgram::prefixProb(VocabIndex word, const VocabIndex *context,
							LogP &contextProb)
{
    /*
     * pos points to the column currently being computed (we skip over the
     *     initial <s> token)
     * prefix points to the tail of context that is used for conditioning
     *     the current word.
     */
    unsigned pos;
    int prefix;

    Boolean wasRunning = running(false);

    if (context == 0) {
	/*
	 * Reset the computation to the last iteration in the loop below
	 */
	pos = prevPos;
	prefix = 0;
	context = prevContext;

	trellis.init(pos);
    } else {
	unsigned len = Vocab::length(context);
	assert(len <= maxWordsPerLine);

	/*
	 * Save these for possible recomputation with different
	 * word argument, using same context
	 */
	prevContext = context;
	prevPos = 0;

	/*
	 * Initialization:
	 * The 0 column corresponds to the <s> prefix, and we are in the
	 * NODF state.
	 */

	if (len > 0 && context[len - 1] == vocab.ssIndex()) {
	    prefix = len - 1;
	} else {
	    prefix = len;
	}

	/*
	 * Start the DP from scratch if the context has less than two items
	 * (including <s>).  This prevents the trellis from accumulating states
	 * over multiple sentences (which computes the right thing, but
	 * slowly).
	 */
 	if (len > 1 &&
	    savedLength > 0 && savedContext[0] == context[prefix])
	{
	    /*
	     * Skip to the position where the saved
	     * context diverges from the new context.
	     */
	    for (pos = 1;
		 pos < savedLength && prefix > 0 &&
		     context[prefix - 1] == savedContext[pos];
		 pos ++, prefix --)
	    {
 		prevPos = pos;
	    }
  
	    savedLength = pos;
	    trellis.init(pos);
	} else {
	    /*
	     * Start a DP from scratch
	     */
	    trellis.init();
	    trellis.setProb(NODF, LogP_One);
	    trellis.step();

	    savedContext[0] = context[prefix];
	    savedLength = 1;
	    pos = 1;
	}
    }

    for ( ; prefix >= 0; pos++, prefix--) {
	/*
	 * Reinitialize the DP probs if the previous iteration has yielded
	 * prob zero.  This allows us to compute conditional probs based on
	 * truncated contexts.
	 */
	if (trellis.sumLogP(pos - 1) == LogP_Zero) {
	    trellis.init(pos - 1);
	    trellis.setProb(NODF, LogP_One);
	    trellis.step();
	}

        VocabIndex currWord;
	
	if (prefix == 0) {
	    currWord = word;
	} else {
	    currWord = context[prefix - 1];

	    /*
	     * Cache the context words for later shortcut processing
	     */
	    savedContext[savedLength ++] = currWord;
	}

	const VocabIndex *currContext = &context[prefix];

	/*
	 * Set up sentence-initial context SDEL case
	 */
        VocabIndex SDELcontext1[3];
	SDELcontext1[0] = context[prefix];
	SDELcontext1[1] = vocab.ssIndex();
	SDELcontext1[2] = Vocab_None;

	/*
	 * context for DEL1 case (restricted to 3gram)
	 */
	VocabIndex DEL1context[3];
	DEL1context[0] = context[prefix];
	DEL1context[1] = (pos > 2) ? context[prefix + 2] : Vocab_None;
	DEL1context[2] = Vocab_None;
	/*
	 * context for DEL2 case (restricted to 3gram)
	 */
	VocabIndex DEL2context[3];
	DEL2context[0] = context[prefix];
	DEL2context[1] = (pos > 3) ? context[prefix + 3] : Vocab_None;
	DEL2context[2] = Vocab_None;
       
	DFstate next;
#ifndef NO_FP
	/*
	 * Filled pauses (FP)
	 *
	 * The next word is always counted as a filled pause if it
	 * matches the UH or UM token. This puts all probability on
	 * the FP state which would otherwise go to the NODF state.
	 */
	if (currWord == UHindex || currWord == UMindex) {
	    next = FP;
	} else
#endif /* !NO_FP */
	{
	    next = NODF;
	}

	/*
	 * No disfluency (NODF)
	 *	switch off debugging after the standard transition,
	 *	to avoid lots of output
	 */
	if (prefix == 0) {
	    running(wasRunning);
	}
	trellis.update(NODF, next, Ngram::wordProb(currWord, currContext));
	running(false);

	    /*
	     * SDEL means the previous word was effectively preceded by <s>
	     */
	trellis.update(SDEL, next, Ngram::wordProb(currWord, SDELcontext1));
	    /*
	     * DEL1 removes the word before the last from the context
	     */
	if (pos > 0) {
	    trellis.update(DEL1, next, Ngram::wordProb(currWord, DEL1context));
	}
	    /*
	     * DEL2 removes the 2 words before the last from the context
	     */
	if (pos > 2) {
	    trellis.update(DEL2, next, Ngram::wordProb(currWord, DEL2context));
	}
	    /*
	     * FP is similar to next, except that we skip the last word
	     * in the context for prediction purposes
	     */
	if (pos > 1) {
	    trellis.update(FP, next, Ngram::wordProb(currWord, currContext + 1));
	}
	    /*
	     * REP1 causes the previous word to be ignored in the 
	     * context since it is a rep of the one before that
	     */
	if (pos > 1) {
	    trellis.update(REP1, next, Ngram::wordProb(currWord, currContext + 1));
	}
	    /*
	     * REP21 causes the previous 2 words to be ignored in the 
	     * context since it is a rep of the two before that
	     */
	if (pos > 3) {
	    trellis.update(REP21, next, Ngram::wordProb(currWord, currContext + 2));
	}

#ifndef NO_SDEL
	/*
	 * Sentence deletion (SDEL)
	 */
        VocabIndex *SDELcontext2 = SDELcontext1 + 1;
	LogP newProb = Ngram::wordProb(currWord, SDELcontext2);

	trellis.update(NODF, SDEL,
		    Ngram::wordProb(SDELindex, currContext) + newProb);
	trellis.update(SDEL, SDEL,
		    Ngram::wordProb(SDELindex, SDELcontext1) + newProb);
	if (pos > 1) {
	    trellis.update(DEL1, SDEL,
			Ngram::wordProb(SDELindex, DEL1context) + newProb);
	}
	if (pos > 2) {
	    trellis.update(DEL2, SDEL,
			Ngram::wordProb(SDELindex, DEL2context) + newProb);
	}
	if (pos > 1) {
	    trellis.update(FP, SDEL,
			Ngram::wordProb(SDELindex, currContext + 1) + newProb);
	}
	if (pos > 1) {
	    trellis.update(REP1, SDEL,
			Ngram::wordProb(SDELindex, currContext + 1) + newProb);
	}
	if (pos > 3) {
	    trellis.update(REP21, SDEL,
			Ngram::wordProb(SDELindex, currContext + 2)  + newProb);
	}
#endif /* !NO_SDEL */

#ifndef NO_DEL
	/*
	 * Deletions (DEL1, DEL2)
	 */
	if (pos > 1) {
	    trellis.update(NODF, DEL1,
			Ngram::wordProb(DEL1index, currContext) +
			Ngram::wordProb(currWord, currContext + 1));
	    trellis.update(SDEL, DEL1,
			Ngram::wordProb(DEL1index, SDELcontext1) +
			Ngram::wordProb(currWord, SDELcontext1 + 1));
	    /*
	     * repeated DEL1's are not currently meaningful
	     */
	    if (pos > 2) {
		trellis.update(FP, DEL1,
			    Ngram::wordProb(DEL1index, currContext + 1) +
			    Ngram::wordProb(currWord, currContext + 2));
	    }
	    /*
	     * REPs are not meaningful after DEL1
	     */
	}
	if (pos > 2) {
	    trellis.update(NODF, DEL2,
			 Ngram::wordProb(DEL2index, currContext) +
			 Ngram::wordProb(currWord, currContext + 2));
	    /*
	     * DEL2 after SDEL is no meaningful
	     */
	    /*
	     * repeated DEL2's are not currently meaningful
	     */
	    if (pos > 3) {
		trellis.update(FP, DEL2,
			     Ngram::wordProb(DEL1index, currContext + 1) +
			     Ngram::wordProb(currWord, currContext + 3));
	    }
	    /*
	     * REPs are not meaningful after DEL2
	     */
	}
#endif /* !NO_DEL */
	
#ifndef NO_REP
	/*
	 * Repetitions (REP1, REP2)
	 */
	if (pos > 1) {
	    if (currWord == currContext[0]) {
		trellis.update(NODF, REP1, Ngram::wordProb(REP1index, currContext));
	    }
	    if (currWord == SDELcontext1[0]) {
		trellis.update(SDEL, REP1, Ngram::wordProb(REP1index, SDELcontext1));
	    }
	    if (currWord == DEL1context[0] && pos > 1) {
		trellis.update(DEL1, REP1, Ngram::wordProb(REP1index, DEL1context));
	    }
	    if (currWord == DEL2context[0] && pos > 2) {
		trellis.update(DEL2, REP1, Ngram::wordProb(REP1index, DEL2context));
	    }
	    if (pos > 2 && currWord == currContext[1]) {
		trellis.update(FP, REP1, Ngram::wordProb(REP1index, currContext + 1));
	    }
	    if (pos > 2 && currWord == currContext[1]) {
		trellis.update(REP1, REP1,
			    Ngram::wordProb(REP1index, currContext + 1));
	    }
	    /*
	     * Repeated REP's are not currently handled here
	     */
	}
	if (pos > 2) {
	    if (currWord == currContext[1]) {
		trellis.update(NODF, REP2, Ngram::wordProb(REP2index, currContext));
	    }
	    /*
	     * REP2 is not possible after SDEL
	     */
	    /*
	     * REP's after deletions amount to substitutions,
	     * which currently aren't modelled here
	     */
	    if (pos > 3 && currWord == currContext[2]) {
		trellis.update(FP, REP2,
			    Ngram::wordProb(REP2index, currContext + 1));
	    }
	    /*
	     * consecutive REP's cannot be handled yet
	     */
	}
	/*
	 * The REP21 on states is used to carry over the presence of
	 * a REP2 the word before.
	 */
	if (pos > 3 && currWord == currContext[1]) {
	    trellis.update(REP2, REP21, 0.0);
	}
#endif /* !NO_REP */

	trellis.step();
	prevPos = pos;
    }

    running(wasRunning);

    if (running() && debug(DEBUG_STATE_PROBS)) {
	dout() << "\nNODF " << trellis.getLogP(NODF)
	       << " FP " << trellis.getLogP(FP)
	       << " SDEL " << trellis.getLogP(SDEL)
	       << " DEL1 " << trellis.getLogP(DEL1)
	       << " DEL2 " << trellis.getLogP(DEL2)
	       << " REP1 " << trellis.getLogP(REP1)
	       << " REP2 = " << trellis.getLogP(REP2)
	       << " REP21 = " << trellis.getLogP(REP21)
	       << endl;
    }
    
    if (prevPos > 0) {
	contextProb = trellis.sumLogP(prevPos - 1);
    } else { 
	contextProb = LogP_One;
    }
    return trellis.sumLogP(prevPos);
}

/*
 * The conditional word probability is computed as
 *	p(w1 .... wk)/p(w1 ... w(k-1)
 *
 * XXX: need to handle OOVs (currently result in NaN result)
 */
LogP
DFNgram::wordProb(VocabIndex word, const VocabIndex *context)
{
    LogP cProb;
    LogP pProb = prefixProb(word, context, cProb);
    return pProb - cProb;
}

LogP
DFNgram::wordProbRecompute(VocabIndex word, const VocabIndex *context)
{
    LogP cProb;
    LogP pProb = prefixProb(word, 0, cProb);
    return pProb - cProb;
}

/*
 * Sentence probabilities from indices
 *	This version computes the result directly using prefixProb so
 *	avoid recomputing prefix probs for each prefix.
 */
LogP
DFNgram::sentenceProb(const VocabIndex *sentence, TextStats &stats)
{
    unsigned int len = vocab.length(sentence);
    LogP totalProb;

    /*
     * The debugging machinery is not duplicated here, so just fall back
     * on the general code for that.
     */
    if (debug(DEBUG_PRINT_WORD_PROBS)) {
	totalProb = Ngram::sentenceProb(sentence, stats);
    } else {
	/*
	 * Contexts are represented most-recent-word-first.
	 * Also, we have to prepend the sentence-begin token,
	 * and append the sentence-end token.
	 */
	makeArray(VocabIndex, reversed, len + 2 + 1);
	len = prepareSentence(sentence, reversed, len);

	/*
	 * Invalidate cache (for efficiency only)
	 */
	savedLength = 0;

	LogP contextProb;
	totalProb = prefixProb(reversed[0], reversed + 1, contextProb);

	/*
	 * XXX: Cannot handle OOVs and zeroProbs yet.
	 */
	stats.numSentences ++;
	stats.prob += totalProb;
	stats.numWords += len;
    }

    if (debug(DEBUG_PRINT_VITERBI)) {
	len = trellis.where();
	makeArray(DFstate, bestStates, len);

	if (trellis.viterbi(bestStates, len) == 0) {
	    dout() << "Viterbi backtrace failed\n";
	} else {
	    dout() << "DF-events:";

	    for (unsigned i = 1; i < len; i ++) {
		dout() << " " << DFnames[bestStates[i]];
	    }

	    dout() << endl;
	}
    }

    return totalProb;
}

