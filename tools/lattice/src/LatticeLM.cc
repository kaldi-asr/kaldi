/*
 * LatticeLM.cc --
 *	Language model using lattice transition probabilities
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2003-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/LatticeLM.cc,v 1.9 2010/06/02 05:54:08 stolcke Exp $";
#endif

#include <stdlib.h>

#include "LatticeLM.h"
#include "Trellis.cc"
#include "Array.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_TRELLIS(NodeIndex);
#endif

#define DEBUG_PRINT_WORD_PROBS          2	/* from LM.cc */
#define DEBUG_PRINT_VITERBI		2
#define DEBUG_STATES			2
#define DEBUG_TRANSITIONS		4


LatticeLM::LatticeLM(Lattice &lat)
    : LM(lat.vocab), lat(lat),
      trellis(maxWordsPerLine + 2 + 1), savedLength(0)
{
}

Boolean
LatticeLM::read(File &file, Boolean limitVocab)
{
    return lat.readPFSG(file);
}

Boolean
LatticeLM::write(File &file)
{
    lat.writeCompactPFSG(file);
    return true;
}

/*
 * Compute the prefix probability of word string (in reversed order)
 * This is done by dynamic programming over the lattice structure,
 * filling in the trellis[] array from left to right.
 * Entry trellis[i][state].prob is the probability that of best path
 * from the lattice start node to the lattice node 'state', under the
 * constraint that the words match the LM context up to position i.
 * The total prefix probability is the column sum in trellis[],
 * summing over all states.
 * For efficiency reasons, we specify the last word separately.
 * If context == 0, reuse the results from the previous call.
 */
LogP
LatticeLM::prefixProb(VocabIndex word, const VocabIndex *context,
					    LogP &contextProb, TextStats &stats)
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
	 * lattice initial state
	 */
	VocabIndex initialContext[2];
	if (len > 0 && context[len - 1] == vocab.ssIndex()) {
	    initialContext[0] = context[len - 1];
	    initialContext[1] = Vocab_None;
	    prefix = len - 1;
	} else {
	    initialContext[0] = Vocab_None;
	    prefix = len;
	}

	/*
	 * Start the DP from scratch if the context has less than two items
	 * (including <s>).  This prevents the trellis from accumulating states
	 * over multiple sentences (which computes the right thing, but
	 * slowly).
	 */
	if (len > 1 &&
	    savedLength > 0 && savedContext[0] == initialContext[0])
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
	    trellis.clear();
	    trellis.setProb(lat.getInitial(), LogP_One);
	    trellis.step();

	    savedContext[0] = initialContext[0];
	    savedLength = 1;
	    pos = 1;
	}
    }

    LogP logMax = LogP_Zero;

    for ( ; prefix >= 0; pos++, prefix--) {
	/*
	 * Keep track of the fact that at least one state has positive
	 * probability, needed for handling of OOVs below.
	 */
	Boolean havePosProb = false;

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

	/*
	 * Put underlying LM in "running" state (for debugging etc.)
	 * only when processing the last (current) word to prevent
	 * redundant output.
	 */
	if (prefix == 0) {
	    running(wasRunning);
	}

	/*
	 * Iterate over all nodes for the previous position in trellis
	 */
	TrellisIter<NodeIndex> prevIter(trellis, pos - 1);

	NodeIndex prevState;
	LogP prevProb;

	while (prevIter.next(prevState, prevProb)) {
	    /*
	     * For each previous lattice node, follow all transitions
	     * that lead to follow-nodes that match the current word.
	     */
	    LatticeNode *prevNode = lat.findNode(prevState);
	    assert(prevNode != 0);

	    // three-fold lookahead to skip non-emitting nodes
	    LatticeFollowIter followIter(lat, *prevNode);

	    LatticeNode *followNode;
	    NodeIndex followState;
	    LogP weight;

	    while ((followNode = followIter.next(followState, weight))) {
		if (followNode->word == currWord) {
		    if (debug(DEBUG_TRANSITIONS)) {
			cerr << "POSITION = " << pos
			     << " FROM: " << prevState
			     << " TO: " << followState
			     << " WORD = " << vocab.getWord(currWord)
			     << " TRANSPROB = " << weight
			     << " INTLOG = " << LogPtoIntlog(weight)
			     << endl;
		    }

		    trellis.update(prevState, followState, weight);

		    if (weight != LogP_Zero) {
			havePosProb = true;
		    }

		    if (running() && debug(DEBUG_STATES)) {
			dout() << "[" << followState << "]";
		    }
		}
            }
	}

	if (prefix > 0 && !havePosProb) {
	    if (currWord == vocab.unkIndex()) {
		stats.numOOVs ++;
	    } else {
	        stats.zeroProbs ++;
	    }
	}
	
	trellis.step();
	prevPos = pos;
    }

    running(wasRunning);
    
    if (prevPos > 0) {
	NodeIndex maxState = trellis.max(prevPos-1);
	if (Map_noKeyP(maxState)) {
	    contextProb = LogP_Zero;
	} else {
	    contextProb = trellis.getMax(maxState, prevPos-1);
	}
    } else { 
	contextProb = LogP_One;
    }

    NodeIndex maxState = trellis.max(prevPos);
    if (Map_noKeyP(maxState)) {
	return LogP_Zero;
    } else {
	return trellis.getMax(maxState, prevPos);
    }
}

/*
 * The conditional word probability is computed as
 *	p(w1 .... wk)/p(w1 ... w(k-1)
 */
LogP
LatticeLM::wordProb(VocabIndex word, const VocabIndex *context)
{
    LogP cProb;
    TextStats stats;
    LogP pProb = prefixProb(word, context, cProb, stats);

    if (pProb == LogP_Zero) {
	return LogP_Zero;
    } else {
	return pProb - cProb;
    }
}

LogP
LatticeLM::wordProbRecompute(VocabIndex word, const VocabIndex *context)
{
    LogP cProb;
    TextStats stats;
    LogP pProb = prefixProb(word, 0, cProb, stats);

    if (pProb == LogP_Zero) {
	return LogP_Zero;
    } else {
	return pProb - cProb;
    }
}

/*
 * Sentence probabilities from indices
 *	This version computes the result directly using prefixProb to
 *	avoid recomputing prefix probs for each prefix.
 */
LogP
LatticeLM::sentenceProb(const VocabIndex *sentence, TextStats &stats)
{
    unsigned int len = vocab.length(sentence);
    LogP totalProb;

    /*
     * The debugging machinery is not duplicated here, so just fall back
     * on the general code for that.
     */
    if (debug(DEBUG_PRINT_WORD_PROBS)) {
	totalProb = LM::sentenceProb(sentence, stats);
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
	totalProb = prefixProb(reversed[0], reversed + 1, contextProb, stats);

	/* 
	 * OOVs and zeroProbs are updated by prefixProb()
	 */
	stats.numSentences ++;
	stats.prob += totalProb;
	stats.numWords += len;
    }

    if (debug(DEBUG_PRINT_VITERBI)) {
	len = trellis.where();
	makeArray(NodeIndex, bestStates, len);

	if (trellis.viterbi(bestStates, len) == 0) {
	    dout() << "Viterbi backtrace failed\n";
	} else {
	    dout() << "Lattice states:";

	    for (unsigned i = 0; i < len; i ++) {
		dout() << " " << bestStates[i];
	    }

	    dout() << endl;
	}
    }

    return totalProb;
}

