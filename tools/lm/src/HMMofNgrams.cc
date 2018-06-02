/*
 * HMMofNgrams.cc --
 *	Hidden Markov Model of Ngram distributions
 * 
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1997-2010 SRI International, 2013-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/HMMofNgrams.cc,v 1.20 2016/04/09 06:53:01 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>

#include "HMMofNgrams.h"
#include "Trellis.cc"
#include "Array.cc"

#define DEBUG_PRINT_WORD_PROBS          2	/* from LM.cc */
#define DEBUG_READ_STATS 		3
#define DEBUG_PRINT_VITERBI		2
#define DEBUG_TRANSITIONS		4
#define DEBUG_STATE_PROBS		5

#define NO_LM		"."		/* pseudo-filename for null LM */
#define INLINE_LM	"-"		/* pseudo-file for inline LM */

const unsigned maxTransPerState = 1000;

HMMofNgrams::HMMofNgrams(Vocab &vocab, unsigned order)
    : LM(vocab), order(order), trellis(maxWordsPerLine + 2 + 1), savedLength(0)
{
    /*
     * Remove standard vocab items not applicable to state space
     */
    stateVocab.remove(stateVocab.unkIndex());
    stateVocab.remove(stateVocab.ssIndex());
    stateVocab.remove(stateVocab.seIndex());
    stateVocab.remove(stateVocab.pauseIndex());

    /*
     * Add initial, final states
     */
    initialState = stateVocab.addWord("INITIAL");
    states.insert(initialState);

    finalState = stateVocab.addWord("FINAL");
    states.insert(finalState);
}

HMMofNgrams::~HMMofNgrams()
{
    LHashIter<HMMIndex,HMMState> iter(states);

    HMMState *state;
    HMMIndex index;
    while ((state = iter.next(index))) {
	state->~HMMState();
    }
}

/*
 * Propagate changes to Debug state to component models
 */

void
HMMofNgrams::debugme(unsigned level)
{
    LHashIter<HMMIndex,HMMState> iter(states);

    HMMState *state;
    HMMIndex index;
    while ((state = iter.next(index))) {
	if (state->ngram) {
	    state->ngram->debugme(level);
	}
    }

    Debug::debugme(level);
}

ostream &
HMMofNgrams::dout(ostream &stream)
{
    LHashIter<HMMIndex,HMMState> iter(states);

    HMMState *state;
    HMMIndex index;
    while ((state = iter.next(index))) {
	if (state->ngram) {
	    state->ngram->dout(stream);
	}
    }

    return Debug::dout(stream);
}

/*
 * Read HMMofNgrams from file.
 * File format: 1 line per state, containing
 *
 *	state name
 *	Ngram model file name
 *	follow-state1 transitiion-prob1
 *	follow-state2 transitiion-prob2.
 *	etc.
 */
Boolean
HMMofNgrams::read(File &file, Boolean limitVocab)
{
    char *line;
    VocabString fields[maxTransPerState + 3];

    while ((line = file.getline())) {
	unsigned numFields =
			vocab.parseWords(line, fields, maxTransPerState + 3);

	if (numFields == maxTransPerState + 3) {
	    file.position() << "too many fields\n";
	    return false;
	}

	if (numFields < 2 || numFields % 2 != 0) {
	    file.position() << "wrong number of fields\n";
	    return false;
	}

	HMMIndex stateIndex = stateVocab.addWord(fields[0]);

	/*
	 * Clear all current transitions
	 */
	states.insert(stateIndex)->transitions.clear();

	/*
	 * Read transitions out of state
	 */
	for (unsigned i = 2; i < numFields; i += 2) {
	    HMMIndex toIndex = stateVocab.addWord(fields[i]);

	    states.insert(toIndex);

	    if (toIndex == initialState) {
		file.position() << "illegal transition to initial state\n";
		return false;
	    }

	    Prob prob;
	    if (!parseProb(fields[i + 1], prob)) {
		file.position() << "bad transition prob "
				<< fields[i + 1] << endl;
		return false;
	    }

	    *(states.find(stateIndex)->transitions.insert(toIndex)) = prob;
	}

	/*
	 * Read LM for state
	 */
	if (stateIndex == initialState || stateIndex == finalState) {
	    if (strcmp(fields[1], NO_LM) != 0) {
		file.position() << "ngram not allowed on initial/final state\n";
		return false;
	    }
	} else {
	    HMMState *state = states.insert(stateIndex);

	    /*
	     * Check for identity of the Ngram filenames.
	     * If they are identical (and not "-") assume that the models
	     * are the same and avoid reloading them.
	     */
	    if (state->ngramName &&
		strcmp(state->ngramName, INLINE_LM) != 0 &&
		strcmp(state->ngramName, fields[1]) == 0)
	    {
		if (debug(DEBUG_READ_STATS)) {
		    dout() << "reusing state ngram " << state->ngramName
			   << endl;
		}
	    } else {
		if (state->ngramName) free(state->ngramName);
		state->ngramName = strdup(fields[1]);
		assert(state->ngramName != 0);

		delete state->ngram;
		state->ngram = new Ngram(vocab, order);
		assert(state->ngram != 0);

		state->ngram->debugme(debuglevel());

		Boolean status;
		if (strcmp(state->ngramName, INLINE_LM) == 0) {
		    status = state->ngram->read(file, limitVocab);

		} else {
		    File ngramFile(state->ngramName, "r", false);

		    if (ngramFile.error()) {
			file.position() << "error opening Ngram file " 
					<< state->ngramName << endl;
			return false;
		    }

		    status = state->ngram->read(ngramFile, limitVocab);
		}

		if (!status) {
		    file.position() << "bad Ngram file " << fields[1] << endl;
		    return false;
		}
	    }
	}
    }

    /*
     * Ensure that all states (except initial and final) have
     * an LM defined
     */
    LHashIter<HMMIndex,HMMState> iter(states);

    HMMState *state;
    HMMIndex stateIndex;
    while ((state = iter.next(stateIndex))) {
	if (stateIndex != initialState && stateIndex != finalState &&
	    !state->ngram)
	{
	    file.position() << "no LM defined for state " 
			    << stateVocab.getWord(stateIndex) << endl;
	    return false;
	}
    }
    
    return true;
}

Boolean
HMMofNgrams::write(File &file)
{
    LHashIter<HMMIndex,HMMState> iter(states);

    HMMState *state;
    HMMIndex stateIndex;
    while ((state = iter.next(stateIndex))) {
	
	VocabString stateName = stateVocab.getWord(stateIndex);

	file.fprintf("%s %s", stateName, (state->ngram ? INLINE_LM : NO_LM));

	LHashIter<HMMIndex, Prob> transIter(state->transitions);

	HMMIndex toIndex;
	Prob *prob;

	while ((prob = transIter.next(toIndex))) {
	    file.fprintf(" %s %.*lf",
				stateVocab.getWord(toIndex),
				Prob_Precision, (double)*prob);
	}
	file.fprintf("\n");

	/*
	 * Output the Ngram inline
	 */
	if (state->ngram) {
	    Boolean status;

	    if (writeInBinary) {
		status = state->ngram->writeBinary(file);
	    } else {
		status = state->ngram->write(file);
	    }

	    if (!status) return false;
	}
    }

    return true;
}

/*
 * LM state change: re-read the model from file
 */
void
HMMofNgrams::setState(const char *state)
{
    char fileName[201];
    if (sscanf(state, " %200s ", fileName) != 1) {
	cerr << "no filename found in state info\n";
    } else {
	File lmFile(fileName, "r", false);

	if (lmFile.error()) {
	    cerr << "error opening HMM file " << fileName << endl;
	    return;
	}

 	if (!read(lmFile)) {
	    cerr << "failed to read HMM from " << fileName << endl;
	    return;
	}
    }
}

/*
 * Forward algorithm for prefix probability computation
 */
LogP
HMMofNgrams::prefixProb(VocabIndex word, const VocabIndex *context,
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
	 * INITIAL state.
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
	    trellis.setProb(initialState, LogP_One);
	    trellis.step();

	    savedContext[0] = context[prefix];
	    savedLength = 1;
	    pos = 1;
	}
    }

    for ( ; prefix >= 0; pos++, prefix--) {
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
	 * Keep track of the fact that at least one state has positive
	 * probability, needed for handling of OOVs below.
	 */
	Boolean havePosProb = false;

	TrellisIter<HMMIndex> prevIter(trellis, pos - 1);

	HMMIndex prevIndex;
	LogP prevProb;
	while (prevIter.next(prevIndex, prevProb)) {
	    HMMState *prevState = states.find(prevIndex);
	    assert(prevState != 0);

	    /*
	     * Stay in the same state, just emit a new word.
	     * (except in the initial and final states, which don't have
	     * LMs attached).
	     * Also, the </s> token cannot be emitted this way; it has
	     * to be matched by the FINAL state.
	     */
	    if (currWord != vocab.seIndex() && prevState->ngram) {
		LogP wProb =
			prevState->ngram->wordProb(currWord, currContext);

		trellis.update(prevIndex, prevIndex, wProb);

		if (prevProb != LogP_Zero && wProb != LogP_Zero) {
		    havePosProb = true;
		}

		if (debug(DEBUG_TRANSITIONS)) {
		    cerr << "position = " << pos
			 << " from: " << stateVocab.getWord(prevIndex)
			 << " to: " << stateVocab.getWord(prevIndex)
			 << " word: " << vocab.getWord(currWord)
			 << " wprob = " << wProb
			 << endl;
		}
	    }

	    /*
	     * Probability of exiting the current state is the
	     * </s> probability in the associated model.
	     * NOTE: We defer computing this until we actually need it
	     * to avoid work for states that don't permit transitions.
	     */
	    LogP eosProb;
	    Boolean haveEosProb = false;

	    LHashIter<HMMIndex,Prob> transIter(prevState->transitions);
	    HMMIndex toIndex;
	    Prob *transProb;

	    while ((transProb = transIter.next(toIndex))) {
		/*
		 * Emit </s> in the current state LM, 
		 * transition to next state,
		 * and emit the current word from that state.
		 */

		LogP wProb;	// word probability in new state
		/*
		 * The </s> token can only be matched by the FINAL state
		 * and vice-versa.
		 */
		if (currWord == vocab.seIndex()) {
		    if (toIndex == finalState) {
			wProb = LogP_One;
		    } else {
			continue;	// </s> only matches FINAL
		    }
		} else {
		    if (toIndex == finalState) {
			continue;	// FINAL only matches </s>
		    } else {
			HMMState *toState = states.find(toIndex);
			assert(toState != 0);

			/*
			 * For the overall model to be normalized we 
			 * force each state to emit at least one word.
			 * This means that the first word emitted by each 
			 * state has it's probability scaled by
			 * 1/(1 - P(</s>), the probability of an empty string.
			 * (This also means that the only way an empty string
			 * can be generated is by a direct transition from
			 * INITIAL to FINAL.)
			 */
			wProb = toState->ngram->wordProb(currWord, currContext)
				- SubLogP(LogP_One,
				    toState->ngram->wordProb(vocab.seIndex(),
							     currContext));
		    }
		}

		if (!haveEosProb) {
		    eosProb = prevState->ngram ? 
			prevState->ngram->wordProb(vocab.seIndex(), currContext) :
			LogP_One;
		    haveEosProb = true;
		}

		LogP localProb = eosProb + ProbToLogP(*transProb) + wProb;

		trellis.update(prevIndex, toIndex, localProb);

		if (prevProb != LogP_Zero && localProb != LogP_Zero) {
		    havePosProb = true;
		}


		if (debug(DEBUG_TRANSITIONS)) {
		    cerr << "position = " << pos
			 << " from: " << stateVocab.getWord(prevIndex)
			 << " to: " << stateVocab.getWord(toIndex)
			 << " word: " << vocab.getWord(currWord)
			 << " eosprob = " << eosProb
			 << " transprob = " << *transProb
			 << " wprob = " << wProb
			 << endl;
		}
	    }
	}

	/*
	 * Preserve the previous state probs if this iteration has yielded
	 * prob zero and we are not yet at the end of the prefix.
	 * This allows us to compute conditional probs based on
	 * truncated contexts, and to compute the total sentence probability
	 * leaving out the OOVs, as required by sentenceProb().
	 */
	if (prefix > 0 && !havePosProb) {
	    TrellisIter<HMMIndex> sIter(trellis, pos - 1);
	    HMMIndex index;
	    LogP prob;

	    while (sIter.next(index, prob)) {
		trellis.update(index, index, LogP_One);
	    }

	    if (currWord == vocab.unkIndex()) {
		stats.numOOVs ++;
	    } else {
		stats.zeroProbs ++;
	    }
	}

	trellis.step();
	prevPos = pos;
    }

    if (debug(DEBUG_STATE_PROBS)) {
	TrellisIter<HMMIndex> sIter(trellis, prevPos);
	HMMIndex index;
	LogP prob;

	dout() << endl;
	while (sIter.next(index, prob)) {
	    dout() << "P(" << stateVocab.getWord(index) <<") = "
		   << LogPtoProb(prob) << endl;
	}
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
 */
LogP
HMMofNgrams::wordProb(VocabIndex word, const VocabIndex *context)
{
    LogP cProb;
    TextStats stats;
    LogP pProb = prefixProb(word, context, cProb, stats);
    return pProb - cProb;
}

LogP
HMMofNgrams::wordProbRecompute(VocabIndex word, const VocabIndex *context)
{
    LogP cProb;
    TextStats stats;
    LogP pProb = prefixProb(word, 0, cProb, stats);
    return pProb - cProb;
}

/*
 * Sentence probabilities from indices
 *	This version computes the result directly using prefixProb to
 *	avoid recomputing prefix probs for each prefix.
 */
LogP
HMMofNgrams::sentenceProb(const VocabIndex *sentence, TextStats &stats)
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
	makeArray(VocabIndex, reversed, len + 2 + 1);

	/*
	 * Contexts are represented most-recent-word-first.
	 * Also, we have to prepend the sentence-begin token,
	 * and append the sentence-end token.
	 */
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
	makeArray(HMMIndex, bestStates, len + 2);

	if (trellis.viterbi(bestStates, len + 2) == 0) {
	    dout() << "Viterbi backtrace failed\n";
	} else {
	    dout() << "HMMstates:";

	    for (unsigned i = 1; i <= len + 1; i ++) {
		dout() << " " << stateVocab.getWord(bestStates[i]);
	    }

	    dout() << endl;
	}
    }

    return totalProb;
}
