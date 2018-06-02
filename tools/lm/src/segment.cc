/*
 * segment --
 *	Segment a text using a hidden segment boundary model
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International, 2015 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Id: segment.cc,v 1.22 2015-06-26 08:06:36 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

#include "option.h"
#include "version.h"
#include "File.h"
#include "Vocab.h"
#include "VocabMap.h"
#include "Ngram.h"
#include "Trellis.cc"
#include "Array.cc"

#define DEBUG_TRANSITIONS	2

static int version = 0;
static unsigned order = 3;
static unsigned debug = 0;
static char *lmFile = 0;
static const char *textFile = "-";
static const char *sTag = 0;
static double bias = 1.0;
static int continuous = 0;
static int useUnknown = 0;
static int usePosteriors = 0;

const LogP LogP_PseudoZero = -100;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_STRING, "lm", &lmFile, "hidden token sequence model" },
    { OPT_UINT, "order", &order, "ngram order to use for lm" },
    { OPT_STRING, "text", &textFile, "text file to disambiguate" },
    { OPT_TRUE, "continuous", &continuous, "read input without line breaks" },
    { OPT_TRUE, "posteriors", &usePosteriors, "use posterior probabilities instead of Viterbi" },
    { OPT_TRUE, "unk", &useUnknown, "use <unk> tag for unknown words" },
    { OPT_UINT, "debug", &debug, "debugging level for lm" },
    { OPT_STRING, "stag", &sTag, "segment tag to use in output" },
    { OPT_FLOAT, "bias", &bias, "bias for segment model" },
};

typedef enum {
	NOS, S, NOSTATE
} SegmentState;

const char *stateNames[] = {
	"NOS", "S"
};

/*
 * Define null key for SegmentState trellis
 */
inline void
Map_noKey(SegmentState &state)
{
    state = NOSTATE;
}

inline Boolean
Map_noKeyP(const SegmentState &state)
{
    return state == NOSTATE;
}

/*
 * Segment a sentence by finding the SegmentState sequence 
 * that has the highest probability
 */
Boolean
segmentSentence(VocabIndex *wids, SegmentState *states, LM &lm, double bias,
							Prob *posteriors = 0)
{
    unsigned len = Vocab::length(wids);
    LogP logBias = ProbToLogP(bias);

    Trellis<SegmentState> trellis(len);

    Vocab &vocab = lm.vocab;

    VocabIndex sContext[2];
    sContext[0] = vocab.ssIndex();
    sContext[1] = Vocab_None;
    VocabIndex *noContext = &sContext[1];

    /*
     * Replace zero probs with a small non-zero prob, so the Viterbi
     * can still distinguish between paths of different probabilities.
     */
#define Z(p) (isZero ? LogP_PseudoZero : (p))

    /*
     * Prime the trellis as follows
     *   score for S   = p(S w1) = p(</s>) p(w1 | <s>)
     *   score for NOS = p(NOS w1) = Sum{w!=<s>} p(w w1)
     *                             = Sum p(w w1) - p(<s> w1)
     *                             = p(w1) - p(/s) p(w1 | <s>)
     */
    {
	Boolean isZero = (lm.wordProb(wids[0], noContext) == LogP_Zero);

	LogP probS = lm.wordProb(vocab.seIndex(), noContext) +
	             Z(lm.wordProb(wids[0], sContext));

	Prob probNOS = LogPtoProb(Z(lm.wordProb(wids[0], noContext))) -
		       LogPtoProb(probS);

	if (probNOS < 0.0) {
	    cerr << "warning: p(w1) < p(<s> w1)) \n";
	    probNOS = 0.0;
	}
        trellis.setProb(S, probS + logBias);
        trellis.setProb(NOS, ProbToLogP(probNOS));

	if (debug >= DEBUG_TRANSITIONS) {
	    cerr << 0 << ": p(NOS) = " << trellis.getProb(NOS)
			<< ", P(S) = " << trellis.getProb(S) << endl;
	}
    }

    unsigned pos = 1;
    while (wids[pos] != Vocab_None) {
	
	trellis.step();

	Boolean isZero = (lm.wordProb(wids[pos], noContext) == LogP_Zero);

	/*
	 * Iterate over all combinations of hidden tags for the previous two
	 * and the current word
	 * XXX: This allows 4-grams to be used as LM, but the segmentation
	 * state prior to the last word is not considered, so this is 
	 * only an approximation.
	 */
	VocabIndex context[4];

	context[0] = wids[pos-1];
	context[1] = pos > 1 ? wids[pos-2] : Vocab_None;
	context[2] = pos > 2 ? wids[pos-3] : Vocab_None;
	context[3] = Vocab_None;

	trellis.update(NOS, NOS, Z(lm.wordProb(wids[pos], context)));

	context[1] = vocab.ssIndex();
	context[2] = Vocab_None;
	trellis.update(S, NOS, Z(lm.wordProb(wids[pos], context)) + logBias);

	context[1] = pos > 1 ? wids[pos-2] : Vocab_None;
	context[2] = pos > 2 ? wids[pos-3] : Vocab_None;
	trellis.update(NOS, S, lm.wordProb(vocab.seIndex(), context) +
	                       Z(lm.wordProb(wids[pos], sContext)));

	context[1] = vocab.ssIndex();
	context[2] = Vocab_None;
	trellis.update(S, S, lm.wordProb(vocab.seIndex(), context) +
	                     Z(lm.wordProb(wids[pos], sContext)) + logBias);

	if (debug >= DEBUG_TRANSITIONS) {
	    cerr << pos << ": p(NOS) = " << trellis.getProb(NOS)
			<< ", P(S) = " << trellis.getProb(S) << endl;
	}

	pos ++;
    }

    unsigned int numWords = pos;

    /*
     * Start the backward pass (only if we want to compute posteriors)
     */
    if (posteriors != 0)  {
	pos --;
	if (pos > 0) {
	    trellis.initBack(pos);
	    trellis.setBackProb(S, LogP_One);
	    trellis.setBackProb(NOS, LogP_One);
	}

	while (pos > 0) {
	    trellis.stepBack();

	    Boolean isZero = (lm.wordProb(wids[pos], noContext) == LogP_Zero);

	    /*
	     * Backwards transitions follow the same pattern for forward ones ...
	     * (see above)
	     * Note pos still points at the word following the current position,
	     * which is convenient.
	     */
	    VocabIndex context[4];

	    context[0] = wids[pos-1];
	    context[1] = pos > 1 ? wids[pos-2] : Vocab_None;
	    context[2] = pos > 2 ? wids[pos-3] : Vocab_None;
	    context[3] = Vocab_None;

	    trellis.updateBack(NOS, NOS, Z(lm.wordProb(wids[pos], context)));

	    context[1] = vocab.ssIndex();
	    context[2] = Vocab_None;
	    trellis.updateBack(S, NOS, Z(lm.wordProb(wids[pos], context)) + logBias);

	    context[1] = pos > 1 ? wids[pos-2] : Vocab_None;
	    context[2] = pos > 2 ? wids[pos-3] : Vocab_None;
	    trellis.updateBack(NOS, S, lm.wordProb(vocab.seIndex(), context) +
				       Z(lm.wordProb(wids[pos], sContext)));

	    context[1] = vocab.ssIndex();
	    context[2] = Vocab_None;
	    trellis.updateBack(S, S, lm.wordProb(vocab.seIndex(), context) +
				 Z(lm.wordProb(wids[pos], sContext)) + logBias);

	    if (debug >= DEBUG_TRANSITIONS) {
		cerr << pos << ": p(NOS) = " << trellis.getBackProb(NOS)
			    << ", P(S) = " << trellis.getBackProb(S) << endl;
	    }

	    pos --;
	}

	/* 
	 * Compute posteriors by combining forward and backward probs
	 */
	for (pos = 0; pos < numWords; pos++) {
	   LogP Sprob = trellis.getLogP(S, pos) +
			    trellis.getBackLogP(S, pos);
	   LogP NOSprob = trellis.getLogP(NOS, pos) +
			    trellis.getBackLogP(NOS, pos);

	   posteriors[pos] = LogPtoProb(Sprob - AddLogP(Sprob, NOSprob));
	}
    }

    if (trellis.viterbi(states, numWords) != numWords) {
	return false;
    } else {
	return true;
    }
}

/*
 * Get one input sequence at a time, map it to wids, 
 * segment it, and print out the result
 */
void
segmentFile(File &file, LM &lm, double bias)
{
    char *line;

    while ((line = file.getline())) {
	VocabString sentence[maxWordsPerLine];
	unsigned numWords = Vocab::parseWords(line, sentence, maxWordsPerLine);

	if (numWords == maxWordsPerLine) {
	    file.position() << "too many words per sentence\n";
	} else {
	    VocabIndex wids[maxWordsPerLine + 1];
	    SegmentState states[maxWordsPerLine + 1];
	    Prob posteriors[maxWordsPerLine];

	    if (useUnknown) {
		lm.vocab.getIndices(sentence, wids, maxWordsPerLine,
							lm.vocab.unkIndex());
	    } else {
		lm.vocab.addWords(sentence, wids, maxWordsPerLine);
	    }

	    wids[numWords] = Vocab_None;

	    if (!segmentSentence(wids, states, lm, bias,
				    usePosteriors ? posteriors : 0))
	    {
		file.position() << "viterbi failed\n";
	    } else {
	        unsigned len = Vocab::length(wids);
		
		for (unsigned i = 0; i < len; i++) {
		    /*
		     * If we're using posteriors for segmentation, check that
		     * p > 0.5.  Otherwise use the Viterbi segmentation.
		     */
		    // @kw false positive: UNINIT.STACK.ARRAY.MIGHT (posteriors)
		    if (usePosteriors ?
				(posteriors[i] > 0.5) :
				(states[i] == S))
		    {
			cout << sTag << " ";
		    }
		    cout << lm.vocab.getWord(wids[i]);
		    if (i != len - 1) {
			cout << " ";
		    }
		}
		cout << endl;
	    }
	}
    }
}

/*
 * Read file as one input sentence, map it to wids, 
 * segment it, and print out the result
 */
void
segmentFileContinuous(File &file, LM &lm, double bias)
{
    char *line;
    Array<VocabIndex> wids;

    unsigned lineStart = 0; // index into the above to mark the offset for the 
			    // current line's data

    while ((line = file.getline())) {
	VocabString words[maxWordsPerLine];
	unsigned numWords =
		Vocab::parseWords(line, words, maxWordsPerLine);

	if (numWords == maxWordsPerLine) {
	    file.position() << "too many words per line\n";
	} else {
	    // This effectively allocates more space
	    wids[lineStart + numWords] = Vocab_None;

	    if (useUnknown) {
		lm.vocab.getIndices(words, &wids[lineStart], numWords,
							lm.vocab.unkIndex());
	    } else {
		lm.vocab.addWords(words, &wids[lineStart], numWords);
	    }
	    
	    lineStart += numWords;
	}
    }

    if (lineStart == 0) {
	return;
    }


    // This implicitly allocates enough space for the state and posterior vector
    Array<SegmentState> states;
    states[lineStart] = NOSTATE;

    Array<Prob> posteriors;
    if (usePosteriors) {
	posteriors[lineStart] = 0.0;
    }

    if (!segmentSentence(&wids[0], &states[0], lm, bias,
			usePosteriors ? &posteriors[0] : 0))
    {
	file.position() << "viterbi failed\n";
    } else {
	for (unsigned i = 0; i < lineStart; i++) {
	    if (usePosteriors ?
			(posteriors[i] > 0.5) :
			(states[i] == S))
	    {
		cout << sTag << endl;
	    }
	    cout << lm.vocab.getWord(wids[i]);

	    if (usePosteriors) {
		cout << " " << posteriors[i];
	    }

	    cout << endl;
	}
    }
}

int
main(int argc, char **argv)
{
    setlocale(LC_CTYPE, "");
    setlocale(LC_COLLATE, "");

    Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (version) {
	printVersion(RcsId);
	exit(0);
    }

    /*
     * Construct language model
     */
    Vocab vocab;
    LM    *lm;

    if (lmFile) {
	File file(lmFile, "r");

	lm = new Ngram(vocab, order);
	assert(lm != 0);

	lm->debugme(debug);
	lm->read(file);
    } else {
	cerr << "need a language model\n";
	exit(1);
    }

    if (!sTag) {
	sTag = vocab.getWord(vocab.ssIndex());
	if (!sTag) {
		cerr << "couldn't find a segment tag in LM\n";
		exit(1);
	}
    }

    if (textFile) {
	File file(textFile, "r");

	if (continuous) {
	    segmentFileContinuous(file, *lm, bias);
	} else {
	    segmentFile(file, *lm, bias);
	}
    }

#ifdef DEBUG
    delete lm;
    return 0;
#endif /* DEBUG */

    exit(0);
}


