/*
 * LM.cc --
 *	Generic LM methods
 *
 */

#ifndef lint
static char LM_Copyright[] = "Copyright (c) 1995-2012 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char LM_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/LM.cc,v 1.103 2016/06/19 04:36:59 stolcke Exp $";
#endif

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <assert.h>
#include <string>
#include "TLSWrapper.h"
#include "tserror.h"
#include "MStringTokUtil.h"

#if !defined(_MSC_VER) && !defined(WIN32)
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/wait.h>

#define SOCKET_ERROR_STRING	srilm_ts_strerror(errno)

#define closesocket(s)	close(s)	// MS compatibility
#define INVALID_SOCKET	-1
#define SOCKET_ERROR	-1
typedef int	SOCKET;

#if __INTEL_COMPILER == 700
// old Intel compiler cannot deal with optimized byteswapping functions
#undef htons
#undef ntohs
#endif

#ifdef NEED_SOCKLEN_T
typedef int	socklen_t;
#endif

#else /* native MSWindows */

#include <winsock.h>

#ifdef _MSC_VER
#pragma comment(lib, "wsock32.lib")
#endif

typedef int socklen_t;

WSADATA wsaData;
int wsaInitialized = 0;

/* 
 * Windows equivalent of strerror()
 */
const char *
WSA_strerror(int errCode)
{
    char *errMsg;

    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM, 0,
			   errCode, 0, (LPSTR)&errMsg, 0, 0) == 0)
    {
	return "unknown error";
    } else {
	// This leaks some memory, but we don't care since code typically exits after error
	return errMsg;
    }
}

#define SOCKET_ERROR_STRING	WSA_strerror(WSAGetLastError())

#endif /* !_MSC_VER && !WIN32 */

#ifdef NEED_RAND48
extern "C" {
    double drand48();
}
#endif

#include "LM.h"
#include "RemoteLM.h"
#include "NgramStats.h"
#include "NBest.h"
#include "Array.cc"

/*
 * Debugging levels used in this file
 */
#define DEBUG_PRINT_DOC_PROBS		0
#define DEBUG_PRINT_SENT_PROBS		1
#define DEBUG_PRINT_WORD_PROBS		2
#define DEBUG_PRINT_PROB_SUMS		3
#define DEBUG_PRINT_PROB_RANKS		4

const char *defaultStateTag = "<LMstate>";

char ctsBuffer[100];		/* used by countToString() */

unsigned LM::initialDebugLevel = 0;

/*
 * Initialization
 *	The LM is created with a reference to a Vocab, so various
 *	LMs and other objects can share one Vocab.  The LM will typically
 *	add words to the Vocab as needed.
 */
LM::LM(Vocab &vocab)
    : vocab(vocab), noiseVocab(vocab)
{
    _running = false;
    reverseWords = false;
    addSentStart = true;
    addSentEnd = true;
    stateTag = defaultStateTag;
    writeInBinary = false;

    debugme(initialDebugLevel);
}

LM::~LM()
{
}

/*
 * Contextual word probabilities from strings
 *	The default method for word strings looks up the word indices
 *	for both the word and its context and gets its probabilities
 *	from the LM.
 */
LogP
LM::wordProb(VocabString word, const VocabString *context)
{
    unsigned int len = vocab.length(context);
    makeArray(VocabIndex, cids, len + 1);

    if (addUnkWords()) {
	vocab.addWords(context, cids, len + 1);
    } else {
	vocab.getIndices(context, cids, len + 1, vocab.unkIndex());
    }

    LogP prob = wordProb(vocab.getIndex(word, vocab.unkIndex()), cids);

    return prob;
}

/* Word probability with cached context
 *	Recomputes the conditional probability of a word using a context
 *	that is guaranteed to be identical to the last call to wordProb.
 * This implementation compute prob from scratch, but the idea is that
 * other language models use caches that depend on the context.
 */
LogP
LM::wordProbRecompute(VocabIndex word, const VocabIndex *context)
{
    return wordProb(word, context);
}

/*
 * Check if LM needs to add unknown words to vocabulary implicitly
 */
Boolean
LM::addUnkWords()
{
    return false;
}

/*
 * Non-word testing
 *	Returns true for pseudo-word tokens that don't correspond to
 *	observable events (e.g., context tags or hidden events).
 */
Boolean
LM::isNonWord(VocabIndex word)
{
    return vocab.isNonEvent(word);
}

/*  
 * Update the ranking statistics
 */
void
LM::updateRanks(LogP logp, const VocabIndex *context,
			FloatCount &r1, FloatCount &r5, FloatCount &r10,
			FloatCount weight)
{
    unsigned rank = 0;
    unsigned eq = 0;

    Prob prob = LogPtoProb(logp);
    
    /*
     * prob summing interrupts sequential processing mode
     */
    Boolean wasRunning = running(false);

    VocabIter iter(vocab);
    VocabIndex wid;
    Boolean first = true;

    while (iter.next(wid)) {
	if (!isNonWord(wid)) {
	    Prob p = LogPtoProb(first ?
				  wordProb(wid, context) :
				  wordProbRecompute(wid, context));

	    if (fabs(p - prob) < Prob_Epsilon) {
		eq ++;
	    } else if (p > prob) {
		rank ++;
	    }

	    first = false;

	    if (rank+eq/2 > 10) // NOTE: this depends on max rank being counted
		break;
	}
    }

    rank = rank+eq/2;
    
    if (rank < 10) {
	r10 += weight;
        if (rank < 5) {
	    r5 += weight;
	    if (rank < 1) {
		r1 += weight;
	    }
	}
    }

    running(wasRunning);
}

/*
 * Total probabilites
 *	For debugging purposes, compute the sum of all word probs
 *	in a context.
 */
Prob
LM::wordProbSum(const VocabIndex *context)
{
    Prob total = 0.0;
    VocabIter iter(vocab);
    VocabIndex wid;
    Boolean first = true;

    /*
     * prob summing interrupts sequential processing mode
     */
    Boolean wasRunning = running(false);

    while (iter.next(wid)) {
	if (!isNonWord(wid)) {
	    total += LogPtoProb(first ?
				wordProb(wid, context) :
				wordProbRecompute(wid, context));
	    first = false;
	}
    }

    running(wasRunning);
    return total;
}

/*
 * Sentence probabilities from strings
 *	The default method for sentences of word strings is to translate
 *	them to word index sequences and get its probability from the LM.
 */
LogP
LM::sentenceProb(const VocabString *sentence, TextStats &stats)
{
    unsigned int len = vocab.length(sentence);
    makeArray(VocabIndex, wids, len + 1);

    if (addUnkWords()) {
	vocab.addWords(sentence, wids, len + 1);
    } else {
	vocab.getIndices(sentence, wids, len + 1, vocab.unkIndex());
    }

    LogP prob = sentenceProb(wids, stats);

    return prob;
}

/*
 * Convenience function that reverses a sentence (for wordProb computation),
 * adds begin/end sentence tokens, and removes pause tokens.
 * It returns the number of words excluding these special tokens.
 */
unsigned
LM::prepareSentence(const VocabIndex *sentence, VocabIndex *reversed,
								unsigned len)
{
    unsigned i, j = 0;

    /*
     * Add </s> token if not already there.
     */
    if (addSentEnd && vocab.seIndex() != Vocab_None &&
        (len == 0 || sentence[reverseWords ? 0 : len - 1] != vocab.seIndex()))
    {
	reversed[j++] = vocab.seIndex();
    }

    for (i = 1; i <= len; i++) {
	VocabIndex word = sentence[reverseWords ? i - 1 : len - i];

	if (word == vocab.pauseIndex() || noiseVocab.getWord(word)) {
	    continue;
	}

	reversed[j++] = word;
    }

    /*
     * Add <s> token if not already there
     */
    if (len == 0 || sentence[reverseWords ? len - 1 : 0] != vocab.ssIndex()) {
	if (addSentStart) {
	    reversed[j++] = vocab.ssIndex();
	} else {
	    reversed[j++] = Vocab_None;
	}
    }
    reversed[j] = Vocab_None;

    return j - 2;
}

/*
 * Convenience functions that strips noise and pause tags from a words string
 */
VocabIndex *
LM::removeNoise(VocabIndex *words)
{
    unsigned from, to;

    for (from = 0, to = 0; words[from] != Vocab_None; from ++) {
	if (words[from] != vocab.pauseIndex() &&
	    !noiseVocab.getWord(words[from]))
	{
	    words[to++] = words[from];
	}
    }
    words[to] = Vocab_None;

    return words;
}

/*
 * Sentence probabilities from indices
 *	The default method is to accumulate the contextual word
 *	probabilities including that of the sentence end.
 */
LogP
LM::sentenceProb(const VocabIndex *sentence, TextStats &stats)
{
    TextStats myStats;

    unsigned int len = vocab.length(sentence);
    makeArray(VocabIndex, reversed, len + 2 + 1);
    unsigned int i;

    /*
     * output log probs with maximal precision
     */
    unsigned oldprec;
    if (debug(DEBUG_PRINT_WORD_PROBS)) {
	oldprec = dout().precision();
	dout().precision(LogP_Precision);
    }

    /*
     * Indicate to lm methods that we're in sequential processing
     * mode.
     */
    Boolean wasRunning = running(true);

    /*
     * Contexts are represented most-recent-word-first.
     * Also, we have to prepend the sentence-begin token,
     * and append the sentence-end token.
     */
    len = prepareSentence(sentence, reversed, len);

    /*
     * Prefetch ngrams if desired
     */
    unsigned prefetching = prefetchingNgrams();
    if (prefetching > 0) {
	NgramStats ngrams(vocab, prefetching);

        Vocab::reverse(reversed);

	/*	
	 * Extract ngrams corresponding to maximal word contexts
	 */
	for (unsigned i = 0; reversed[i] != Vocab_None; i++) {
	    unsigned minNgramLen;

	    if (i == 0) minNgramLen = 1;
	    else if (len - i < prefetching) minNgramLen = len - i;
	    else minNgramLen = prefetching;

	    ngrams.incrementCounts(reversed + i, minNgramLen);
	}

	prefetchNgrams(ngrams);

        Vocab::reverse(reversed);
    }

    for (i = len; (int)i >= 0; i--) {
	Prob probSum = 0.0;

	if (debug(DEBUG_PRINT_WORD_PROBS)) {
	    dout() << "\tp( " << vocab.getWord(reversed[i]) << " | "
		   << (reversed[i+1] != Vocab_None ?
		   		vocab.getWord(reversed[i+1]) : "")
		   << (i < len ? " ..." : " ") << ") \t= " ;

	    if (debug(DEBUG_PRINT_PROB_SUMS) &&
	  	!debug(DEBUG_PRINT_PROB_RANKS))
	    {
		/*
		 * XXX: because wordProb can change the state of the LM
		 * we need to compute wordProbSum first.
		 */
		probSum = wordProbSum(&reversed[i + 1]);
	    }
	}

	LogP prob = wordProb(reversed[i], &reversed[i + 1]);
        
	if (debug(DEBUG_PRINT_PROB_RANKS)) {
	    if (reversed[i] != vocab.seIndex()) {
		// exclude end of sentence marker
		updateRanks(prob, &reversed[i + 1],
			    myStats.r1, myStats.r5, myStats.r10);
	    } else {
		updateRanks(prob, &reversed[i + 1],
			    myStats.r1se, myStats.r5se, myStats.r10se);
	    }

	    myStats.rTotal += 1;
	}

	if (debug(DEBUG_PRINT_WORD_PROBS)) {
	    dout() << " " << LogPtoProb(prob) << " [ " << prob << " ]";
	    if (debug(DEBUG_PRINT_PROB_SUMS) && !debug(DEBUG_PRINT_PROB_RANKS)) {
		dout() << " / " << probSum;
		if (fabs(probSum - 1.0) > 0.0001) {
		    cerr << "\nwarning: word probs for this context sum to "
			 << probSum << " != 1 : " 
			 << (vocab.use(), &reversed[i + 1]) << endl;
		}
	    }
	    dout() << endl;
	}

	/*
	 * If the probability returned is zero but the
	 * word in question is <unk> we assume this is closed-vocab
	 * model and count it as an OOV.  (This allows open-vocab
	 * models to return regular probabilities for <unk>.)
	 * If this happens and the word is not <unk> then we are
	 * dealing with a broken language model that return
	 * zero probabilities for known words, and we count them
	 * as a "zeroProb".
	 */
	if (prob == LogP_Zero) {
	    if (reversed[i] == vocab.unkIndex()) {
		myStats.numOOVs ++;
	    } else {
		myStats.zeroProbs ++;

                myStats.posQuadLoss += 1.0;
                myStats.posAbsLoss += 1.0;
	    }
	} else {
	    myStats.prob += prob;

	    Prob loss = 1.0 - LogPtoProb(prob);
	    if (loss < 0.0) loss = 0.0;
	    myStats.posQuadLoss += loss*loss;
	    myStats.posAbsLoss += loss;
	}
    }

    running(wasRunning);

    if (debug(DEBUG_PRINT_WORD_PROBS)) {
	dout().precision(oldprec);
    }

    /*
     * Update stats with this sentence
     */
    if (reversed[0] == vocab.seIndex()) {
	myStats.numSentences = 1;
	myStats.numWords += len;
    } else {
	myStats.numWords += len + 1;
    }

    stats.increment(myStats);

    return myStats.prob;
}

/*
 * Compute joint probability of a word context (a reversed word sequence)
 */
LogP
LM::contextProb(const VocabIndex *context, unsigned clength)
{
    unsigned useLength = Vocab::length(context);
    LogP jointProb = LogP_One;

    if (clength < useLength) {
	useLength = clength;
    }

    /*
     * If the context is empty there is nothing left to do: return LogP_One 
     */
    if (useLength > 0) {
	/*
	 * Turn off debugging for contextProb computation
	 */
	Boolean wasRunning = running(false);

	TruncatedContext usedContext(context, useLength);

	/*
	 * Accumulate conditional probs for all words in used context
	 */
	for (unsigned i = useLength; i > 0; i--) {
	    VocabIndex word = usedContext[i - 1];

	    /*
	     * If we're computing the marginal probability of the unigram
	     * <s> context we have to look up </s> instead since the former
	     * has prob = 0.
	     */
	    if (i == useLength && word == vocab.ssIndex()) {
		word = vocab.seIndex();
	    }

	    LogP wprob = wordProb(word, &usedContext[i]);

	    /*
	     * If word is a non-event it has probability zero in the model,
	     * so the best we can do is to skip it.
	     * Note that above mapping turns <s> into a non-non-event, so
	     * it will be included.
	     */
	    if (wprob != LogP_Zero || !vocab.isNonEvent(word)) {
		jointProb += wprob;
	    }
	}

	running(wasRunning);
    }

    return jointProb;
}

/*
 * Compute an aggregate log probability, perplexity, etc., much like
 * sentenceProb, except that it uses counts instead of actual 
 * sentences.
 */
template <class CountT>
LogP
LM::countsProb(NgramCounts<CountT> &counts, TextStats &stats,
					unsigned countorder, Boolean entropy)
{
    unsigned prefetching = prefetchingNgrams();
    if (prefetching > 0) {
	prefetchNgrams(counts);
    }
    
    makeArray(VocabIndex, ngram, countorder + 1);

    LogP totalProb = 0.0;

    /*
     * output log probs with maximal precision
     */
    unsigned oldprec;
    if (debug(DEBUG_PRINT_WORD_PROBS)) {
	oldprec = dout().precision();
	dout().precision(LogP_Precision);
    }

    /*
     * Indicate to lm methods that we're in sequential processing
     * mode.
     */
    Boolean wasRunning = running(true);

    /*
     * Enumerate all counts up to the order indicated
     */
    for (unsigned i = 1; i <= countorder; i++ ) {
	// use sorted enumeration in debug mode only
	NgramCountsIter<CountT> ngramIter(counts, ngram, i,
					!debug(DEBUG_PRINT_WORD_PROBS) ? 0 :
							vocab.compareIndex());

	CountT *count;

	/*
	 * This enumerates all ngrams of the given order
	 */
	while ((count = ngramIter.next())) {
	    TextStats ngramStats;

	    /*
	     * Skip zero counts since they don't contribute anything to
	     * the probability
	     */
	    if (*count == 0) {
		continue;
	    }

	    /*
	     * reverse ngram for lookup
	     */
	    Vocab::reverse(ngram);

	    /*
	     * The rest of this loop is patterned after LM::sentenceProb()
	     */

	    if (debug(DEBUG_PRINT_WORD_PROBS)) {
		dout() << "\tp( " << vocab.getWord(ngram[0]) << " | "
		       << (vocab.use(), &ngram[1])
		       << " ) \t= " ;
	    }
	    LogP prob = wordProb(ngram[0], &ngram[1]);

	    LogP jointProb = !entropy ? LogP_One :
					contextProb(ngram, countorder);
	    Prob weight = *count * LogPtoProb(jointProb);

	    if (debug(DEBUG_PRINT_WORD_PROBS)) {
		dout() << " " << LogPtoProb(prob) << " [ " << prob;

		/* 
		 * Include ngram count if not unity, so we can compute the
		 * aggregate log probability from the output
		 */
		if (weight != 1.0) {
			dout() << " *" << weight;
		}
		dout() << " ]";

		if (debug(DEBUG_PRINT_PROB_RANKS)) {
		    if (ngram[0] != vocab.seIndex()) {
			// exclude end of sentence marker
			updateRanks(prob, &ngram[1],
				    ngramStats.r1, ngramStats.r5, ngramStats.r10, *count);
		    } else {
			updateRanks(prob, &ngram[1],
				    ngramStats.r1se, ngramStats.r5se, ngramStats.r10se, *count);
		    }

		    ngramStats.rTotal = *count;
		}

		if (debug(DEBUG_PRINT_PROB_SUMS) && !debug(DEBUG_PRINT_PROB_RANKS)) {
		    Prob probSum = wordProbSum(&ngram[1]);
		    dout() << " / " << probSum;
		    if (fabs(probSum - 1.0) > 0.0001) {
			cerr << "\nwarning: word probs for this context sum to "
			     << probSum << " != 1 : " 
			     << (vocab.use(), &ngram[1]) << endl;
		    }
		}
		dout() << endl;
	    }

	    /*
	     * ngrams ending in </s> are counted as sentences, all others
	     * as words.  This keeps the output compatible with that of
	     * LM::pplFile().
	     */
	    if (ngram[0] == vocab.seIndex()) {
		ngramStats.numSentences = *count;
	    } else {
		ngramStats.numWords = *count;
	    }

	    /*
	     * If the probability returned is zero but the
	     * word in question is <unk> we assume this is closed-vocab
	     * model and count it as an OOV.  (This allows open-vocab
	     * models to return regular probabilities for <unk>.)
	     * If this happens and the word is not <unk> then we are
	     * dealing with a broken language model that return
	     * zero probabilities for known words, and we count them
	     * as a "zeroProb".
	     */
	    if (prob == LogP_Zero) {
		if (ngram[0] == vocab.unkIndex()) {
		    ngramStats.numOOVs = *count;
		} else {
		    ngramStats.zeroProbs = *count;

                    ngramStats.posQuadLoss = 1.0 * *count;
                    ngramStats.posAbsLoss = 1.0 * *count;
		}
	    } else {
		totalProb +=
		    (ngramStats.prob = weight * prob);

		Prob loss = 1.0 - LogPtoProb(prob);
	        if (loss < 0.0) loss = 0.0;
	        ngramStats.posQuadLoss = loss*loss * *count;
	        ngramStats.posAbsLoss = loss * *count;
	    }

	    stats.increment(ngramStats);

	    Vocab::reverse(ngram);
	}
    }

    running(wasRunning);

    if (debug(DEBUG_PRINT_WORD_PROBS)) {
	dout().precision(oldprec);
    }

    /* 
     * If computing entropy set total number of events to 1 so that 
     * ppl computation reflects entropy.
     */
    if (entropy) {
	stats.numSentences = 0;
	stats.numWords = 1;
    }

    return totalProb;
}

/*
 * instantiate countsProb() for count types used
 */
template LogP
LM::countsProb(NgramCounts<Count> &counts, TextStats &stats,
                                    unsigned order, Boolean entropy);
#ifdef USE_XCOUNTS
template LogP
LM::countsProb(NgramCounts<NgramCount> &counts, TextStats &stats,
                                    unsigned order, Boolean entropy);
#endif
template LogP
LM::countsProb(NgramCounts<FloatCount> &counts, TextStats &stats,
                                    unsigned order, Boolean entropy);

/*
 * Perplexity from counts
 *	The escapeString is an optional line prefix that marks information
 *	that should be passed through unchanged.  This is useful in
 *	constructing rescoring filters that feed hypothesis strings to
 *	pplCountsFile(), but also need to pass other information to downstream
 *	processing.
 *	If the entropy flag is true, the count log probabilities will be 
 *	weighted by the joint probabilities on the ngrams.  I.e., the
 *	output will be p(w,h) log p(pw|h) for each ngram, and the overall 
 *	result will be the entropy of the conditional N-gram distribution.
 */
template <class CountT>
CountT
LM::pplCountsFile(File &file, unsigned order, TextStats &stats,
			const char *escapeString, Boolean entropy,
			NgramCounts<CountT> *counts)
{
    char *line;
    unsigned escapeLen = escapeString ? strlen(escapeString) : 0;
    unsigned stateTagLen = stateTag ? strlen(stateTag) : 0;

    VocabString words[maxNgramOrder + 1];
    makeArray(VocabIndex, wids, order + 1);
    TextStats sentenceStats;
    Boolean haveData = false;
    Boolean useCounts = (counts != 0);

    if (!useCounts) {
	counts = new NgramCounts<CountT>(vocab, order);
	assert(counts != 0);
    }

    while ((line = file.getline())) {

	if (escapeString && strncmp(line, escapeString, escapeLen) == 0) {
	    /*
	     * Output sentence-level statistics before each escaped line
	     */
	    if (haveData) {
		countsProb(*counts, sentenceStats, order, entropy);

		if (debug(DEBUG_PRINT_SENT_PROBS)) {
		    dout() << sentenceStats << endl;
		}

		stats.increment(sentenceStats);
		sentenceStats.reset();

		if (useCounts) {
		    counts->clear();
		} else {
		    delete counts;
		    counts = new NgramCounts<CountT>(vocab, order);
		    assert(counts != 0);
		}
		haveData = false;
	    }

	    dout() << line;

	    continue;
	}

	/*
	 * check for directives to change the global LM state
	 */
	if (stateTag && strncmp(line, stateTag, stateTagLen) == 0) {
	    /*
	     * pass the state info the lm to let it do whatever
	     * it wants with it
	     */
	    setState(&line[stateTagLen]);
	    continue;
	}

	CountT count;
	unsigned howmany =
		    counts->parseNgram(line, words, maxNgramOrder + 1, count);

	/*
	 * Skip this entry if the length of the ngram exceeds our 
	 * maximum order
	 */
	if (howmany == 0) {
	    file.position() << "malformed N-gram count or more than "
			    << maxNgramOrder << " words per line\n";
	    continue;
	} else if (howmany > order) {
	    continue;
	}

	/* 
	 * Map words to indices
	 */
	if (addUnkWords()) {
	    vocab.addWords(words, wids, order + 1);
	} else {
	    vocab.getIndices(words, wids, order + 1, vocab.unkIndex());
	}

	/*
	 *  Update the counts
	 */
	*counts->insertCount(wids) += count;

	haveData = true;
    }

    /* 
     * Output and update final sentence-level statistics
     */
    if (haveData) {
	countsProb(*counts, sentenceStats, order, entropy);

	if (debug(DEBUG_PRINT_SENT_PROBS)) {
	    dout() << sentenceStats << endl;
	}

	stats.increment(sentenceStats);
    }

    if (!useCounts) {
	delete counts;
    }

    return (CountT)stats.numWords;
}

/*
 * instantiate pplCountsFile() for count types used
 */
template Count
LM::pplCountsFile(File &file, unsigned order, TextStats &stats,
			const char *escapeString, Boolean entropy,
			NgramCounts<Count> *counts);
#ifdef USE_XCOUNTS
template NgramCount
LM::pplCountsFile(File &file, unsigned order, TextStats &stats,
			const char *escapeString, Boolean entropy,
			NgramCounts<NgramCount> *counts);
#endif
template FloatCount
LM::pplCountsFile(File &file, unsigned order, TextStats &stats,
			const char *escapeString, Boolean entropy,
			NgramCounts<FloatCount> *counts);

/*
 * Perplexity from text
 *	The escapeString is an optional line prefix that marks information
 *	that should be passed through unchanged.  This is useful in
 *	constructing rescoring filters that feed hypothesis strings to
 *	pplFile(), but also need to pass other information to downstream
 *	processing.
 *	If weighted is true, the input sentences are assumed to be preceded
 *	by weights that act as multipliers on the log likelihoods.
 */
unsigned int
LM::pplFile(File &file, TextStats &stats,
		const char *escapeString, Boolean weighted)
{
    char *line;
    unsigned escapeLen = escapeString ? strlen(escapeString) : 0;
    unsigned stateTagLen = stateTag ? strlen(stateTag) : 0;
    VocabString sentence[maxWordsPerLine + 1];
    unsigned totalWords = 0;
    unsigned sentNo = 0;
    TextStats documentStats;
    Boolean printDocumentStats = false;

    while ((line = file.getline())) {

	if (escapeString && strncmp(line, escapeString, escapeLen) == 0) {
            if (sentNo > 0 && debuglevel() == DEBUG_PRINT_DOC_PROBS) {
		dout() << documentStats << endl;
		documentStats.reset();
		printDocumentStats = true;
            }
	    dout() << line;
	    continue;
	}

	/*
	 * check for directives to change the global LM state
	 */
	if (stateTag && strncmp(line, stateTag, stateTagLen) == 0) {
	    /*
	     * pass the state info the lm to let it do whatever
	     * it wants with it
	     */
	    setState(&line[stateTagLen]);
	    continue;
	}

	sentNo ++;

	unsigned int numWords =
			vocab.parseWords(line, sentence, maxWordsPerLine + 1);

	if (numWords == maxWordsPerLine + 1) {
	    file.position() << "too many words per sentence\n";
	} else {
	    TextStats sentenceStats;
	    FloatCount weight;
	    VocabString *sentenceStart;

	    if (weighted) {
		/*
		 * Parse the weight string as a float count
		 */
		if (!stringToCount(sentence[0], weight)) {
		    file.position() << "bad sentence weight " << sentence[0] << endl;
		    continue;
		}
		sentenceStart = &sentence[1];
	    } else {
		weight = 1.0;
		sentenceStart = sentence;
	    } 
	

	    if (debug(DEBUG_PRINT_SENT_PROBS)) {
		if (weighted) {
		   dout() << weight << "* ";
		}
		dout() << sentenceStart << endl;
	    }
	    LogP prob = sentenceProb(sentenceStart, sentenceStats);

	    totalWords += numWords;

	    if (debug(DEBUG_PRINT_SENT_PROBS)) {
		TextStats weightedStats;
		weightedStats.increment(sentenceStats, weight);

		dout() << weightedStats << endl;
	    }

	    stats.increment(sentenceStats, weight);
	    documentStats.increment(sentenceStats, weight);
	}
    }

    if (printDocumentStats) {
	dout() << documentStats << endl;
    }

    return totalWords;
}

unsigned
LM::rescoreFile(File &file, double lmScale, double wtScale,
		   LM &oldLM, double oldLmScale, double oldWtScale,
		   const char *escapeString)
{
    char *line;
    unsigned escapeLen = escapeString ? strlen(escapeString) : 0;
    unsigned stateTagLen = stateTag ? strlen(stateTag) : 0;
    unsigned sentNo = 0;

    while ((line = file.getline())) {

	if (escapeString && strncmp(line, escapeString, escapeLen) == 0) {
	    fputs(line, stdout);
	    continue;
	}

	/*
	 * check for directives to change the global LM state
	 */
	if (stateTag && strncmp(line, stateTag, stateTagLen) == 0) {
	    /*
	     * pass the state info the lm to let let if do whatever
	     * it wants with it
	     */
	    setState(&line[stateTagLen]);
	    continue;
	}

	sentNo ++;

	/*
	 * parse an n-best hyp from this line
	 */
	NBestHyp hyp;

	if (!hyp.parse(line, vocab)) {
	    file.position() << "bad n-best hyp format\n";
	} else {
	    hyp.decipherFix(oldLM, oldLmScale, oldWtScale);
	    hyp.rescore(*this, lmScale, wtScale);
	    // hyp.write((File)stdout, vocab);
	    /*
	     * Instead of writing only the total score back to output,
	     * keep all three scores: acoustic, LM, word transition penalty.
	     * Also, write this in straight log probs, not bytelog.
	     */
	    fprintf(stdout, "%.*lg %.*lg %lu", LogP2_Precision, (double)hyp.acousticScore,
						LogP2_Precision, (double)hyp.languageScore,
						(unsigned long)hyp.numWords);
	    for (unsigned i = 0; hyp.words[i] != Vocab_None; i++) {
		fprintf(stdout, " %s", vocab.getWord(hyp.words[i]));
	    }
	    fprintf(stdout, "\n");
	}
    }
    return sentNo;
}

/*
 * Act as a server of N-gram probabilities, listening on port
 */
unsigned
LM::probServer(unsigned port, unsigned maxClients)
{
#ifdef SOCK_STREAM

#if defined(_MSC_VER) || defined(WIN32)
    if (!wsaInitialized) {
	int result = WSAStartup(MAKEWORD(2,2), &wsaData);
	if (result != 0) {
	    cerr << "could not initialize winsocket: " << SOCKET_ERROR_STRING << endl;
	    return 0;
	}
	wsaInitialized = 1;
    }
#endif /* _MSC_VER || WIN32 */

    SOCKET sockfd = socket(AF_INET, SOCK_STREAM, 0); 
    if (sockfd == INVALID_SOCKET) {
	cerr << "could not create socket: " << SOCKET_ERROR_STRING << endl;
	return 0;
    }

    struct sockaddr_in myAddr;
    memset(&myAddr, 0, sizeof(myAddr));
    myAddr.sin_family = AF_INET;
    myAddr.sin_port = htons(port);
    myAddr.sin_addr.s_addr = INADDR_ANY; // auto-fill with my IP

    if (::bind(sockfd, (struct sockaddr *)&myAddr, sizeof(myAddr)) == SOCKET_ERROR) {
	cerr << "could not bind socket: " << SOCKET_ERROR_STRING << endl;
	closesocket(sockfd);
	return 0;
    }

    if (listen(sockfd, maxClients ? maxClients * 10 : 1000) == SOCKET_ERROR) {
	cerr << "could not bind socket: " << SOCKET_ERROR_STRING << endl;
	closesocket(sockfd);
	return 0;
    }

    unsigned numClients = 0;

    while (1) {
	/*
	 * Reap children until number of clients is below max
	 */
#if !defined(_MSC_VER) && !defined(WIN32)
	do {
	    while (waitpid(-1, (int *)NULL, WNOHANG) > 0) {
		numClients -= 1;
	    }

	    if (maxClients > 0 && numClients >= maxClients) {
		sleep(5);
	    }
	} while (maxClients > 0 && numClients >= maxClients);
#endif

	struct sockaddr_in theirAddr; // connector's address information
	socklen_t sin_size = sizeof(theirAddr);
	int client = accept(sockfd, (struct sockaddr *)&theirAddr, &sin_size);
	if (client == SOCKET_ERROR) {
	    cerr << "could not accept connection: " << SOCKET_ERROR_STRING << endl;
	    closesocket(sockfd);
	    return 0;
	}

	/*
	 * With a max of 1 client, we run the server in the main process,
	 * otherwise fork a child for each client.
	 */
	int serverPID;

	if (maxClients == 1) {
	    serverPID = 0;	// simulate child of fork()
	} else {
#if defined(_MSC_VER) || defined(WIN32)
	    cerr << "LM server supports only one client at a time\n";
	    return 0;
#else
	    serverPID = fork();
#endif
	}

	if (serverPID < 0) {
	    cerr << "fork failed: " << srilm_ts_strerror(errno) << endl;
	    closesocket(client);
	    closesocket(sockfd);
	    return 0;
	} else if (serverPID > 0) {
	    /*
	     * Parent of server process --
	     * close client socket and accept next connection
	     */
	    closesocket(client);
	    numClients += 1;
	} else {
	    /*
	     * Forked child process -- act as the server
	     */

	    const char *clientName = inet_ntoa(theirAddr.sin_addr);
	    unsigned clientPort = ntohs(theirAddr.sin_port);

	    cerr << "client " <<  clientPort << "@" << clientName 
		 << ": connection accepted\n";

	    /*
	     * Indicate to LM methods that we're in sequential processing mode.
	     */
	    Boolean wasRunning = running(true);

	    unsigned numProcessed = 0;

	    const char *msg = "probserver ready\n";
	    if (send(client, msg, strlen(msg), 0) == SOCKET_ERROR) {
		cerr << "client " << clientPort << "@" << clientName
		     << ": send: " << SOCKET_ERROR_STRING << endl;
		exit(-1);
	    }

	    char msgBuffer[REMOTELM_MAXREQUESTLEN + 1];
	    int msgLen;
	    unsigned protocolVersion = 1;

	    while ((msgLen = recv(client, msgBuffer, sizeof(msgBuffer)-1, 0)) != SOCKET_ERROR) {
		if (msgLen == 0) break;

		msgBuffer[msgLen] = '\0';
		string response = "";

		char *strtok_ptr = NULL;
		char *line;
		
		/*
		 * Break message into commands, one per line
		 */
		for (line = MStringTokUtil::strtok_r(msgBuffer, "\n", &strtok_ptr);
		     line != 0;
		     line = MStringTokUtil::strtok_r(0, "\n", &strtok_ptr))
		{
		    if (debug(DEBUG_PRINT_WORD_PROBS)) {
			dout() << "client " << clientPort << "@" << clientName 
			       << ": " << line << endl;
		    }

		    VocabString words[maxWordsPerLine + 2];

		    unsigned len =
				vocab.parseWords(line, words, maxWordsPerLine + 2);

		    if (len > 0) {
			char outbuf[REMOTELM_MAXRESULTLEN];

			if (debug(DEBUG_PRINT_WORD_PROBS)) {
			    dout() << "client " << clientPort << "@" << clientName
				   << ": ";
			}

			/*
			 * Decode Remote LM command
			 */
			if (strcmp(words[0], REMOTELM_VERSION2) == 0) {
			    protocolVersion = 2;

			    sprintf(outbuf, "%s\n", REMOTELM_OK);
			} else if (protocolVersion == 1 ||
				   strcmp(words[0], REMOTELM_WORDPROB) == 0)
			{
			    /*
			     * Handle old or new protocol wordProb call
			     */
			    VocabString last = words[len-1];
			    words[len-1] = 0;

			    // reverse N-gram prefix to obtain context
			    Vocab::reverse(protocolVersion > 1 ? words + 1 : words);

			    LogP prob = wordProb(last, protocolVersion > 1 ?
								words + 1 : words);

			    if (protocolVersion == 1) {
				sprintf(outbuf, "%.*g\n", LogP_Precision, prob);
			    } else {
				sprintf(outbuf, "%s %.*g\n", REMOTELM_OK,
							     LogP_Precision, prob);
			    }
			    numProcessed += 1;
			} else if (strcmp(words[0], REMOTELM_CONTEXTID1) == 0) {
			    VocabIndex wids[maxWordsPerLine + 1];

			    vocab.getIndices(words + 1, wids, maxWordsPerLine,
								vocab.unkIndex());

			    // reverse N-gram prefix to obtain context
			    Vocab::reverse(wids);

			    unsigned clen;
			    void *cid = contextID(wids, clen);

			    sprintf(outbuf, "%s %llu %u\n", REMOTELM_OK,
						(long long unsigned)(size_t)cid, clen);
			} else if (strcmp(words[0], REMOTELM_CONTEXTID2) == 0) {
			    VocabIndex wids[maxWordsPerLine + 1];

			    vocab.getIndices(words + 1, wids, maxWordsPerLine,
								vocab.unkIndex());
			    VocabIndex last = wids[len - 1];
			    wids[len - 1] = Vocab_None;

			    // reverse N-gram prefix to obtain context
			    Vocab::reverse(wids);

			    unsigned clen;
			    void *cid = contextID(last, wids, clen);

			    sprintf(outbuf, "%s %llu %u\n", REMOTELM_OK,
						(long long unsigned)(size_t)cid, clen);
			} else if (strcmp(words[0], REMOTELM_CONTEXTBOW) == 0) {
			    unsigned clen;
			    sscanf(words[len - 1], "%u", &clen);
			    words[len - 1] = 0;

			    VocabIndex wids[maxWordsPerLine + 1];
			    vocab.getIndices(words + 1, wids, maxWordsPerLine,
								vocab.unkIndex());

			    // reverse N-gram prefix to obtain context
			    Vocab::reverse(wids);

			    LogP bow = contextBOW(wids, clen);

			    sprintf(outbuf, "%s %.*g\n", REMOTELM_OK,
							 LogP_Precision, bow);
			} else {
			    sprintf(outbuf, "%s command unknown\n", REMOTELM_ERROR);
			}

			/*
			 * Concatenate responses for all commands in the message
			 */
			response += outbuf;
		    }

		}

		if (send(client, response.c_str(), response.length(), 0) == SOCKET_ERROR) {
		    cerr << "client " << clientPort << "@" << clientName
			 << ": send: " << SOCKET_ERROR_STRING << endl;
		    exit(-1);
		}

		if (debug(DEBUG_PRINT_WORD_PROBS)) {
		    dout() << response;
		}
	    }

	    closesocket(client);

	    running(wasRunning);

	    cerr << "client " << clientPort << "@" << clientName 
	         << ": " << numProcessed << " probabilities served\n";

	    if (maxClients != 1) {
		/*
		 * Child server process is done
		 */
		exit(0);
	    }
	}
    }

    return 1;
#else
    cerr << "probServer not supported\n";
    return 0;
#endif 
}

/*
 * Random sample generation
 *
 * generateSentence and generateWord are non-deterministic when used by multiple
 * threads because of the drand call in generateWord. This could be addressed by 
 * having the caller provide a seed or introducing a TLS seed. The former 
 * approach would provide isolation from other drand calls that may be 
 * introduced. 
 */
VocabIndex
LM::generateWord(const VocabIndex *context)
{
    /*
     * Algorithm: generate random number between 0 and 1, and partition
     * the interval 0..1 into pieces corresponding to the word probs.
     * Choose the word whose interval contains the random value.
     */
    VocabIndex wid = Vocab_None;
    unsigned numtries = 0;
    const unsigned generateMaxTries = 10;

    while (wid == Vocab_None && numtries < generateMaxTries) {
	Prob rval = drand48();
	Prob prob = 0, totalProb = 0;
	VocabIter iter(vocab);
	Boolean first = true;

	while (totalProb <= rval && iter.next(wid)) {
	    prob = LogPtoProb(first ? wordProb(wid, context) :
	                              wordProbRecompute(wid, context));
	    totalProb += prob;
	    first = false;
	}

	/*
	 * We've drawn a word that shouldn't have any probability mass.
	 * Issue warning and try again.
	 */
	if (isNonWord(wid)) {
	    if (prob > 0.0 && debug(DEBUG_PRINT_WORD_PROBS)) {
		dout() << "nonword " << vocab.getWord(wid)
		       << " has nonzero probability " << prob << endl;
	    }
	    wid = Vocab_None;
	}
    }

    if (wid == Vocab_None) {
	dout() << "giving up word generation after " << generateMaxTries << endl;
	wid = vocab.seIndex();
    }

    return wid;
}

static TLSW(unsigned, viDefaultResultSize);
static TLSW(VocabIndex*, viDefaultResult);

/*
 * generateSentence and generateWord are non-deterministic when used by multiple
 * threads because of the drand call in generateWord. This could be addressed by 
 * having the caller provide a seed or introducing a TLS seed. The former 
 * approach would provide isolation from other drand calls that may be 
 * introduced. 
 */
VocabIndex *
LM::generateSentence(unsigned maxWords, VocabIndex *sentence,
						VocabIndex *prefix)
{
    /*
     * If no result buffer is supplied use our own.
     */
    unsigned &defaultResultSize = TLSW_GET(viDefaultResultSize);
    VocabIndex* &defaultResult  = TLSW_GET(viDefaultResult);

    if (sentence == 0) {
	if (maxWords + 1 > defaultResultSize) {
	    defaultResultSize = maxWords + 1;
	    if (defaultResult) {
		delete [] defaultResult;
	    }
	    defaultResult = new VocabIndex[defaultResultSize];
	    assert(defaultResult != 0);
	}
	sentence = defaultResult;
    }

    /*
     * Check if a prefix is to be used, and how long it is
     */
    unsigned contextLength;

    if (prefix == 0) {
	if (vocab.ssIndex() != Vocab_None) {
	    contextLength = 1;
	} else {
	    contextLength = 0;
	}
    } else {
	contextLength = Vocab::length(prefix);
    }

    /*
     * Since we need to add the begin/end sentences tokens, and
     * partial contexts are represented in reverse we use a second
     * buffer for partial sentences.
     */
    unsigned last = maxWords + contextLength;

    makeArray(VocabIndex, genBuffer, last + 1);
    genBuffer[last] = Vocab_None;

    if (prefix == 0) {
	if (contextLength == 1) {
	    genBuffer[--last] = vocab.ssIndex();
	}
    } else {
	for (unsigned i = 0; i < contextLength; i ++) {
	    genBuffer[--last] = prefix[i];
	}
    }

    /*
     * Generate words one-by-one until hitting an end-of-sentence.
     */
    while (last > 0 && genBuffer[last] != vocab.seIndex()) {
	last --;
	genBuffer[last] = generateWord(&genBuffer[last + 1]);
    }
    
    /*
     * Copy reversed sentence to output buffer
     */
    unsigned i, j;
    for (i = 0, j = maxWords - 1; j > last; i++, j--) {
	sentence[i] = genBuffer[j];
    }
    sentence[i] = Vocab_None;

    return sentence;
}

static TLSW(unsigned, vsDefaultResultSize);
static TLSW(VocabString*, vsDefaultResult);

/*
 * generateSentence and generateWord are non-deterministic when used by multiple
 * threads because of the drand call in generateWord. This could be addressed by 
 * having the caller provide a seed or introducing a TLS seed. The former 
 * approach would provide isolation from other drand calls that may be 
 * introduced. 
 */
VocabString *
LM::generateSentence(unsigned maxWords, VocabString *sentence,
					VocabString *prefix)
{
    unsigned &defaultResultSize = TLSW_GET(vsDefaultResultSize);
    VocabString* &defaultResult  = TLSW_GET(vsDefaultResult);
    /*
     * If no result buffer is supplied use our own.
     */
    if (sentence == 0) {
	if (maxWords + 1 > defaultResultSize) {
	    defaultResultSize = maxWords + 1;
	    if (defaultResult) {
		delete [] defaultResult;
	    }
	    defaultResult = new VocabString[defaultResultSize];
	    assert(defaultResult != 0);
	}
	sentence = defaultResult;
    }

    VocabIndex *resultIds;

    /*
     * Generate words indices
     */
    if (prefix) {
	unsigned contextLength = Vocab::length(prefix);
        makeArray(VocabIndex, prefixIds, contextLength + 1);

	vocab.getIndices(prefix, prefixIds, contextLength + 1,
							vocab.unkIndex());
	resultIds = generateSentence(maxWords, (VocabIndex *)0, prefixIds);
    } else {
	resultIds = generateSentence(maxWords, (VocabIndex *)0);
    }

    /*
     * Map them to strings
     */
    vocab.getWords(resultIds, sentence, maxWords + 1);
    return sentence;
}

void
LM::freeThread() {
    VocabIndex *vi = TLSW_GET(viDefaultResult);
    VocabString *vs = TLSW_GET(vsDefaultResult);

    delete [] vi;
    delete [] vs;
 
    TLSW_FREE(viDefaultResultSize);
    TLSW_FREE(viDefaultResult);
    TLSW_FREE(vsDefaultResultSize);
    TLSW_FREE(vsDefaultResult);
}

/*
 * Context identification
 *	This returns a unique ID for the portion of a context used in
 *	computing follow-word probabilities. Used for path merging in
 *	lattice search (see the HTK interface).
 *	The length parameter returns the number of words used in context.
 *	The default is to return 0, to indicate all contexts are unique.
 */
void *
LM::contextID(VocabIndex word, const VocabIndex *context, unsigned &length)
{
    length = Vocab::length(context);
    return 0;
}

/*
 * Back-off weight
 *	Computes the backoff weight applied to probabilities that are 
 *	computed from a truncated context.  Used for weight computation in
 *	lattice expansion (see Lattice::expandNodeToLM()).
 */
LogP
LM::contextBOW(const VocabIndex *context, unsigned length)
{
    return LogP_One;
}

/*
 * Global state changes (ignored)
 */
void
LM::setState(const char *state)
{
}

/*
 * LM reading/writing (dummy)
 */
Boolean
LM::read(File &file, Boolean limitVocab)
{
    cerr << "read() method not implemented\n";
    return false;
}

Boolean
LM::write(File &file)
{
    cerr << "write() method not implemented\n";
    return false;
}

Boolean
LM::writeBinary(File &file)
{
    // default is to set invoke write method with binary-write flag set
    Boolean wasBinary = writeInBinary;
    writeInBinary = true;

    Boolean result = write(file);

    // restore binary-write flag
    writeInBinary = wasBinary;
    return result;
}

/*
 * Memory statistics
 */
void
LM::memStats(MemStats &stats)
{
    stats.total += sizeof(*this);
}

/*
 * Iteration over follow words
 *	The generic follow-word iterator enumerates all of vocab.
 */

_LM_FollowIter::_LM_FollowIter(LM &lm, const VocabIndex *context)
    : myLM(lm), myContext(context), myIter(lm.vocab)
{
}

void
_LM_FollowIter::init()
{
    myIter.init();
}

VocabIndex
_LM_FollowIter::next()
{
    VocabIndex index = Vocab_None;
    (void)myIter.next(index);
    return index;
}

VocabIndex
_LM_FollowIter::next(LogP &prob)
{
    VocabIndex index = Vocab_None;
    (void)myIter.next(index);
    
    if (index != Vocab_None) {
	prob = myLM.wordProb(index, myContext);
    }

    return index;
}

