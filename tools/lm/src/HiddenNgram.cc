/*
 * HiddenNgram.cc --
 *	N-gram model with hidden between-word events
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1999-2012 SRI International, 2013 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/HiddenNgram.cc,v 1.31 2014-08-30 02:25:18 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>

#include "option.h"
#include "HiddenNgram.h"
#include "Trellis.cc"
#include "LHash.cc"
#include "SArray.cc"
#include "Array.cc"

#define DEBUG_PRINT_WORD_PROBS          2	/* from LM.cc */
#define DEBUG_NGRAM_HITS		2	/* from Ngram.cc */
#define DEBUG_PRINT_VITERBI		2
#define DEBUG_TRANSITIONS		4

const VocabString noHiddenEvent = "*noevent*";

/* 
 * We use structured of type HiddenNgramState
 * as keys into the trellis.  Define the necessary support functions.
 */

static inline size_t
LHash_hashKey(const HiddenNgramState &key, unsigned maxBits)
{
    return (LHash_hashKey(key.context, maxBits) + key.repeatFrom + key.event)
	   & hashMask(maxBits);
}

static inline int
SArray_compareKey(const HiddenNgramState &key1, const HiddenNgramState &key2)
{
    int c = SArray_compareKey(key1.context, key2.context);

    if (c != 0) {
	return c;
    } else {
	if (key1.event != key2.event) {
	    return key1.event - key2.event;
	} else {
	    return key1.repeatFrom - key2.repeatFrom;
	}
    }
}

static inline HiddenNgramState
Map_copyKey(const HiddenNgramState &key)
{
    HiddenNgramState copy;

    copy.context = Map_copyKey(key.context);
    copy.repeatFrom = key.repeatFrom;
    copy.event = key.event;
    return copy;
}

static inline void
#if __GNUC__ == 2 && __GNUC_MINOR__ <= 7
Map_freeKey(HiddenNgramState key)
#else	// gcc 2.7 doesn't match f(&T) with f(T) here
Map_freeKey(HiddenNgramState &key) // gcc 2.7 has problem matching this
#endif
{
    Map_freeKey(key.context);
}

static inline Boolean
#if __GNUC__ == 2 && __GNUC_MINOR__ <= 7
LHash_equalKey(const HiddenNgramState key1, const HiddenNgramState key2)
#else	// gcc 2.7 doesn't match f(&T,&T) with f(T,T) here
LHash_equalKey(const HiddenNgramState &key1, const HiddenNgramState &key2)
#endif
{
    return LHash_equalKey(key1.context, key2.context) &&
	   key1.repeatFrom == key2.repeatFrom &&
	   key1.event == key2.event;
}

static inline Boolean
Map_noKeyP(const HiddenNgramState &key)
{
    return (key.context == 0);
}

static inline void
Map_noKey(HiddenNgramState &key)
{
    key.context = 0;
}

/*
 * output operator
 */
ostream &
operator<< (ostream &stream, const HiddenNgramState &state)
{
    stream << "<" << state.context << "," << state.repeatFrom << ">";
    return stream;
}

/*
 * LM code 
 */

HiddenNgram::HiddenNgram(Vocab &vocab, SubVocab &hiddenVocab, unsigned order,
							    Boolean notHidden)
    : Ngram(vocab, order),
      trellis(maxWordsPerLine + 2 + 1), savedLength(0),
      hiddenVocab(hiddenVocab), notHidden(notHidden)
{
    /*
     * Make sure the hidden events are subset of base vocabulary 
     */
    assert(&vocab == &hiddenVocab.baseVocab());

    /*
     * Make sure noevent token is not used in LM, and add to the event
     * vocabulary.
     */
    assert(hiddenVocab.getIndex(noHiddenEvent) == Vocab_None);
    noEventIndex = hiddenVocab.addWord(noHiddenEvent);

    /*
     * Give default vocabulary props to all hidden events.
     * This is needed because we later iterate over all vocab props and
     * need to catch all event tags that way.
     */
    VocabIter viter(hiddenVocab);
    VocabIndex event;

    while (viter.next(event)) {
	Boolean foundP;
	HiddenVocabProps *props = vocabProps.insert(event, foundP);
	if (!foundP) {
	    props->insertWord = Vocab_None;
	}
    }

    /*
     * The "no-event" event is excluded from the context
     */
    vocabProps.insert(noEventIndex)->omitFromContext = true;
}

HiddenNgram::~HiddenNgram()
{
}

void *
HiddenNgram::contextID(VocabIndex word, const VocabIndex *context,
							    unsigned &length)
{
    /*
     * Due to the DP algorithm, we always use the full context
     * (don't inherit Ngram::contextID()).
     */
    return LM::contextID(word, context, length);
}

LogP
HiddenNgram::contextBOW(const VocabIndex *context, unsigned length)
{
    /*
     * Due to the DP algorithm, we always use the full context
     * (don't inherit Ngram::contextBOW()).
     */
    return LM::contextBOW(context, length);
}

/*
 * Return the vocab properties for a word
 * (return defaults if none are defined)
 */
static const HiddenVocabProps defaultProps = 
     { 0, 0, false, false, Vocab_None };

const HiddenVocabProps &
HiddenNgram::getProps(VocabIndex word)
{
    HiddenVocabProps *props = vocabProps.find(word);
    if (props) {
	return *props;
    } else {
	return defaultProps;
    }
}

Boolean
HiddenNgram::isNonWord(VocabIndex word)
{
    if (LM::isNonWord(word)) {
	return true;
    } else {
	HiddenVocabProps *props = vocabProps.find(word);

	if (!props) {
	    return false;
	} else {
	    return !props->isObserved;
	}
    }
}

Boolean
HiddenNgram::read(File &file, Boolean limitVocab)
{
    /*
     * First, read the regular N-gram model
     */
    if (!Ngram::read(file, limitVocab)) {
	return false;
    } 

    /*
     * Read vocab property specs from the LM file
     */
    if (!readHiddenVocab(file)) {
	return false;
    }

    return true;
}

Boolean
HiddenNgram::readHiddenVocab(File &file)
{
    char *line;

    while ((line = file.getline())) {
	VocabString argv[maxWordsPerLine + 1];

	unsigned argc = Vocab::parseWords(line, argv, maxWordsPerLine + 1);
	if (argc > maxWordsPerLine) {
	    file.position() << "too many words in line\n";
	    return false;
	}

	if (argc == 0) {
	    continue;
	}

	/*
	 * First word on line defines the vocabulary item
	 */
	VocabIndex word = hiddenVocab.addWord(argv[0]);

	unsigned deleteWords = 0;
	unsigned repeatWords = 0;
	char *insertWord = 0;
	int isObserved = 0;;
	int omitFromContext = 0;
	Option options[] = {
#	    define PROP_DELETE "delete"
		{ OPT_UINT, PROP_DELETE, &deleteWords, "words to delete" },
#	    define PROP_REPEAT "repeat"
		{ OPT_UINT, PROP_REPEAT, &repeatWords, "words to repeat" },
#	    define PROP_INSERT "insert"
		{ OPT_STRING, PROP_INSERT, &insertWord, "insert word" },
#	    define PROP_OBSERVED "observed"
		{ OPT_TRUE, PROP_OBSERVED, &isObserved, "observed event" },
#	    define PROP_OMIT "omit"
		{ OPT_TRUE, PROP_OMIT, &omitFromContext, "omit from context" },
	};

	if (Opt_Parse(argc, (char **)argv, options, Opt_Number(options),
						    OPT_UNKNOWN_IS_ERROR) != 1)
	{
	    file.position() << "allowed vocabulary properties for word " << argv[0] << " are\n";
	    Opt_PrintUsage(NULL, options, Opt_Number(options));
	    return false;
	}

	HiddenVocabProps *props = vocabProps.insert(word);

	props->deleteWords = deleteWords;
	props->repeatWords = repeatWords;
	props->isObserved = isObserved;
	props->omitFromContext = omitFromContext;
	if (insertWord) {
	    props->insertWord = vocab.addWord(insertWord);
	    props->isObserved = false;		/* implied */
	} else {
	    props->insertWord = Vocab_None;
	}

	/* 
	 * Unless event is observable, add it to hidden vocabulary
	 */
	if (!isObserved) {
	    hiddenVocab.addWord(word);
	}
    }

    return true;
}

Boolean
HiddenNgram::write(File &file)
{
    if (!Ngram::write(file)) {
	return false;
    }

    file.fprintf("\n");

    if (!writeHiddenVocab(file)) {
	return false;
    }

    return true;
}

Boolean
HiddenNgram::writeHiddenVocab(File &file)
{
    LHashIter<VocabIndex, HiddenVocabProps> iter(vocabProps);

    HiddenVocabProps *props;
    VocabIndex word;

    while ((props = iter.next(word))) {
	file.fprintf("%s", vocab.getWord(word));

	if (props->deleteWords) {
	    file.fprintf(" -%s %u", PROP_DELETE, props->deleteWords);
	}
	if (props->repeatWords) {
	    file.fprintf(" -%s %u", PROP_REPEAT, props->repeatWords);
	}
	if (props->insertWord != Vocab_None) {
	    file.fprintf(" -%s %s", PROP_INSERT,
					vocab.getWord(props->insertWord));
	}
	if (props->isObserved) {
	    file.fprintf(" -%s", PROP_OBSERVED);
	}
	if (props->omitFromContext) {
	    file.fprintf(" -%s", PROP_OMIT);
	}
	file.fprintf("\n");
    }

    return true;
}

/*
 * Compute the prefix probability of word string (in reversed order)
 * taking hidden events into account.
 * This is done by dynamic programming, filling in the trellis[]
 * array from left to right.
 * Entry trellis[i][state].prob is the probability that of the first
 * i words while being in state.
 * The total prefix probability is the column sum in trellis[],
 * summing over all states.
 * For efficiency reasons, we specify the last word separately.
 * If context == 0, reuse the results from the previous call.
 */
LogP
HiddenNgram::prefixProb(VocabIndex word, const VocabIndex *context,
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
	 * no-event state.
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
	    HiddenNgramState initialState;
	    initialState.context = initialContext;
	    initialState.repeatFrom = 0;
	    initialState.event = noEventIndex;

	    trellis.clear();
	    trellis.setProb(initialState, LogP_One);
	    trellis.step();

	    savedContext[0] = initialContext[0];
	    savedLength = 1;
	    pos = 1;
	}
    }

    LogP logSum = LogP_Zero;

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

	const HiddenVocabProps &currWordProps = getProps(currWord);

	const VocabIndex *currContext = &context[prefix];

        /*
         * Set up new context to transition to
         * (allow enough room for one hidden event per word)
         */
        const unsigned newContextSize = 2 * maxWordsPerLine + 2;
        VocabIndex newContext[newContextSize];

	/*
	 * Iterate over all contexts for the previous position in trellis
	 */
	TrellisIter<HiddenNgramState> prevIter(trellis, pos - 1);

	HiddenNgramState prevState;
	LogP prevProb;

	while (prevIter.next(prevState, prevProb)) {
	    VocabContext prevContext = prevState.context;

            /*
             * Set up the extended context.  Allow room for adding 
             * the current word and a new hidden event.
             */
            unsigned i;
            for (i = 0; i < 2 * maxWordsPerLine && 
                                    prevContext[i] != Vocab_None; i ++)
            {
                newContext[i + 2] = prevContext[i];
            }

            if ((i + 2) < newContextSize) {
                newContext[i + 2] = Vocab_None;
            }

	    unsigned prevContextLength = i;

            /*
             * Iterate over all hidden events
             */
	    LHashIter<VocabIndex, HiddenVocabProps> eventIter(vocabProps);
            VocabIndex currEvent;
	    HiddenVocabProps *currEventProps;

            while ((currEventProps = eventIter.next(currEvent))) {
		/*
		 * Observable events are dealt with as regular words in 
		 * the input string
		 */
		if (currEventProps->isObserved) {
		    continue;
		}

		/* 
		 * While repeating words all other events are disallowed
		 */
		if (prevState.repeatFrom > 0 && currEvent != noEventIndex) {
		    continue;
		}

                /*
                 * Prepend current event and word to context
                 */ 
		VocabIndex *startNewContext = &newContext[2];

		/*
		 * Add event to context unless it's omissible
		 */
		if (!(currEventProps->omitFromContext)) {
		    startNewContext --;
		    startNewContext[0] = currEvent;
		}

		VocabIndex *startWordContext = startNewContext;

                LogP eventProb;
		LogP wordProb;
		unsigned repeatFrom;
		VocabIndex savedContextWord = Vocab_None;

		/*
		 * Check if we're repeating words, either by virtue of 
		 * the current event or a pending repeat
		 */
		if (prevState.repeatFrom > 0) {
		    repeatFrom = prevState.repeatFrom;
		} else if (currEventProps->repeatWords > 0) {
		    repeatFrom = currEventProps->repeatWords;
		} else {
		    repeatFrom = 0;
		}

		/*
		 * Manipulate context for current word for special 
		 * disfluency-type events
		 */
		if (repeatFrom > 0) {
		    /*
		     * If repeated word doesn't match current word 
		     * we can skip extending this state.
		     * Note: we don't allow repeats to apply to <unk>!
		     */
		    if (repeatFrom > prevContextLength ||
			currWord != prevContext[repeatFrom - 1] ||
			currWord == vocab.unkIndex())
		    {
			continue;
		    }

		    /*
		     * If we're extending a previous repeat event there is
		     * no further charge
		     * Otherwise, use the prob of the repeat event itself.
		     */
		    if (prevState.repeatFrom > 0) {
			eventProb = LogP_One;
		    } else {
			eventProb = Ngram::wordProb(currEvent, prevContext);
		    }

		    /*
		     * There is never a charge for the repeated word
		     */
		    wordProb = LogP_One;

		    repeatFrom --;
		} else {
		    if (currEventProps->insertWord != Vocab_None) {
			/* 
			 * We cannot leave both the event itself and 
			 * and inserted tag in the context.  Overwrite the
			 * former if necessary.
			 */
			if (currEventProps->omitFromContext) {
			    startNewContext --;
			}

			/*
			 * Insert designated token
			 */
			startNewContext[0] = currEventProps->insertWord;
		    }

		    /*
		     * Delete specified number of words from context
		     */
		    unsigned i;
		    for (i = 0; i < currEventProps->deleteWords; i ++) {
			if (startNewContext[0] == Vocab_None ||
			    startNewContext[0] == vocab.ssIndex()) 
			{
			    break;
			}
			startNewContext ++;
		    }

		    /*
		     * Eliminate deletion events that would go past the 
		     * start of the sentence
		     */
		    if (i < currEventProps->deleteWords) {
			continue;
		    }

		    /*
		     * Add current word to new context unless it's omissible
		     * Since the position we're storing the new word in may
		     * actually be part of the old context we need to save the
		     * old contents at that position so we can restore it later
		     * (for the next run through this loop).
		     */
		    if (!(currWordProps.omitFromContext)) {
			startNewContext --;
			savedContextWord = startNewContext[0];
			startNewContext[0] = currWord;
		    }

		    /*
		     * Event probability
		     */
		    eventProb = (currEvent == noEventIndex) ? LogP_One
				      : Ngram::wordProb(currEvent, prevContext);

		    /*
		     * Transition prob out of previous context to current word.
		     * Put underlying Ngram in "running" state (for debugging)
		     * only when processing a "no-event" context.
		     */
		    if (prefix == 0 && prevState.event == noEventIndex &&
			currEvent == noEventIndex)
		    {
			running(wasRunning);
		    }
		    wordProb = Ngram::wordProb(currWord, startWordContext);

		    if (prefix == 0 && prevState.event == noEventIndex &&
			currEvent == noEventIndex)
		    {
			running(false);
		    }

		    if (prevProb != LogP_Zero && wordProb != LogP_Zero) {
			havePosProb = true;
		    }
		}

                /*
                 * Truncate context to what is actually used by LM,
                 * but keep at least one word so we can recover words later.
		 * When inside a repeat make sure we retain the words to be
		 * repeated.
                 */
                unsigned usedLength;
                Ngram::contextID(Vocab_None, startNewContext, usedLength);
		if (usedLength < repeatFrom) {
		    assert(repeatFrom < prevContextLength);
		    usedLength = repeatFrom;
		} else if (usedLength == 0) {
                    usedLength = 1;
                }

                VocabIndex truncatedContextWord = startNewContext[usedLength];
                startNewContext[usedLength] = Vocab_None;
	   
		HiddenNgramState newState;
		newState.context = startNewContext;
		newState.repeatFrom = repeatFrom;
		newState.event = currEvent;

                if (debug(DEBUG_TRANSITIONS)) {
                    cerr << "POSITION = " << pos
                         << " FROM: " << (vocab.use(), prevState)
                         << " TO: " << (vocab.use(), newState)
                         << " WORD = " << vocab.getWord(currWord)
                         << " EVENT = " << vocab.getWord(currEvent)
                         << " EVENTPROB = " << eventProb
                         << " WORDPROB = " << wordProb
                         << endl;
                }

		/*
		 * For efficiency reasons we don't update the trellis
		 * when at the final word.  In that case we just record
		 * the total probability.
		 */
		if (prefix > 0 || debug(DEBUG_PRINT_VITERBI)) {
		    trellis.update(prevState, newState, eventProb + wordProb);
		}

		if (prefix == 0) {
		    logSum = AddLogP(logSum, prevProb + eventProb + wordProb);
		}

                /*
                 * Restore newContext
                 */
		if (savedContextWord != Vocab_None) {
		    startNewContext[0] = savedContextWord;
		}
                startNewContext[usedLength] = truncatedContextWord;
            }
	}

	/*
	 * Set noevent state probability to the previous total prefix
	 * probability if the current word had probability zero in all
	 * states, and we are not yet at the end of the prefix.
	 * This allows us to compute conditional probs based on
	 * truncated contexts, and to compute the total sentence probability
	 * leaving out the OOVs, as required by sentenceProb().
	 */
	if (prefix > 0 && !havePosProb) {
	    VocabIndex emptyContext[3];
	    emptyContext[0] = noEventIndex;
	    emptyContext[1] = currWord;
	    emptyContext[2] = Vocab_None;
	    HiddenNgramState emptyState;
	    emptyState.context = emptyContext;
	    emptyState.repeatFrom = 0;
	    emptyState.event = noEventIndex;

	    trellis.init(pos);
	    trellis.setProb(emptyState, trellis.sumLogP(pos - 1));

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
	contextProb = trellis.sumLogP(prevPos - 1);
    } else { 
	contextProb = LogP_One;
    }
    return logSum;
}

/*
 * The conditional word probability is computed as
 *	p(w1 .... wk)/p(w1 ... w(k-1)
 */
LogP
HiddenNgram::wordProb(VocabIndex word, const VocabIndex *context)
{
    if (notHidden) {
	/*
	 * In "nothidden" we assume that we are processing a token stream
	 * that contains overt event tokens.  Hence we give event tokens
	 * probability zero (so they don't contribute to the perplexity),
	 * and we scale non-event token probabilities by
	 *		1/(1-sum of all event probs)
	 */
	if (hiddenVocab.getWord(word) != 0) {
	    if (running() && debug(DEBUG_NGRAM_HITS)) {
		dout() << "[event]";
	    }
	    return LogP_Zero;
	} else {
	    LogP totalWordProb = LogP_One;

	    VocabIter eventIter(hiddenVocab);
	    VocabIndex event;

	    Boolean wasRunning = running(false);
	    while (eventIter.next(event)) {
		totalWordProb = SubLogP(totalWordProb,
					    Ngram::wordProb(event, context));
	    }
	    running(wasRunning);

	    return Ngram::wordProb(word, context) - totalWordProb;
	}
    } else {
	/*
	 * Standard hidden event mode: do that dynamic programming thing...
	 */
	LogP cProb;
	TextStats stats;
	LogP pProb = prefixProb(word, context, cProb, stats);
	return pProb - cProb;
    }
}

LogP
HiddenNgram::wordProbRecompute(VocabIndex word, const VocabIndex *context)
{
    if (notHidden) {
	return wordProb(word, context);
    } else {
	LogP cProb;
	TextStats stats;
	LogP pProb = prefixProb(word, 0, cProb, stats);
	return pProb - cProb;
    }
}

/*
 * Sentence probabilities from indices
 *	This version computes the result directly using prefixProb to
 *	avoid recomputing prefix probs for each prefix.
 */
LogP
HiddenNgram::sentenceProb(const VocabIndex *sentence, TextStats &stats)
{
    unsigned int len = vocab.length(sentence);
    LogP totalProb;

    /*
     * The debugging machinery is not duplicated here, so just fall back
     * on the general code for that.
     */
    if (notHidden || debug(DEBUG_PRINT_WORD_PROBS)) {
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
	totalProb = prefixProb(reversed[0], reversed + 1, contextProb, stats);

	/* 
	 * OOVs and zeroProbs are updated by prefixProb()
	 */
	stats.numSentences ++;
	stats.prob += totalProb;
	stats.numWords += len;
    }

    if (!notHidden && debug(DEBUG_PRINT_VITERBI)) {
	len = trellis.where();
	makeArray(HiddenNgramState, bestStates, len);

	if (trellis.viterbi(bestStates, len) == 0) {
	    dout() << "Viterbi backtrace failed\n";
	} else {
	    dout() << "Hidden events:";

	    for (unsigned i = 1; i < len; i ++) {
		dout() << " " << vocab.getWord(bestStates[i].event);
	    }

	    dout() << endl;
	}
    }

    return totalProb;
}

