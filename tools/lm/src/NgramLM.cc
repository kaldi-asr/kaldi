/*
 * NgramLM.cc --
 *	N-gram backoff language models
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2011 SRI International, 2012-2015 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramLM.cc,v 1.146 2016/06/19 04:36:59 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <new.h>
# include <iostream.h>
#else
# include <new>
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#if !defined(_MSC_VER) && !defined(WIN32)
#include <sys/param.h>
#endif

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#include "tserror.h"
#include "Ngram.h"
#include "BayesMix.h"
#include "File.h"
#include "Array.cc"
#include "SArray.h"	// for SArray_compareKey()

#ifdef USE_SARRAY

#include "SArray.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_SARRAY(VocabIndex,LogP); 
#endif

#else /* ! USE_SARRAY */

#include "LHash.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_LHASH(VocabIndex,LogP); 
#endif

#endif /* USE_SARRAY */

#include "Trie.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_TRIE(VocabIndex,BOnode);
#endif

/*
 * Debug levels used
 */
#define DEBUG_ESTIMATE_WARNINGS	1
#define DEBUG_FIXUP_WARNINGS 3
#define DEBUG_PRINT_GTPARAMS 2
#define DEBUG_READ_STATS 1
#define DEBUG_WRITE_STATS 1
#define DEBUG_NGRAM_HITS 2
#define DEBUG_ESTIMATES 4

/* these are the same as in LM.cc */
#define DEBUG_PRINT_SENT_PROBS		1
#define DEBUG_PRINT_WORD_PROBS		2
#define DEBUG_PRINT_PROB_SUMS		3

const LogP LogP_PseudoZero = -99.0;	/* non-inf value used for log 0 */

const char NgramBinaryIOWildcard = '*';

/*
 * Constant for binary format
 */
const char *Ngram_BinaryV1FormatString = "SRILM_BINARY_NGRAM_001\n";
const char *Ngram_BinaryFormatString = "SRILM_BINARY_NGRAM_002\n";
const long long Ngram_BinaryMagicLonglong = 0x0123456789abcdefLL;
const LogP Ngram_BinaryMagicFloat = (LogP)9.8765432e10;

/*
 * Low level methods to access context (BOW) nodes and probs
 */

void
Ngram::memStats(MemStats &stats)
{
    stats.total += sizeof(*this) - sizeof(contexts);
    contexts.memStats(stats);

    /*
     * The probs tables are not included in the above call,
     * so we have to count those separately.
     */
    makeArray(VocabIndex, context, order + 1);

    for (unsigned i = 1; i <= order; i++) {
	NgramBOsIter iter(*this, context, i - 1);
	BOnode *node;

	while ((node = iter.next())) {
	    stats.total -= sizeof(node->probs);
	    node->probs.memStats(stats);
	}
    }
}

Ngram::Ngram(Vocab &vocab, unsigned neworder)
    : LM(vocab), contexts(vocab.numWords()),
      order(neworder), _skipOOVs(false), _trustTotals(false),
      codebook(0)
{
    if (order < 1) {
	order = 1;
    }
}

unsigned 
Ngram::setorder(unsigned neworder)
{
    unsigned oldorder = order;

    if (neworder > 0) {
	order = neworder;
    }

    return oldorder;
}

/*
 * Locate a BOW entry in the n-gram trie
 */
LogP *
Ngram::findBOW(const VocabIndex *context) const
{
    BOnode *bonode = contexts.find(context);
    if (bonode) {
	return &(bonode->bow);
    } else {
	return 0;
    }
}

/*
 * Locate a prob entry in the n-gram trie
 */
LogP *
Ngram::findProb(VocabIndex word, const VocabIndex *context) const
{
    BOnode *bonode = contexts.find(context);
    if (bonode) {
	return bonode->probs.find(word);
    } else {
	return 0;
    }
}

/*
 * Locate or create a BOW entry in the n-gram trie
 */
LogP *
Ngram::insertBOW(const VocabIndex *context)
{
    Boolean found;
    BOnode *bonode = contexts.insert(context, found);

    if (!found) {
	/*
	 * initialize the index in the BOnode
	 */
	new (&bonode->probs) PROB_INDEX_T<VocabIndex,LogP>(0);
    }
    return &(bonode->bow);
}

/*
 * Locate or create a prob entry in the n-gram trie
 */
LogP *
Ngram::insertProb(VocabIndex word, const VocabIndex *context)
{
    Boolean found;
    BOnode *bonode = contexts.insert(context, found);

    if (!found) {
	/*
	 * initialize the index in the BOnode
	 * We know we need at least one entry!
	 */
	new (&bonode->probs) PROB_INDEX_T<VocabIndex,LogP>(1);
    }
    return bonode->probs.insert(word);
}

/*
 * Remove a BOW node (context) from the n-gram trie
 */
void
Ngram::removeBOW(const VocabIndex *context)
{
    contexts.removeTrie(context);
}

/*
 * Remove a prob entry from the n-gram trie
 */
void
Ngram::removeProb(VocabIndex word, const VocabIndex *context)
{
    BOnode *bonode = contexts.find(context);

    if (bonode) {
	bonode->probs.remove(word);
    }
}

/*
 * Remove all probabilities and contexts from n-gram trie
 */
void
Ngram::clear()
{
    makeArray(VocabIndex, context, order);

    BOnode *node;

    /*
     * Remove all ngram probabilities
     */
    for (unsigned i = order; i > 0; i--) {
	BOnode *node;
	NgramBOsIter iter(*this, context, i - 1);
	
	while ((node = iter.next())) {
	    node->probs.clear(0);
	}
    }
    
    /*
     * Remove all unigram context tries
     * (since it won't work to delete at the root)
     * This also has the advantage of preserving the allocated space in
     * the root node, chosen to match the vocabulary.
     */
    if (order > 1) {
	NgramBOsIter iter(*this, context, 1);

	while ((node = iter.next())) {
	    removeBOW(context);
	}
    }
}

/*
 * Higher level methods, implemented low-level for efficiency
 */

/*
 * Find the longest BO node that matches a prefix of context.
 * If word != Vocab_None, return longest context defining prob for word.
 * Returns its address as the ID.
 */
void *
Ngram::contextID(VocabIndex word, const VocabIndex *context, unsigned &length)
{
    BOtrie *trieNode = &contexts;

    unsigned i = length = 0;
    BOtrie *next = trieNode;
    while (i < order - 1 && context[i] != Vocab_None) {
	next = next->findTrie(context[i]);
	if (next != 0) {
	    i ++;
	    if (word == Vocab_None || next->value().probs.find(word)) {
	        trieNode = next;
		length = i;
	    }
	} else {
	    break;
	}
    }

    return (void *)trieNode;
}

/*
 * Compute backoff weight corresponding to truncating context at given length
 */
LogP
Ngram::contextBOW(const VocabIndex *context, unsigned length)
{
    BOtrie *trieNode = &contexts;
    LogP bow = LogP_One;

    unsigned i = 0;
    while (i < order - 1 && context[i] != Vocab_None) {
	BOtrie *next = trieNode->findTrie(context[i]);
	if (next) {
	    trieNode = next;
	    i ++;

	    if (i > length) {
		bow += next->value().bow;
	    }
	} else {
	    break;
	}
    }

    return bow;
}

/*
 * This method implements the backoff algorithm in a way that descends into
 * the context trie only once, finding the maximal ngram and accumulating 
 * backoff weights along the way.
 */
LogP
Ngram::wordProbBO(VocabIndex word, const VocabIndex *context, unsigned int clen)
{
    LogP logp = LogP_Zero;
    LogP bow = LogP_One;
    unsigned found = 0;

    BOtrie *trieNode = &contexts;
    unsigned i = 0;

    do {
	LogP *prob = trieNode->value().probs.find(word);

	if (prob) {
	    /*
	     * If a probability is found at this level record it as the 
	     * most specific one found so far and reset the backoff weight.
	     */
	    logp = *prob;
	    bow = LogP_One;
	    found = i + 1;
	} 

        if  (i >= clen || context[i] == Vocab_None) break;

	BOtrie *next = trieNode->findTrie(context[i]);
	if (next) {
	    /*
	     * Accumulate backoff weights 
	     */
	    bow += next->value().bow;
	    trieNode = next;
	    i ++;
	} else {
	    break;
	}
    } while (1);

    if (running() && debug(DEBUG_NGRAM_HITS)) {
	if (found) {
	    dout() << "[" << found << "gram]";
	} else {
	    dout() << "[OOV]";
	}
    }

    return logp + bow;
}

/*
 * Higher level methods (using the lower-level methods)
 */

LogP
Ngram::wordProb(VocabIndex word, const VocabIndex *context)
{
    unsigned int clen = Vocab::length(context);

    if (skipOOVs()) {
	/*
	 * Backward compatibility with the old broken perplexity code:
	 * return prob 0 if any of the context-words have an unknown
	 * word.
	 */
	if (word == vocab.unkIndex() ||
	    (order > 1 && context[0] == vocab.unkIndex()) ||
	    (order > 2 && clen > 0 && context[1] == vocab.unkIndex()))
	{
	    return LogP_Zero;
	}
    }

    /*
     * Perform the backoff algorithm for a context lenth that is 
     * the minimum of what we were given and the maximal length of
     * the contexts stored in the LM
     */
    return wordProbBO(word, context, ((clen > order - 1) ? order - 1 : clen));
}

Boolean
Ngram::read(File &file, Boolean limitVocab)
{
    char *line;
    unsigned maxOrder = 0;	/* maximal n-gram order in this model */
    Count numNgrams[maxNgramOrder + 1];
				/* the number of n-grams for each order */
    Count numRead[maxNgramOrder + 1];
				/* Number of n-grams actually read */
    Count numOOVs = 0;	/* Numer of n-gram skipped due to OOVs */
    int state = -1 ;		/* section of file being read:
				 * -1 - pre-header, 0 - header,
				 * 1 - unigrams, 2 - bigrams, ... */
    Boolean warnedAboutUnk = false; /* at most one warning about <unk> */

    for (unsigned i = 0; i <= maxNgramOrder; i++) {
	numNgrams[i] = 0;
	numRead[i] = 0;
    }

    clear();

    /*
     * The ARPA format implicitly assumes a zero-gram backoff weight of 0.
     * This has to be properly represented in the BOW trie so various
     * recursive operations work correctly.
     */
    VocabIndex nullContext[1];
    nullContext[0] = Vocab_None;
    *insertBOW(nullContext) = LogP_Zero;

    long long startOffset = file.ftell();

    while ((line = file.getline())) {
	
	Boolean backslash = (line[0] == '\\');

	switch (state) {

	case -1: 	/* looking for start of header */

            if (strcmp(line, Ngram_BinaryFormatString) == 0) {
		// reopen file in binary mode
		File binaryFile(file.name, "rb");

		// start reading binary file from same offset as before
		if (startOffset != -1) {
		    binaryFile.fseek(startOffset, SEEK_SET);
		}

		if (debug(DEBUG_READ_STATS)) {
		    dout() << "reading " << file.name << " in binary format\n";
		}

		Boolean success = readBinary(binaryFile, limitVocab);

		// move the original file stream to after the LM just read
		// so calling functions can continue reading as expected.
		if (success) {
		    long long endOffset = binaryFile.ftell();
		    if (endOffset != -1) {
			file.fseek(endOffset, SEEK_SET);
		    }
		}
		return success;
	    } else if (strcmp(line, Ngram_BinaryV1FormatString) == 0) {
		if (debug(DEBUG_READ_STATS)) {
		    dout() << "reading " << file.name
		    	   << " in old binary format\n";
		}

		return readBinaryV1(file, limitVocab);
            }
            
	    if (backslash && strncmp(line, "\\data\\", 6) == 0) {
		state = 0;
		continue;
	    }
	    /*
	     * Everything before "\\data\\" is ignored
	     */
	    continue;

	case 0:		/* ngram header */
	    unsigned thisOrder;
	    long long nNgrams;

	    if (backslash && sscanf(line, "\\%d-grams", &state) == 1) {
		/*
		 * start reading n-grams
		 */
		if (state < 1 || (unsigned)state > maxOrder) {
		    file.position() << "invalid ngram order " << state << "\n";
		    return false;
		}

	        if (debug(DEBUG_READ_STATS)) {
		    dout() << ((unsigned)state <= order ? "reading " : "skipping ")
			   << numNgrams[state] << " "
			   << state << "-grams\n";
		}

		continue;
	    } else if (sscanf(line, "ngram %u=%lld", &thisOrder, &nNgrams) == 2) {
		/*
		 * scanned a line of the form
		 *	ngram <N>=<howmany>
		 * now perform various sanity checks
		 */
		if (thisOrder <= 0 || thisOrder > maxNgramOrder) {
		    file.position() << "ngram order " << thisOrder
				    << " out of range\n";
		    return false;
		}
		if (nNgrams < 0) {
		    file.position() << "ngram number " << nNgrams
				    << " out of range\n";
		    return false;
		}
		if (thisOrder > maxOrder) {
		    maxOrder = thisOrder;
		}
		numNgrams[thisOrder] = nNgrams;
		continue;
	    } else {
		file.position() << "unexpected input\n";
		return false;
	    }

	default:	/* reading n-grams, where n == state */

	    if (backslash) {
		if (numOOVs > 0) {
		    if (debug(DEBUG_READ_STATS)) {
			dout() << "discarded " << numOOVs
			       << " OOV " << state << "-grams\n";
		    }
		    numOOVs = 0;
		}
	    }

	    if (backslash && sscanf(line, "\\%d-grams", &state) == 1) {
		if (state < 1 || (unsigned)state > maxOrder) {
		    file.position() << "invalid ngram order " << state << "\n";
		    return false;
		}

	        if (debug(DEBUG_READ_STATS)) {
		    dout() << ((unsigned)state <= order ? "reading " : "skipping ")
			   << numNgrams[state] << " "
			   << state << "-grams\n";
		}

		/*
		 * start reading more n-grams
		 */
		continue;
	    } else if (backslash && strncmp(line, "\\end\\", 5) == 0) {
		/*
		 * Check that the total number of ngrams read matches
		 * that found in the header
		 */
		for (unsigned i = 0; i <= maxOrder && i <= order; i++) {
		    if (numNgrams[i] != numRead[i]) {
			file.position() << "warning: " << numRead[i] << " "
			                << i << "-grams read, expected "
			                << numNgrams[i] << "\n";
		    }
		}

		return true;
	    } else if ((unsigned)state > order) {
		/*
		 * Save time and memory by skipping ngrams outside
		 * the order range of this model
		 */
		continue;
	    } else {
		VocabString words[1+ maxNgramOrder + 1 + 1];
				/* result of parsing an n-gram line
				 * the first and last elements are actually
				 * numerical parameters, but so what? */
		VocabIndex wids[maxNgramOrder + 1];
				/* ngram translated to word indices */
		LogP prob, bow = LogP_One;
				/* probability and back-off-weight */

		/*
		 * Parse a line of the form
		 *	<prob>	<w1> <w2> ... [ <bow> ]
		 */
		unsigned howmany = Vocab::parseWords(line, words, state + 3);

		if (howmany < (unsigned)state + 1 || howmany > (unsigned)state + 2) {
		    file.position() << "ngram line has " << howmany
				    << " fields (" << state + 2
				    << " expected)\n";
		    return false;
		}

		if (codebook != 0) {
		    /*
		     * Parse codebook index and map to prob
		     */
		    unsigned bin;
		    if (sscanf(words[0], "%u", &bin) != 1 || !codebook->valid(bin)) {
			file.position() << "invalid codebook index \"" << words[0] << "\"\n";
		        return false;
		    }
		    prob = codebook->getProb(bin);
		} else {
		    /*
		     * Parse prob
		     */
		    if (!parseLogP(words[0], prob)) {
			file.position() << "bad prob \"" << words[0] << "\"\n";
			return false;
		    }
		}

		if (prob > LogP_One || prob != prob) {
		    file.position() << "warning: questionable prob \""
				    << words[0] << "\"\n";
		} else if (prob == LogP_PseudoZero) {
		    /*
		     * convert pseudo-zeros back into real zeros
		     */
		    prob = LogP_Zero;
		}

		/* 
		 * Parse bow, if any
		 */
		if (howmany == (unsigned)state + 2) {
		    /*
		     * Parsing floats strings is the most time-consuming
		     * part of reading in backoff models.  We therefore
		     * try to avoid parsing bows where they are useless,
		     * i.e., for contexts that are longer than what this
		     * model uses.  We also do a quick sanity check to
		     * warn about non-zero bows in that position.
		     */
		    if ((unsigned)state == maxOrder) {
			if (words[state + 1][0] != '0') {
			    file.position() << "ignoring non-zero bow \""
					    << words[state + 1]
					    << "\" for maximal ngram\n";
			}
		    } else if ((unsigned)state == order) {
			/*
			 * save time and memory by skipping bows that will
			 * never be used as a result of higher-order ngram
			 * skipping
			 */
			;
		    } else {
			if (codebook != 0) {
			    /*
			     * Parse codebook index and map to BOW
			     * XXX: probs and BOWs share the same codebook
			     */
			    unsigned bin;
			    if (sscanf(words[state + 1], "%u", &bin) != 1 || !codebook->valid(bin)) {
				file.position() << "invalid codebook index \"" << words[state + 1] << "\"\n";
				return false;
			    }
			    bow = codebook->getProb(bin);
			} else {
			    /* 
			     * Parse BOW
			     */
			    if (!parseLogP(words[state + 1], bow)) {
				file.position() << "bad bow \"" << words[state + 1]
						<< "\"\n";
				return false;
			    }
			}

			if (bow == LogP_Inf || bow != bow) {
			    file.position() << "warning: questionable bow \""
					    << words[state + 1] << "\"\n";
			} else if (bow == LogP_PseudoZero) {
			    /*
			     * convert pseudo-zeros back into real zeros
			     */
			    bow = LogP_Zero;
			}
		    }
		}

		numRead[state] ++;

		/* 
		 * Terminate the words array after the last word,
		 * then translate it to word indices.  We also
		 * reverse the ngram since that's how we'll need it
		 * to index the trie.
		 */
		words[state + 1] = 0;
		if (limitVocab) {
		    /*
		     * Skip N-gram if it contains OOV word.
		     * Do NOT add new words to the vocabulary.
		     */
		    if (!vocab.checkWords(&words[1], wids, maxNgramOrder)) {
			numOOVs ++;
			/* skip rest of processing and go to next ngram */
			continue;
		    }
		} else {
		    vocab.addWords(&words[1], wids, maxNgramOrder);
		}
		Vocab::reverse(wids);
		
		/*
		 * Store bow, if any
		 */
		if (howmany == (unsigned)state + 2 && (unsigned)state < order) {
		    *insertBOW(wids) = bow;
		}

		/*
		 * Save the last word (which is now the first, due to reversal)
		 * then use the first n-1 to index into
		 * the context trie, storing the prob.
		 */
		BOnode *bonode = contexts.find(&wids[1]);
		if (!bonode) {
		    file.position() << "warning: no bow for prefix of ngram \""
				    << &words[1] << "\"\n";
		} else {
		    if (!warnedAboutUnk &&
			wids[0] == vocab.unkIndex() &&
			prob != LogP_Zero &&
		    	!vocab.unkIsWord())
		    {
			file.position() << "warning: non-zero probability for "
			                << vocab.getWord(vocab.unkIndex())
			                << " in closed-vocabulary LM\n";
			warnedAboutUnk = true;
		    }

		    /* efficient for: *insertProb(wids[0], &wids[1]) = prob */
		    *bonode->probs.insert(wids[0]) = prob;
		}

		/*
		 * Hey, we're done with this ngram!
		 */
	    }
	}
    }

    /*
     * we reached a premature EOF
     */
    file.position() << "reached EOF before \\end\\\n";
    return false;
}

Boolean
Ngram::write(File &file)
{
    if (writeInBinary) {
	return writeBinaryNgram(file);
    } else {
	return writeWithOrder(file, order);
    }
};

Boolean
Ngram::writeWithOrder(File &file, unsigned order)
{
    unsigned i;
    Count howmanyNgrams[maxNgramOrder + 1];
    VocabIndex context[maxNgramOrder + 2];
    VocabString scontext[maxNgramOrder + 1];

    if (order > maxNgramOrder) {
	order = maxNgramOrder;
    }

    file.fprintf("\n\\data\\\n");

    for (i = 1; i <= order; i++ ) {
	howmanyNgrams[i] = numNgrams(i);
	file.fprintf("ngram %d=%lld\n", i, (long long)howmanyNgrams[i]);
    }

    for (i = 1; i <= order; i++ ) {
	file.fprintf("\n\\%d-grams:\n", i);

        if (debug(DEBUG_WRITE_STATS)) {
	    dout() << "writing " << howmanyNgrams[i] << " "
		   << i << "-grams\n";
	}
        
	NgramBOsIter iter(*this, context + 1, i - 1, vocab.compareIndex());
	BOnode *node;

	while ((node = iter.next())) {

	    vocab.getWords(context + 1, scontext, maxNgramOrder + 1);
	    Vocab::reverse(scontext);

	    NgramProbsIter piter(*node, vocab.compareIndex());
	    VocabIndex pword;
	    LogP *prob;

	    while ((prob = piter.next(pword))) {
		if (file.error()) {
		    return false;
		}

		if (codebook) {
		    file.fprintf("%u\t", codebook->getBin(*prob));
		} else {
		    file.fprintf("%.*lg\t", LogP_Precision,
				    (double)(*prob == LogP_Zero ?
						    LogP_PseudoZero : *prob));
		}
		Vocab::write(file, scontext);
		file.fprintf("%s%s", (i > 1 ? " " : ""), vocab.getWord(pword));

		if (i < order) {
		    context[0] = pword;

		    LogP *bow = findBOW(context);
		    if (bow) {
			if (codebook) {
			    file.fprintf("\t%u", codebook->getBin(*bow));
			} else {
			    file.fprintf("\t%.*lg", LogP_Precision,
					    (double)(*bow == LogP_Zero ?
							LogP_PseudoZero : *bow));
			}
		    }
		}

		file.fprintf("\n");
	    }
	}
    }

    file.fprintf("\n\\end\\\n");

    return true;
}

/* 
 * New binary format:
	magic string \n
	max order \n
	vocabulary index map
	probs-and-bow-trie (binary)
 */
Boolean
Ngram::writeBinaryNgram(File &file)
{
    /*
     * Magic string
     */
    file.fprintf("%s", Ngram_BinaryFormatString);

    /*
     * Maximal count order
     */
    file.fprintf("maxorder %u\n", order);

    /*
     * Vocabulary index
     */
    vocab.writeIndexMap(file, true);

    long long offset = file.ftell();

    // detect if file is not seekable
    if (offset < 0) {
	file.position() << srilm_ts_strerror(errno) << endl;
	return false;
    }

    /*
     * Context trie data 
     */
    return writeBinaryNode(contexts, 1, file, offset);
}

/*
 * Binary format for context trie:
	length-of-subtrie
	back-off-weight
	number-of-probs
  	word1
	prob1
 	word2
	prob2
	...
	word1
	subtrie1
	word2
	subtrie2
	...
 */
Boolean
Ngram::writeBinaryNode(BOtrie &node, unsigned level, File &file,
							long long &offset)
{
    BOnode &bo = node.value();

    // guess number of bytes needed for storing subtrie rooted at node
    // based on its depth (if we guess wrong we need to redo the whole
    // subtrie later)
    unsigned subtrieDepth = order - level;
    unsigned offsetBytes = subtrieDepth == 0 ? 2 :
			    subtrieDepth <= 3 ? 4 : 8;

    long long startOffset = offset;	// remember start offset

retry:
    // write placeholder value
    unsigned nbytes = writeBinaryCount(file, (unsigned long long)0,
							    offsetBytes);
    if (!nbytes) return false;
    offset += nbytes;

    // write backoff weight
    nbytes = writeBinaryCount(file, bo.bow);
    if (!nbytes) return false;
    offset += nbytes;

    // write probs
    unsigned numProbs = bo.probs.numEntries();

    nbytes = writeBinaryCount(file, numProbs);
    if (!nbytes) return false;
    offset += nbytes;

    VocabIndex word;
    LogP *pprob;

    // write probabilities -- always in index-sorted order to ensure fast
    // reading regardless of data structure used
#ifdef USE_SARRAY_TRIE
    PROB_ITER_T<VocabIndex,LogP> piter(bo.probs);
#else
    PROB_ITER_T<VocabIndex,LogP> piter(bo.probs, SArray_compareKey);
#endif

    while ((pprob = piter.next(word))) {
	nbytes = writeBinaryCount(file, word);
	if (!nbytes) return false;
	offset += nbytes;

	nbytes = writeBinaryCount(file, *pprob);
	if (!nbytes) return false;
	offset += nbytes;
    }

    // write subtries
#ifdef USE_SARRAY_TRIE
    TrieIter<VocabIndex,BOnode> iter(node);
#else
    TrieIter<VocabIndex,BOnode> iter(node, SArray_compareKey);
#endif

    BOtrie *sub;
    while ((sub = iter.next(word))) {
	nbytes = writeBinaryCount(file, word);
	if (!nbytes) return false;
	offset += nbytes;

	if (!writeBinaryNode(*sub, level + 1, file, offset)) {
	    return false;
	}
    }

    long long endOffset = offset;

    if (file.fseek(startOffset, SEEK_SET) < 0) {
	file.offset() << srilm_ts_strerror(errno) << endl;
	return false;
    }

    // don't update offset since we're skipping back in file
    nbytes = writeBinaryCount(file,
			      (unsigned long long)(endOffset-startOffset),
			      offsetBytes);
    if (!nbytes) return false;

    // now check that the number of bytes used for offset was actually ok
    if (nbytes > offsetBytes) {
	file.offset() << "increasing offset bytes from " << offsetBytes
		      << " to " << nbytes
		      << " (order " << order << ","
		      << " level " << level << ")\n";

	offsetBytes = nbytes;

	if (file.fseek(startOffset, SEEK_SET) < 0) {
	    file.offset() << srilm_ts_strerror(errno) << endl;
	    return false;
	}
	offset = startOffset;

	goto retry;
    }

    if (file.fseek(endOffset, SEEK_SET) < 0) {
	file.offset() << srilm_ts_strerror(errno) << endl;
	return false;
    }

    return true;
}

/*
 * old, machine-dependent binary format output -- no longer used 
 */
Boolean
Ngram::writeBinaryV1(File &file)
{
    const char *lmName;
    
    if (!file.name || stdio_filename_p(file.name)) {
    	lmName = "NGRAM";
    } else {
    	lmName = file.name;
    }

    char indexFile[MAXPATHLEN], dataFile[MAXPATHLEN];

    sprintf(indexFile, "%.1010s.idx%s", lmName, GZIP_SUFFIX);
    sprintf(dataFile, "%.1020s.dat", lmName);

    // open files in binary mode, except when using compression
    File idx(indexFile, (GZIP_SUFFIX)[0] ? "w" : "wb"), dat(dataFile, "wb");

    file.fprintf("%s", Ngram_BinaryFormatString);
    if (file.name && !stdio_filename_p(file.name)) {
    	// use wildcard notation since data files use same prefix as top-level 
	file.fprintf("index: %c.idx%s\n", NgramBinaryIOWildcard, GZIP_SUFFIX);
	file.fprintf("data: %c.dat\n", NgramBinaryIOWildcard);  
    } else {
	file.fprintf("index: %s\n", indexFile);
	file.fprintf("data: %s\n", dataFile);  
    }

    vocab.writeIndexMap(file, true);

    long long offset = dat.ftell();

    // detect if dat file is not seekable
    if (offset < 0) {
	dat.position() << srilm_ts_strerror(errno) << endl;
	return false;
    }

    // write magic numbers
    if (idx.fwrite(&Ngram_BinaryMagicLonglong, sizeof(Ngram_BinaryMagicLonglong), 1) != 1) {
    	idx.offset() << "write failure: " << srilm_ts_strerror(errno) << endl;
	return false;
    }

    if (dat.fwrite(&Ngram_BinaryMagicFloat, sizeof(Ngram_BinaryMagicFloat), 1) != 1) {
    	dat.offset() << "write failure: " << srilm_ts_strerror(errno) << endl;
    	return false;
    }
    
    return writeBinaryV1Node(contexts, idx, dat, offset, 1);  
}

Boolean
Ngram::writeBinaryV1Node(BOtrie &trie, File &idx, File &dat, long long &offset,
							      unsigned myOrder)
{
    BOnode &bo = trie.value();
    
    // data
    unsigned num = bo.probs.numEntries();
    if (dat.fwrite(&num, sizeof(unsigned), 1) != 1 ||
        dat.fwrite(&bo.bow, sizeof(LogP), 1) != 1)
    {
    	dat.offset() << "write failure: " << srilm_ts_strerror(errno) << endl;
	return false;
    }

    VocabIndex word;
    LogP *pprob;

    PROB_ITER_T<VocabIndex,LogP> piter(bo.probs);
    while ((pprob = piter.next(word))) {
    	if (dat.fwrite(&word, sizeof(VocabIndex), 1) != 1 ||
	    dat.fwrite(pprob, sizeof(LogP), 1) != 1)
	{
	    dat.offset() << "write failure: " << srilm_ts_strerror(errno) << endl;
	    return false;
	}
    }
    
    // move the cursor
    offset = dat.ftell();
    if (offset < 0) {
	dat.offset() << srilm_ts_strerror(errno) << endl;
    	return false;
    }
    TrieIter<VocabIndex,BOnode> iter(trie);

    BOtrie *sub;
    while ((sub = iter.next(word))) {
	if (idx.fwrite(&myOrder, sizeof(unsigned), 1) != 1 ||
	    idx.fwrite(&word, sizeof(VocabIndex), 1) != 1 ||
	    idx.fwrite(&offset, sizeof(long long), 1) != 1)
	{
	    idx.offset() << "write failure: " << srilm_ts_strerror(errno) << endl;
	    return false;
	}
	if (!writeBinaryV1Node(*sub, idx, dat, offset, myOrder + 1)) {
	    return false;
	}
    }
    
    long long tmpo = -1;
    VocabIndex tmpw = Vocab_None;

    if (idx.fwrite(&myOrder, sizeof(unsigned), 1) != 1 ||
        idx.fwrite(&tmpw, sizeof(VocabIndex), 1) != 1 ||
	idx.fwrite(&tmpo, sizeof(long long), 1) != 1)
    {
	idx.offset() << "write failure: " << srilm_ts_strerror(errno) << endl;
    	return false;
    }

    return true;
}

/*
 * Machine-independent binary format
 */
Boolean 
Ngram::readBinary(File &file, Boolean limitVocab)
{
    char *line = file.getline();

    if (!line || strcmp(line, Ngram_BinaryFormatString) != 0) {
	file.position() << "bad binary format\n";
	return false;
    }

    /*
     * Maximal count order
     */
    line = file.getline();
    unsigned maxOrder;
    if (!line || (sscanf(line, "maxorder %u", &maxOrder) != 1)) {
    	file.position() << "could not read ngram order\n";
	return false;
    }

    /*
     * Vocabulary map
     */
    Array<VocabIndex> vocabMap;  
    
    if (!vocab.readIndexMap(file, vocabMap, limitVocab)) {
	return false;
    }

    long long offset = file.ftell();

    // detect if file is not seekable
    if (offset < 0) {
	file.position() << srilm_ts_strerror(errno) << endl;
	return false;
    }

    clear();

    /* 
     * LM data
     */
    return readBinaryNode(contexts, this->order, maxOrder, file, offset,
							limitVocab, vocabMap);
}

Boolean
Ngram::readBinaryNode(BOtrie &node, unsigned order, unsigned maxOrder,
					File &file, long long &offset,
					Boolean limitVocab,
					Array<VocabIndex> &vocabMap)
{
    if (maxOrder == 0) {
    	return true;
    } else {
	long long endOffset;
	unsigned long long trieLength;
	unsigned nbytes;

	nbytes = readBinaryCount(file, trieLength);
	if (!nbytes) {
	    return false;
	}
	endOffset = offset + trieLength;
	offset += nbytes;

	if (order == 0) {
	    if (file.fseek(endOffset, SEEK_SET) < 0) {
		file.offset() << srilm_ts_strerror(errno) << endl;
		return false;
	    }
	    offset = endOffset;
	} else {
	    BOnode &bo = node.value();

	    // read backoff weight
	    nbytes = readBinaryCount(file, bo.bow);
	    if (!nbytes) return false;
	    offset += nbytes;
	   
	    // read probs
	    unsigned numProbs;
	    nbytes = readBinaryCount(file, numProbs);
	    if (!nbytes) return false;
	    offset += nbytes;

	    for (unsigned i = 0; i < numProbs; i ++) {
		VocabIndex oldWid;

		nbytes = readBinaryCount(file, oldWid);
		if (!nbytes) return false;
		offset += nbytes;

		if (oldWid >= vocabMap.size()) {
		    file.offset() << "word index " << oldWid
		                  << " out of range\n";
		    return false;
		}
		VocabIndex wid = vocabMap[oldWid];

		LogP prob;

		nbytes = readBinaryCount(file, prob);
		if (!nbytes) return false;
		offset += nbytes;

		if (wid != Vocab_None) {
		    *bo.probs.insert(wid) = prob;
		}
	    }

	    // read subtries
	    while (offset < endOffset) {
		VocabIndex oldWid;

		nbytes = readBinaryCount(file, oldWid);
		if (!nbytes) return false;
		offset += nbytes;

		if (oldWid >= vocabMap.size()) {
		    file.offset() << "word index " << oldWid
		                  << " out of range\n";
		    return false;
		}
		VocabIndex wid = vocabMap[oldWid];

		if (wid == Vocab_None) {
		    // skip subtrie
		    if (!readBinaryNode(node, 0, maxOrder-1, file, offset,
							limitVocab, vocabMap)) {
			return false;
		    }
		} else {
		    // create subtrie and read it
		    BOtrie *child = node.insertTrie(wid);

		    if (!readBinaryNode(*child, order-1, maxOrder-1,
					file, offset, limitVocab, vocabMap)) {
			return false;
		    }
		}
	    }

	    if (offset != endOffset) {
	    	file.offset() << "data misaligned\n";
		return false;
	    }
	}

	return true;
    }
}


/* 
 * Old, machine-dependent binary format
 * 	(supported temporarily for backward compatibility)
 */
Boolean 
Ngram::readBinaryV1(File &file, Boolean limitVocab)
{
    char datFile[MAXPATHLEN], idxFile[MAXPATHLEN];
    char *line;

    if (!(line = file.getline()) ||
	sscanf(line, "index: %1023s", idxFile) != 1 ||
	!(line = file.getline()) ||
	sscanf(line, "data: %1023s", datFile) != 1) {

	file.position() << "invalid binary LM format!" << endl;
	return false;
    }

    if (file.name && idxFile[0] == NgramBinaryIOWildcard) {
	makeArray(char, buffer, strlen(file.name) + strlen(idxFile));
    	sprintf(buffer, "%s%s", file.name, &idxFile[1]);
	sprintf(idxFile, "%.1023s", (char *)buffer);
    }
    if (file.name && datFile[0] == NgramBinaryIOWildcard) {
	makeArray(char, buffer, strlen(file.name) + strlen(datFile));
    	sprintf(buffer, "%s%s", file.name, &datFile[1]);
	sprintf(datFile, "%.1023s", (char *)buffer);
    }

    Array<VocabIndex> vocabMap;  
    
    if (!vocab.readIndexMap(file, vocabMap, limitVocab)) {
	return false;
    }
    
    // open files in binary mode, except when using compression
    File idx(idxFile, (GZIP_SUFFIX)[0] ? "r" : "rb"), dat(datFile, "rb");

    long long tmpll;
    LogP tmpf;
    if (idx.fread(&tmpll, sizeof(tmpll), 1) != 1 ||
	tmpll != Ngram_BinaryMagicLonglong)
    {
	idx.offset() << "incompatible binary format" << endl;
	return false;
    }

    if (dat.fread(&tmpf, sizeof(tmpf), 1) != 1 ||
	tmpf != Ngram_BinaryMagicFloat)
    {
	dat.offset() << "incompatible binary format" << endl;
	return false;
    }
	
    clear();

    return readBinaryV1Node(contexts, idx, dat, limitVocab, vocabMap, 1);
}

Boolean
Ngram::readBinaryV1Node(BOtrie &trie, File &idx, File &dat, Boolean limitVocab,
			    Array<VocabIndex> &vocabMap, unsigned myOrder)
{
    BOnode &bo = trie.value();
    
    LogP bow, prob;
    unsigned nw;
    VocabIndex word, nwid;
    VocabIndex *map = vocabMap.data();
    VocabIndex vmax = vocabMap.size();
    long long offset;

    if (myOrder > order) {
	return skipToNextTrie(idx, myOrder);
    }

    unsigned num;

    if (dat.fread(&num, sizeof(unsigned), 1) != 1 ||
	dat.fread(&bow, sizeof(LogP), 1) != 1)
    {
      dat.offset() << "failed to read from data file" << endl;
      return false;
    }

    bo.bow = bow;

    if (!limitVocab) {
    	// make probs table just as large as needed
    	// @kw false positive: SV.TAINTED.ALLOC_SIZE (num)
    	bo.probs.clear(num);
    }
    
    PROB_INDEX_T<VocabIndex, LogP> & probs = bo.probs;
    
    unsigned i = 0;
    for (i = 0; i < num; i ++) {
	if (dat.fread(&word, sizeof(VocabIndex), 1) != 1 ||
	    dat.fread(&prob, sizeof(LogP), 1) != 1)
	{
	    dat.offset() << "failed to read from data file" << endl;
	    return false;
	}

	if (word >= vmax) {
	    idx.offset() << "index (" << word << ") out of range" << endl;
	    return false;
	}
	nwid = map[word];

	if (nwid == Vocab_None) continue; // skip if out of vocab
	
	// insert data 
	*bo.probs.insert(nwid) = prob;
    }

    unsigned l;

    while (idx.fread(&l, sizeof(unsigned), 1) == 1 &&
	   idx.fread(&word, sizeof(VocabIndex), 1) == 1 &&
	   idx.fread(&offset, sizeof(long long), 1) == 1)
    {
	assert (l == myOrder);
      
	if (offset == -1 && word == Vocab_None) {
	    // this is the end of the trie
	    return true;
	}

	if (word >= vmax) {
	    idx.offset() << "index (" << word << ") out of range" << endl;
	    return false;
	}
	
	nwid = map[word];
	if (nwid == Vocab_None) {
	    skipToNextTrie(idx, l + 1);
	    continue;  // skip this word if not in vocab;     
	}

	BOtrie *sub = trie.insertTrie(nwid);
	
	if (dat.fseek(offset, SEEK_SET) < 0) {
	    dat.offset() << srilm_ts_strerror(errno) << endl;
	    return false;
	}
	
	if (!readBinaryV1Node(*sub, idx, dat, limitVocab, vocabMap, l + 1)) {
	    return false;
	}
    }

    return true;
}

Boolean
Ngram::skipToNextTrie(File &idx, unsigned myOrder)
{
    unsigned o;
    VocabIndex w;
    long long t;

    while (idx.fread(&o, sizeof(unsigned), 1) == 1 &&
	   idx.fread(&w, sizeof(VocabIndex), 1) == 1 &&
	   idx.fread(&t, sizeof(long long), 1) == 1)
    {
	if (o == myOrder && w == Vocab_None && t == -1) return true;

	if (o < myOrder) break;
    }

    idx.offset() << "skipToNextTrie failed for order " << myOrder << endl;
    return false;
}

Count
Ngram::numNgrams(unsigned int order) const
{
    if (order < 1) {
	return 0;
    } else {
	Count howmany = 0;

	makeArray(VocabIndex, context, order + 1);

	NgramBOsIter iter(*this, context, order - 1);
	BOnode *node;

	while ((node = iter.next())) {
	    howmany += node->probs.numEntries();
	}

	return howmany;
    }
}

static int
compareLogPs(const void *p1, const void *q1)
{
    LogP p = *(LogP *)p1;
    LogP q = *(LogP *)q1;

    if (p < q) return -1;
    else if (p > q) return 1;
    else return 0;
}

Count
Ngram::countParams(SArray<LogP, FloatCount> &params)
{
    Count totalCount = 0;

    Array<LogP> probsAndBows;

    Count howmanyNgrams[maxNgramOrder + 1];
    VocabIndex context[maxNgramOrder + 2];

    for (unsigned i = 1; i <= order; i++ ) {
	NgramBOsIter iter(*this, context + 1, i - 1);
	BOnode *node;

	while ((node = iter.next())) {
	    /*
	     * Back-off weight
	     */
	    probsAndBows[totalCount ++] = node->bow;

	    /*
	     * Probabilities at this node
	     */
	    NgramProbsIter piter(*node);
	    VocabIndex pword;
	    LogP *prob;

	    while ((prob = piter.next(pword))) {
		probsAndBows[totalCount ++] = *prob;
	    }
	}
    }

    /*
     * sort params to allow efficient insertion in SArray
     */
    qsort(probsAndBows.data(), totalCount, sizeof(LogP), compareLogPs);

    for (unsigned i = 0; i < totalCount; i ++) {
	*params.insert(probsAndBows[i]) += 1;
    }

    return totalCount;
}

/*
 * Estimation
 */

Boolean
Ngram::estimate(NgramStats &stats, Count *mincounts, Count *maxcounts)
{
    /*
     * If no discount method was specified we do the default, standard
     * thing. Good Turing discounting with the specified min and max counts
     * for all orders.
     */
    Discount **discounts = new Discount *[order];
    assert(discounts != 0);
    unsigned i;
    Boolean error = false;

    for (i = 1; !error && i <= order; i++) {
	discounts[i-1] =
		new GoodTuring(mincounts ? mincounts[i-1] : GT_defaultMinCount,
			       maxcounts ? maxcounts[i-1] : GT_defaultMaxCount);
	/*
	 * Transfer the LMStats's debug level to the newly
	 * created discount objects
	 */
	discounts[i-1]->debugme(stats.debuglevel());

	if (!discounts[i-1]->estimate(stats, i)) {
	    cerr << "failed to estimate GT discount for order " << i + 1
		 << endl;
	    error = true;
	} else if (debug(DEBUG_PRINT_GTPARAMS)) {
		dout() << "Good Turing parameters for " << i << "-grams:\n";
		File errfile(stderr);
		discounts[i-1]->write(errfile);
	}
    }

    if (!error) {
	error = !estimate(stats, discounts);
    }

    for (i = 1; i <= order; i++) {
	delete discounts[i-1];
    }
    delete [] discounts;

    return !error;
}

/*
 * Count number of vocabulary items that get probability mass
 */
unsigned
Ngram::vocabSize()
{
    unsigned numWords = 0;
    VocabIter viter(vocab);
    VocabIndex word;

    while (viter.next(word)) {
	if (!vocab.isNonEvent(word) && !vocab.isMetaTag(word)) {
	    numWords ++;
	}
    }
    return numWords;
}

/*
 * Generic version of estimate(NgramStats, Discount)
 *                and estimate(NgramCounts<FloatCount>, Discount)
 *
 * XXX: This function is essentially duplicated in other places.
 * Propate changes to VarNgram::estimate().
 */
template <class CountType>
Boolean
Ngram::estimate2(NgramCounts<CountType> &stats, Discount **discounts)
{
    /*
     * For all ngrams, compute probabilities and apply the discount
     * coefficients.
     */
    makeArray(VocabIndex, context, order);
    unsigned vocabSize = Ngram::vocabSize();

    /*
     * Remove all old contexts ...
     */
    clear();

    /*
     * ... but save time by allocating unigram probabilities for all words in
     * the vocabulary.
     */
    {
	VocabIndex emptyContext = Vocab_None;
	contexts.find(&emptyContext)->probs.setsize(vocab.numWords());
    }

    /*
     * Ensure <s> unigram exists (being a non-event, it is not inserted
     * in distributeProb(), yet is assumed by much other software).
     */
    if (vocab.ssIndex() != Vocab_None) {
	context[0] = Vocab_None;
	*insertProb(vocab.ssIndex(), context) = LogP_Zero;
    }

    for (unsigned i = 1; i <= order; i++) {
	unsigned noneventContexts = 0;
	unsigned noneventNgrams = 0;
	unsigned discountedNgrams = 0;

	/*
	 * check if discounting is disabled for this round
	 */
	Boolean noDiscount =
			(discounts == 0) ||
			(discounts[i-1] == 0) ||
			discounts[i-1]->nodiscount();

	/*
	 * modify counts are required by discounting method
	 */
	if (!noDiscount && discounts && discounts[i-1]) {
	    discounts[i-1]->prepareCounts(stats, i, order);
	}

	/*
	 * This enumerates all contexts, i.e., i-1 grams.
	 */
	CountType *contextCount;
	NgramCountsIter<CountType> contextIter(stats, context, i-1);

	while ((contextCount = contextIter.next())) {
	    /*
	     * Skip contexts ending in </s>.  This typically only occurs
	     * with the doubling of </s> to generate trigrams from
	     * bigrams ending in </s>.
	     * If <unk> is not real word, also skip context that contain
	     * it.
	     */
	    if ((i > 1 && context[i-2] == vocab.seIndex()) ||
	        (vocab.isNonEvent(vocab.unkIndex()) &&
				 vocab.contains(context, vocab.unkIndex())))
	    {
		noneventContexts ++;
		continue;
	    }

	    /*
	     * Re-determine if interpolated discounting is in effect.
	     * (This might be modified in the retry loop below.)
	     */
	    Boolean interpolate =
			(discounts != 0) &&
			(discounts[i-1] != 0) &&
			discounts[i-1]->interpolate;

	    VocabIndex word[2];	/* the follow word */
	    NgramCountsIter<CountType> followIter(stats, context, word, 1);
	    CountType *ngramCount;

	    /*
	     * Total up the counts for the denominator
	     * (the lower-order counts may not be consistent with
	     * the higher-order ones, so we can't just use *contextCount)
	     * Only if the trustTotal flag is set do we override this
	     * with the count from the context ngram.
	     */
	    CountType totalCount = 0;
	    Count observedVocab = 0, min2Vocab = 0, min3Vocab = 0;
	    while ((ngramCount = followIter.next())) {
		if (vocab.isNonEvent(word[0]) ||
		    ngramCount == 0 ||
		    (i == 1 && vocab.isMetaTag(word[0])))
		{
		    continue;
		}

		if (!vocab.isMetaTag(word[0])) {
		    totalCount += *ngramCount;
		    observedVocab ++;
		    if (*ngramCount >= 2) {
			min2Vocab ++;
		    }
		    if (*ngramCount >= 3) {
			min3Vocab ++;
		    }
		} else {
		    /*
		     * Process meta-counts
		     */
		    unsigned type = vocab.typeOfMetaTag(word[0]);
		    if (type == 0) {
			/*
			 * a count total: just add to the totalCount
			 * the corresponding type count can't be known,
			 * but it has to be at least 1
			 */
			totalCount += *ngramCount;
			observedVocab ++;
		    } else {
			/*
			 * a count-of-count: increment the word type counts,
			 * and infer the totalCount
			 */
			totalCount += type * *ngramCount;
			observedVocab += (Count)*ngramCount;
			if (type >= 2) {
			    min2Vocab += (Count)*ngramCount;
			}
			if (type >= 3) {
			    min3Vocab += (Count)*ngramCount;
			}
		    }
		}
	    }

	    if (i > 1 && trustTotals()) {
		totalCount = *contextCount;
	    }

	    if (totalCount == 0) {
		continue;
	    }

	    /*
	     * reverse the context ngram since that's how
	     * the BO nodes are indexed.
	     */
	    Vocab::reverse(context);

	    /*
	     * Compute the discounted probabilities
	     * from the counts and store them in the backoff model.
	     */
	retry:
	    followIter.init();
	    Prob totalProb = 0.0;

	    while ((ngramCount = followIter.next())) {
		LogP lprob;
		double discount;

		/*
		 * Ignore zero counts.
		 * They are there just as an artifact of the count trie
		 * if a higher order ngram has a non-zero count.
		 */
		if (i > 1 && *ngramCount == 0) {
		    continue;
		}

		if (vocab.isNonEvent(word[0]) || vocab.isMetaTag(word[0])) {
		    /*
		     * Discard all pseudo-word probabilities,
		     * except for unigrams.  For unigrams, assign
		     * probability zero.  This will leave them with
		     * prob zero in all cases, due to the backoff
		     * algorithm.
		     * Also discard the <unk> token entirely in closed
		     * vocab models, its presence would prevent OOV
		     * detection when the model is read back in.
		     */
		    if (i > 1 ||
			word[0] == vocab.unkIndex() ||
			vocab.isMetaTag(word[0]))
		    {
			noneventNgrams ++;
			continue;
		    }

		    lprob = LogP_Zero;
		    discount = 1.0;
		} else {
		    /*
		     * Ths discount array passed may contain 0 elements
		     * to indicate no discounting at this order.
		     */
		    if (noDiscount) {
			discount = 1.0;
		    } else {
			discount =
			    discounts[i-1]->discount(*ngramCount, totalCount,
								observedVocab);
		    }
		    Prob prob = (discount * *ngramCount) / totalCount;

		    /*
		     * For interpolated estimates we compute the weighted 
		     * linear combination of the high-order estimate
		     * (computed above) and the lower-order estimate.
		     * The high-order weight is given by the discount factor,
		     * the lower-order weight is obtained from the Discount
		     * method (it may be 0 if the method doesn't support
		     * interpolation).
		     */
		    double lowerOrderWeight;
		    LogP lowerOrderProb;
		    if (interpolate) {
			lowerOrderWeight = 
			    discounts[i-1]->lowerOrderWeight(totalCount,
							     observedVocab,
							     min2Vocab,
							     min3Vocab);
			if (i > 1) {
			    lowerOrderProb = wordProbBO(word[0], context, i-2);
			} else {
			    lowerOrderProb = - log10((double)vocabSize);
			}

			prob += lowerOrderWeight * LogPtoProb(lowerOrderProb);
		    }

		    lprob = ProbToLogP(prob);

		    if (discount != 0.0) {
			totalProb += prob;
		    }

		    if (discount != 0.0 && debug(DEBUG_ESTIMATES)) {
			dout() << "CONTEXT " << (vocab.use(), context)
			       << " WORD " << vocab.getWord(word[0])
			       << " NUMER " << *ngramCount
			       << " DENOM " << totalCount
			       << " DISCOUNT " << discount;

			if (interpolate) {
			    dout() << " LOW " << lowerOrderWeight
				   << " LOLPROB " << lowerOrderProb;
			}
			dout() << " LPROB " << lprob << endl;
		    }
		}
		    
		/*
		 * A discount coefficient of zero indicates this ngram
		 * should be omitted entirely (presumably to save space).
		 */
		if (discount == 0.0) {
		    discountedNgrams ++;
		    removeProb(word[0], context);
		} else {
		    *insertProb(word[0], context) = lprob;
		} 
	    }

	    /*
	     * This is a hack credited to Doug Paul (by Roni Rosenfeld in
	     * his CMU tools).  It may happen that no probability mass
	     * is left after totalling all the explicit probs, typically
	     * because the discount coefficients were out of range and
	     * forced to 1.0.  Unless we have seen all vocabulary words in
	     * this context, to arrive at some non-zero backoff mass,
	     * we try incrementing the denominator in the estimator by 1.
	     * Another hack: If the discounting method uses interpolation 
	     * we first try disabling that because interpolation removes
	     * probability mass.
	     */
	    if (!noDiscount && totalCount > 0 &&
		observedVocab < vocabSize &&
		totalProb > 1.0 - Prob_Epsilon)
	    {
		if (debug(DEBUG_ESTIMATE_WARNINGS)) {
		    cerr << "warning: " << (1.0 - totalProb)
			 << " backoff probability mass left for \""
			 << (vocab.use(), context)
			 << "\" -- ";
		    if (interpolate) {
			 cerr << "disabling interpolation\n";
		    } else {
			 cerr << "incrementing denominator\n";
		    }
		}

		if (interpolate) {
		    interpolate = false;
		} else {
		    totalCount += 1;
		}

		goto retry;
	    }

	    /*
	     * Undo the reversal above so the iterator can continue correctly
	     */
	    Vocab::reverse(context);
	}

	if (debug(DEBUG_ESTIMATE_WARNINGS)) {
	    if (noneventContexts > 0) {
		dout() << "discarded " << noneventContexts << " "
		       << i << "-gram contexts containing pseudo-events\n";
	    }
	    if (noneventNgrams > 0) {
		dout() << "discarded " << noneventNgrams << " "
		       << i << "-gram probs predicting pseudo-events\n";
	    }
	    if (discountedNgrams > 0) {
		dout() << "discarded " << discountedNgrams << " "
		       << i << "-gram probs discounted to zero\n";
	    }
	}

	/*
	 * With all the probs in place, BOWs are obtained simply by the usual
	 * normalization.
	 * We do this right away before computing probs of higher order since 
	 * the estimation of higher-order N-grams can refer to lower-order
	 * ones (e.g., for interpolated estimates).
	 */
	computeBOWs(i-1);
    }

    fixupProbs();

    return true;
}

Boolean
Ngram::estimate(NgramStats &stats, Discount **discounts)
{
    return estimate2(stats, discounts);
}

Boolean
Ngram::estimate(NgramCounts<FloatCount> &stats, Discount **discounts)
{
    return estimate2(stats, discounts);
}

/*
 * Mixing backoff models
 *	the first version mixes an existing models destructively with
 *	a second one
 *	the second version mixes two existing models non-destructively leaving
 *	the result in *this.  lambda is the weight of the first input model.
 *	the third version mixes two or more Ngram models that are encapsulated 
 *	into NgramBayesMix object.
 */
void
Ngram::mixProbs(Ngram &lm2, double lambda)
{
    makeArray(VocabIndex, context, order + 1);

    /*
     * In destructive merging we need to process the longer ngrams first
     * so that we can still use the model being modified to compute its own
     * original probability estimates.
     */
    for (int i = order - 1; i >= 0 ; i--) {
	BOnode *node;
	NgramBOsIter iter1(*this, context, i);
	
	/*
	 * First, find all explicit ngram probs in *this, and mix them
	 * with the corresponding probs of lm2 (explicit or backed-off).
	 */
	while ((node = iter1.next())) {
	    NgramProbsIter piter(*node);
	    VocabIndex word;
	    LogP *prob1;

	    while ((prob1 = piter.next(word))) {
		*prob1 =
		    MixLogP(*prob1, lm2.wordProbBO(word, context, i), lambda);
	    }
	}

	/*
	 * Do the same for lm2, except we dont't need to recompute 
	 * those cases that were already handled above (explicit probs
	 * in both *this and lm2).
	 */
	NgramBOsIter iter2(lm2, context, i);

	while ((node = iter2.next())) {
	    NgramProbsIter piter(*node);
	    VocabIndex word;
	    LogP *prob2;

	    while ((prob2 = piter.next(word))) {
		if (!findProb(word, context)) {
		    LogP mixProb =
			MixLogP(wordProbBO(word, context, i), *prob2, lambda);
		    *insertProb(word, context) = mixProb;
		}
	    }
	}
    }

    recomputeBOWs();
}

void
Ngram::mixProbs(Ngram &lm1, Ngram &lm2, double lambda)
{
    NgramBayesMix mixLM(vocab, lm1, lm2, 0, lambda);

    return mixProbs(mixLM);
}

void
Ngram::mixProbs(NgramBayesMix &mixLMs)
{
    makeArray(VocabIndex, context, order + 1);

    unsigned numLMs = mixLMs.numLMs;
    assert(numLMs >= 1);

    for (unsigned i = 0; i < order; i++) {
	/*
	 * First, find all explicit ngram probs in lm[0], and mix them
	 * with the corresponding probs of the other lms (explicit or backed-off).
	 * Then do the same for all other lms, except we dont't need to recompute 
	 * those cases that were already handled above (explicit probs
	 * in one of the preceding lms).
	 */
	for (unsigned j = 0; j < numLMs; j ++) {
	    const Ngram &lm_j = *mixLMs.subLM(j);
	    NgramBOsIter iter_j(lm_j, context, i);
	    BOnode *node;

	    while ((node = iter_j.next())) {
		/*
		 * Get context-dependent mixture weights
		 */
		Array<Prob> &lambdas = mixLMs.findPriors(context);

		NgramProbsIter piter(*node);
		VocabIndex word;
		LogP *prob_j;

		while ((prob_j = piter.next(word))) {
		    /*
		     * All previously seen ngrams are already inserted in *this
		     */
		    if (j == 0 || !findProb(word, context)) {
			Prob probSum = lambdas[j] * LogPtoProb(*prob_j);

			for (unsigned k = 0; k < numLMs; k++) {
			    if (k != j) {
				probSum +=
				    lambdas[k] *
				    LogPtoProb(mixLMs.subLM(k)->wordProbBO(word, context, i));
			    }
			}
			*insertProb(word, context) = ProbToLogP(probSum);
		    }
		}
	    }
	}
    }

    recomputeBOWs();
}

/*
 * Compute the numerator and denominator of a backoff weight,
 * checking for sanity.  Returns true if values make sense,
 * and prints a warning if not.
 */
Boolean
Ngram::computeBOW(BOnode *node, const VocabIndex *context, unsigned clen,
				Prob &numerator, Prob &denominator)
{
    NgramProbsIter piter(*node);
    VocabIndex word;
    LogP *prob;

    /*
     * The BOW(c) for a context c is computed to be
     *
     *	BOW(c) = (1 - Sum p(x | c)) /  (1 - Sum p_BO(x | c))
     *
     * where Sum is a summation over all words x with explicit probabilities
     * in context c, p(x|c) is that probability, and p_BO(x|c) is the 
     * probability for that word according to the backoff algorithm.
     */
    numerator = 1.0;
    denominator = 1.0;

    while ((prob = piter.next(word))) {
	numerator -= LogPtoProb(*prob);
	if (clen > 0) {
	    denominator -=
		LogPtoProb(wordProbBO(word, context, clen - 1));
	}
    }

    /*
     * Avoid some predictable anomalies due to rounding errors
     */
    if (fabs(numerator) < Prob_Epsilon) {
	numerator = 0.0;
    }
    if (fabs(denominator) < Prob_Epsilon) {
	denominator = 0.0;
    }

    if (denominator < Prob_Epsilon && numerator > Prob_Epsilon) {
	/* 
	 * Backoff distribution has no probability left.  To avoid wasting
	 * probability mass scale the N-gram probabilities to sum to 1.
	 */
	cerr << "BOW denominator for context \""
		 << (vocab.use(), context)
		 << "\" is zero; scaling probabilities to sum to 1\n";

	LogP scale = - ProbToLogP(1 - numerator);

	piter.init();
	while ((prob = piter.next(word))) {
	    *prob += scale;
	}

	denominator = 0.0;
	numerator = 0.0;
	return true;
    } else if (numerator < 0.0) {
	cerr << "BOW numerator for context \""
	     << (vocab.use(), context)
	     << "\" is " << numerator << " < 0\n";
	return false;
    } else if (denominator <= 0.0) {
	if (numerator > Prob_Epsilon) {
	    cerr << "BOW denominator for context \""
		 << (vocab.use(), context)
		 << "\" is " << denominator << " <= 0,"
		 << "numerator is " << numerator
		 << endl;
	    return false;
	} else {
	    numerator = 0.0;
	    denominator = 0.0;	/* will give bow = 0 */
	    return true;
	}
    } else {
	return true;
    }
}

/*
 * Recompute backoff weight for all contexts of a given order
 */
Boolean
Ngram::computeBOWs(unsigned order)
{
    Boolean result = true;

    /*
     * Note that this will only generate backoff nodes for those
     * contexts that have words with explicit probabilities.  But
     * this is precisely as it should be.
     */
    BOnode *node;
    makeArray(VocabIndex, context, order + 1);

    NgramBOsIter iter1(*this, context, order);
    
    while ((node = iter1.next())) {
	NgramProbsIter piter(*node);

	double numerator, denominator;

	if (computeBOW(node, context, order, numerator, denominator)) {
	    /*
	     * If unigram probs leave a non-zero probability mass
	     * then we should give that mass to the zero-order (uniform)
	     * distribution for zeroton words.  However, the ARPA
	     * BO format doesn't support a "null context" BOW.
	     * We simluate the intended distribution by spreading the
	     * left-over mass uniformly over all vocabulary items that
	     * have a zero probability.
	     * NOTE: We used to do this only if there was prob mass left,
	     * but some ngram software requires all words to appear as
	     * unigrams, which we achieve by giving them zero probability.
	     */
	    if (order == 0 /*&& numerator > 0.0*/) {
		if (numerator < Prob_Epsilon) {
		    /*
		     * Avoid spurious non-zero unigram probabilities
		     */
		    numerator = 0.0;
		}
		distributeProb(numerator, context);
	    } else if (numerator < Prob_Epsilon && denominator < Prob_Epsilon) {
		node->bow = LogP_One;
	    } else {
		node->bow = ProbToLogP(numerator) - ProbToLogP(denominator);
	    }
	} else {
	    /*
	     * Dummy value for improper models
	     */
	    node->bow = LogP_Zero;
	    result = false;
	}

	if (debug(DEBUG_ESTIMATES)) {
	    dout() << "CONTEXT " << (vocab.use(), context)
		   << " numerator " << numerator
		   << " denominator " << denominator
		   << " BOW " << node->bow
		   << endl;
	}
    }

    return result;
}

/*
 * Renormalize language model by recomputing backoff weights.
 */
void
Ngram::recomputeBOWs()
{
    /*
     * Here it is important that we compute the backoff weights in
     * increasing order, since the higher-order ones refer to the
     * lower-order ones in the backoff algorithm.
     * Note that this will only generate backoff nodes for those
     * contexts that have words with explicit probabilities.  But
     * this is precisely as it should be.
     */
    for (unsigned i = 0; i < order; i++) {
	computeBOWs(i);
    }
}

/*
 * Prune probabilities from model so that the change in training set perplexity
 * is below a threshold.
 * The minorder parameter limits pruning to ngrams of that length and above.
 */
void
Ngram::pruneProbs(double threshold, unsigned minorder, LM *historyLM)
{
    /*
     * Hack alert: allocate the context buffer for NgramBOsIter, but leave
     * room for a word id to be prepended.
     */
    makeArray(VocabIndex, wordPlusContext, order + 2);
    VocabIndex *context = &wordPlusContext[1];

    for (unsigned i = order - 1; i > 0 && i >= minorder - 1; i--) {
	unsigned prunedNgrams = 0;

	BOnode *node;
	NgramBOsIter iter1(*this, context, i);
	
	while ((node = iter1.next())) {
	    LogP bow = node->bow;	/* old backoff weight, BOW(h) */
	    double numerator, denominator;

	    /* 
	     * Compute numerator and denominator of the backoff weight,
	     * so that we can quickly compute the BOW adjustment due to
	     * leaving out one prob.
	     */
	    if (!computeBOW(node, context, i, numerator, denominator)) {
		continue;
	    }

	    /*
	     * Compute the marginal probability of the context, P(h)
	     * If a historyLM was given (e.g., for KN smoothed LMs), use it,
	     * otherwise use the LM to be pruned.
	     */
	    LogP cProb = historyLM == 0 ?
				contextProb(context, i) :
				historyLM->contextProb(context, i);

	    NgramProbsIter piter(*node);
	    VocabIndex word;
	    LogP *ngramProb;

	    Boolean allPruned = true;

	    while ((ngramProb = piter.next(word))) {
		/*
		 * lower-order estimate for ngramProb, P(w|h')
		 */
		LogP backoffProb = wordProbBO(word, context, i - 1);

		/*
		 * Compute BOW after removing ngram, BOW'(h)
		 */
		LogP newBOW =
			ProbToLogP(numerator + LogPtoProb(*ngramProb)) -
			ProbToLogP(denominator + LogPtoProb(backoffProb));

		/*
		 * Compute change in entropy due to removal of ngram
		 * deltaH = - P(H) x
		 *  {P(W | H) [log P(w|h') + log BOW'(h) - log P(w|h)] +
		 *   (1 - \sum_{v,h ngrams} P(v|h)) [log BOW'(h) - log BOW(h)]}
		 *
		 * (1-\sum_{v,h ngrams}) is the probability mass left over from
		 * ngrams of the current order, and is the same as the 
		 * numerator in BOW(h).
		 */
		LogP deltaProb = backoffProb + newBOW - *ngramProb;
		Prob deltaEntropy = - LogPtoProb(cProb) *
					(LogPtoProb(*ngramProb) * deltaProb +
					 numerator * (newBOW - bow));

		/*
		 * compute relative change in model (training set) perplexity
		 *	(PPL' - PPL)/PPL = PPL'/PPL - 1
		 *	                 = exp(H')/exp(H) - 1
		 *	                 = exp(H' - H) - 1
		 */
		double perpChange = LogPtoProb(deltaEntropy) - 1.0;

		Boolean pruned = threshold > 0 && perpChange < threshold;

		/*
		 * Make sure we don't prune ngrams whose backoff nodes are
		 * needed ...
		 */
		if (pruned) {
		    /*
		     * wordPlusContext[1 ... i-1] was already filled in by
		     * NgramBOIter. Just add the word to complete the ngram.
		     */
		    wordPlusContext[0] = word;

		    if (findBOW(wordPlusContext)) {
			pruned = false;
		    }
		}

		if (debug(DEBUG_ESTIMATES)) {
		    dout() << "CONTEXT " << (vocab.use(), context)
			   << " WORD " << vocab.getWord(word)
			   << " CONTEXTPROB " << cProb
			   << " OLDPROB " << *ngramProb
			   << " NEWPROB " << (backoffProb + newBOW)
			   << " DELTA-H " << deltaEntropy
			   << " DELTA-LOGP " << deltaProb
			   << " PPL-CHANGE " << perpChange
			   << " PRUNED " << pruned
			   << endl;
		}

		if (pruned) {
		    removeProb(word, context);
		    prunedNgrams ++;
		} else {
		    allPruned = false;
		}
	    }

	    /*
	     * If we removed all ngrams for this context we can 
	     * remove the context itself, but only if the present
	     * context is not a prefix to a longer one.
	     */
	    if (allPruned && contexts.numEntries(context) == 0) {
		removeBOW(context);
	    }
	}

	if (debug(DEBUG_ESTIMATE_WARNINGS)) {
	    if (prunedNgrams > 0) {
		dout() << "pruned " << prunedNgrams << " "
		       << (i + 1) << "-grams\n";
	    }
	}
    }

    recomputeBOWs();
}

/*
 * Prune low probability N-grams
 *	Removes all N-gram probabilities that are lower than the
 *	corresponding backed-off estimates.  This is a crucial preprocessing
 *	step for N-gram models before conversion to finite-state networks.
 */
void
Ngram::pruneLowProbs(unsigned minorder)
{
    makeArray(VocabIndex, context, order);

    Boolean havePruned;
    
    /*
     * Since pruning changes backoff weights we have to keep pruning
     * interatively as long as N-grams keep being removed.
     */
    do {
	havePruned = false;

	for (unsigned i = minorder - 1; i < order; i++) {
	    unsigned prunedNgrams = 0;

	    BOnode *node;
	    NgramBOsIter iter1(*this, context, i);

	    while ((node = iter1.next())) {
		LogP bow = node->bow;	/* old backoff weight, BOW(h) */

		NgramProbsIter piter(*node);
		VocabIndex word;
		LogP *ngramProb;

	        Boolean allPruned = true;
	    
		while ((ngramProb = piter.next(word))) {
		    /*
		     * lower-order estimate for ngramProb, P(w|h')
		     */
		    LogP backoffProb = wordProbBO(word, context, i - 1);

		    if (backoffProb + bow > *ngramProb) {

			if (debug(DEBUG_ESTIMATES)) {
			    dout() << "CONTEXT " << (vocab.use(), context)
				   << " WORD " << vocab.getWord(word)
				   << " LPROB " << *ngramProb
				   << " BACKOFF-LPROB " << (backoffProb + bow)
				   << " PRUNED\n";
			}

			removeProb(word, context);
			prunedNgrams ++;
		    } else {
			allPruned = false;
		    }
		}

		/*
		 * If we removed all ngrams for this context we can 
		 * remove the context itself, but only if the present
		 * context is not a prefix to a longer one.
		 */
		if (allPruned && contexts.numEntries(context) == 0) {
		    removeBOW(context);
		}
	    }

	    if (prunedNgrams > 0) {
		havePruned = true;
		if (debug(DEBUG_ESTIMATE_WARNINGS)) {
		    dout() << "pruned " << prunedNgrams << " "
			   << (i + 1) << "-grams\n";
		}
	    }
	}
	recomputeBOWs();

    } while (havePruned);

    fixupProbs();
}

/* 
 * Reassign ngram probabilities using a different LM
 */
void
Ngram::rescoreProbs(LM &lm)
{
    makeArray(VocabIndex, context, order + 1);

    for (int i = order - 1; i >= 0 ; i--) {
	BOnode *node;
	NgramBOsIter iter(*this, context, i);
	
	/*
	 * Enumerate all explicit ngram probs in *this, and recompute their
	 * probs using the supplied LM.
	 */
	while ((node = iter.next())) {
	    NgramProbsIter piter(*node);
	    VocabIndex word;
	    LogP *prob;

	    while ((prob = piter.next(word))) {
		*prob = lm.wordProb(word, context);
	    }
	}
    }

    recomputeBOWs();
}

/*
 * Insert probabilities for all context ngrams
 *
 * 	The ARPA format forces us to have a probability
 * 	for every context ngram.  So we create one if necessary and
 * 	set the probability to the same as what the backoff algorithm
 * 	would compute (so as not to change the distribution).
 */
void
Ngram::fixupProbs()
{
    makeArray(VocabIndex, context, order + 1);

    /*
     * we cannot insert entries into the context trie while we're
     * iterating over it, so we need another trie to collect
     * the affected contexts first. yuck.
     * Note: We use the same trie type as in the Ngram model itself,
     * so we don't need to worry about another template instantiation.
     */
    Trie<VocabIndex,NgramCount> contextsToAdd;

    /*
     * Find the contexts xyz that don't have a probability p(z|xy) in the
     * model already.
     */
    unsigned i;
    for (i = 1; i < order; i++) {
	NgramBOsIter iter(*this, context, i);
	
	while (iter.next()) {
	    /*
	     * For a context abcd we need to create probability entries
	     * p(d|abc), p(c|ab), p(b|a), p(a) if necessary.
	     * If any of these is found in that order we can stop since a
	     * previous pass will already have created the remaining ones.
	     */
	    for (unsigned j = 0; j < i; j++) {
		LogP *prob = findProb(context[j], &context[j+1]);

		if (prob) {
		    break;
		} else {
		    /*
		     * Just store something non-zero to distinguish from
		     * internal nodes that are added implicitly.
		     */
		    *contextsToAdd.insert(&context[j]) = 1;
		}
	    }
	}
    }


    /*
     * Store the missing probs
     */
    for (i = 1; i < order; i++) {
	unsigned numFakes = 0;

	Trie<VocabIndex,NgramCount> *node;
	TrieIter2<VocabIndex,NgramCount> iter(contextsToAdd, context, i);
	

	while ((node = iter.next())) {
	    if (node->value()) {
		numFakes ++;

		/*
		 * Note: we cannot combine the two statements below
		 * since insertProb() creates a zero prob entry, which would
		 * prevent wordProbBO() from computing the backed-off 
		 * estimate!
		 */
		LogP backoffProb = wordProbBO(context[0], &context[1], i - 1);
		*insertProb(context[0], &(context[1])) = backoffProb;

		if (debug(DEBUG_FIXUP_WARNINGS)) {
		    dout() << "faking probability for context "
			   << (vocab.use(), context) << endl;
		}
	    }
	}

	if (debug(DEBUG_ESTIMATE_WARNINGS)) {
	    if (numFakes > 0) {
		dout() << "inserted " << numFakes << " redundant " 
		       << i << "-gram probs\n";
	    }
	}
    }
}

/*
 * Redistribute probability mass over ngrams of given context
 */
void
Ngram::distributeProb(Prob mass, VocabIndex *context)
{
    /*
     * First enumerate the vocabulary to count the number of
     * items affected
     */
    unsigned numWords = 0;
    unsigned numZeroProbs = 0;
    VocabIter viter(vocab);
    VocabIndex word;

    while (viter.next(word)) {
	if (!vocab.isNonEvent(word) && !vocab.isMetaTag(word)) {
	    numWords ++;

	    LogP *prob = findProb(word, context);
	    if (!prob || *prob == LogP_Zero) {
		numZeroProbs ++;
	    }
	    /*
	     * create zero probs so we can update them below
	     */
	    if (!prob) {
		*insertProb(word, context) = LogP_Zero;
	    }
	}
    }

    /*
     * If there are no zero-probability words 
     * then we add the left-over prob mass to all unigrams.
     * Otherwise do as described above.
     */
    viter.init();

    if (numZeroProbs > 0) {
	if (debug(DEBUG_ESTIMATE_WARNINGS)) {
	    cerr << "warning: distributing " << mass
		 << " left-over probability mass over "
		 << numZeroProbs << " zeroton words" << endl;
	}
	Prob add = mass / numZeroProbs;

	while (viter.next(word)) {
	    if (!vocab.isNonEvent(word) && !vocab.isMetaTag(word)) {
		LogP *prob = insertProb(word, context);
		if (*prob == LogP_Zero) {
		    *prob = ProbToLogP(add);
		}
	    }
	}
    } else {
	if (mass > 0.0 && debug(DEBUG_ESTIMATE_WARNINGS)) {
	    cerr << "warning: distributing " << mass
		 << " left-over probability mass over all "
		 << numWords << " words" << endl;
	}
	Prob add = mass / numWords;

	while (viter.next(word)) {
	    if (!vocab.isNonEvent(word) && !vocab.isMetaTag(word)) {
		LogP *prob = insertProb(word, context);
		*prob = ProbToLogP(LogPtoProb(*prob) + add);
	    }
	}
    }
}

