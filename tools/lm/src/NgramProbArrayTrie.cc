/*
 * NgramProbArrayTrie.cc --
 *	Tries of Ngrams indexing Prob vectors
 *
 */

#ifndef _NgramProbArrayTrie_cc_
#define _NgramProbArrayTrie_cc_

#ifndef lint
static char NgramProbArrayTrie_Copyright[] = "Copyright (c) 2013-2016 Microsoft Corp.  All Rights Reserved.";
static char NgramProbArrayTrie_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramProbArrayTrie.cc,v 1.9 2016/04/09 06:53:01 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "NgramProbArrayTrie.h"
#include "NgramStats.h" 

#include "tserror.h"
#include "Trie.cc"
#include "SArray.h"	// for SArray_compareKey()
#include "Array.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_TRIE(VocabIndex, ZeroArray<Prob> );
#endif

void
NgramProbArrayTrie::memStats(MemStats &stats)
{
    stats.total += sizeof(*this) - sizeof(probs) - sizeof(vocab);
    vocab.memStats(stats);
    probs.memStats(stats);
}

void
NgramProbArrayTrie::clear()
{
    probs.clear();

    for (unsigned i = 0; i < dimension; i++) {
	probs.value()[i] = 0.0;
    }
}

NgramProbArrayTrie::NgramProbArrayTrie(Vocab &vocab, unsigned maxOrder, unsigned dim)
    : vocab(vocab), order(maxOrder), dimension(dim)
{
    if (dimension == 0) {
	dimension = 1;
    }
}


int
NgramProbArrayTrie::parseNgram(char *line,
			       VocabString *words,
			       unsigned max,
			       ZeroArray<Prob> &pr)
{
    makeArray(VocabString, wordsAndProbs, max + dimension);

    unsigned howmany = Vocab::parseWords(line, wordsAndProbs, max + dimension);

    if (howmany == max + dimension || howmany < dimension) {
	return -1;
    }

    /*
     * Parse the last dimension words as Probs
     */
    for (unsigned i = 0; i < dimension; i ++) {
	if (!parseProb(wordsAndProbs[howmany - dimension + i], pr[i])) {
	    return -1;
	}
    }

    /*
     * Copy the VocabString
     */
    howmany -= dimension;
    wordsAndProbs[howmany] = 0;
    Vocab::copy(words, wordsAndProbs);

    return howmany;
}

int
NgramProbArrayTrie::readNgram(File &file,
			      VocabString *words,
			      unsigned max,
			      ZeroArray<Prob> &pr)
{
    char *line;

    /*
     * Read next ngram and prob vector from file, skipping blank lines
     */
    line = file.getline();
    if (line == 0) {
	return -1;
    }

    int howmany = parseNgram(line, words, max, pr);

    if (howmany < 0) {
	file.position() << "bad "
	                << dimension << "-dimensional probs vector or more than "
	                << max - 1 << " words per line\n";
	return -2;
    }

    return howmany;
}

Boolean
NgramProbArrayTrie::read(File &file, unsigned maxOrder, Boolean limitVocab)
{
    VocabString words[maxNgramOrder + 1];
    VocabIndex wids[maxNgramOrder + 1];
    ZeroArray<Prob> pr;
    int howmany;

    while ((howmany = readNgram(file, words, maxNgramOrder + 1, pr)) >= 0) {

	/*
	 * Skip this entry if the length of the ngram exceeds our 
	 * maximum order
	 */
	if (howmany > (int)maxOrder) {
	    continue;
	}

	/* 
	 * Map words to indices
	 */
	if (limitVocab) {
	    /*
	     * skip ngram if not in-vocabulary
	     */
	    if (!vocab.checkWords(words, wids, maxNgramOrder)) {
	    	continue;
	    }
	} else {
	    vocab.addWords(words, wids, maxNgramOrder);
	}

	/*
	 *  Insert the probs
	 */
        *probs.insert(wids) = pr;
    }

    return (howmany == -1);
}

unsigned
NgramProbArrayTrie::writeNgram(File &file,
			       const VocabString *words,
			       ZeroArray<Prob> &pr)
{
    unsigned i = 0;

    if (words[0]) {
	file.fprintf("%s", words[0]);
	for (i = 1; words[i]; i++) {
	    file.fprintf(" %s", words[i]);
	}
    }

    for (unsigned i = 0; i < dimension; i++) {
	file.fprintf("%c%.*lg", (i == 0 ? '\t' : ' '),
				Prob_Precision, (double)pr[i]);
    }
    file.fprintf("\n");

    return i;
}

/*
 * For reasons of efficiency the write() method doesn't use
 * writeNgram()  (yet).  Instead, it fills up a string buffer 
 * as it descends the tree recursively.  this avoid having to
 * lookup shared prefix words and buffer the corresponding strings
 * repeatedly.
 */
void
NgramProbArrayTrie::writeNode(
    NgramProbArrayTrieNode &node, /* the trie node we're at */
    File &file,			/* output file */
    char *buffer,		/* output buffer */
    char *bptr,			/* pointer into output buffer */
    unsigned level,		/* current trie level */
    unsigned maxOrder,		/* target trie level */
    Boolean sorted)		/* produce sorted output */
{
    NgramProbArrayTrieNode *child;
    VocabIndex wid;

    TrieIter<VocabIndex,ZeroArray<Prob> > iter(node, sorted ? vocab.compareIndex() : 0);

    /*
     * Iterate over the child nodes at the current level,
     * appending their word strings to the buffer
     */
    while (!file.error() && (child = iter.next(wid))) {
	VocabString word = vocab.getWord(wid);

	if (word == 0) {
	   cerr << "undefined word index " << wid << "\n";
	   continue;
	}

	unsigned wordLen = strlen(word);

	if (bptr + wordLen + 1 > buffer + maxLineLength) {
	   *bptr = '0';
	   cerr << "ngram ["<< buffer << word
		<< "] exceeds write buffer\n";
	   continue;
	}
        
	strcpy(bptr, word);

	/*
	 * If this is the final level, print out the ngram and the prob vector.
	 * Otherwise set up another level of recursion.
	 */
	if (maxOrder == 0 || level == maxOrder) {
           ZeroArray<Prob> &pr = child->value();

	   file.fprintf("%s", buffer);
	   for (unsigned i = 0; i < dimension; i ++) {
	        file.fprintf("%c%.*lg", (i == 0 ? '\t' : ' '),
					Prob_Precision, (double)pr[i]);
	   }
	   file.fprintf("\n");
	}
	
	if (maxOrder == 0 || level < maxOrder) {
	   *(bptr + wordLen) = ' ';
	   writeNode(*child, file, buffer, bptr + wordLen + 1, level + 1,
			maxOrder, sorted);
	}
    }
}

void
NgramProbArrayTrie::write(File &file, unsigned maxOrder, Boolean sorted)
{
    VocabString empty[1]; empty[0] = 0;

    // @kw false positive: ABV.GENERAL (empty)
    writeNgram(file, empty, probs.value());
    char *buffer = TLSW_GET_ARRAY(writeBufferTLS);
    writeNode(probs, file, buffer, buffer, 1, maxOrder, sorted);
}

#endif /* _NgramProbArrayTrie_cc_ */
