/*
 * NgramStats.cc --
 *	N-gram counting
 *
 */

#ifndef _NgramStats_cc_
#define _NgramStats_cc_

#ifndef lint
static char NgramStats_Copyright[] = "Copyright (c) 1995-2012 SRI International.  All Rights Reserved.";
static char NgramStats_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStats.cc,v 1.72 2014-08-29 21:35:48 frandsen Exp $";
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

#ifdef INSTANTIATE_TEMPLATES
static
#endif
const char *NgramStats_BinaryFormatString = "SRILM_BINARY_COUNTS_001\n";

#include "NgramStats.h"

#include "tserror.h"
#include "Trie.cc"
#include "LHash.cc"
#include "Array.cc"
#include "SArray.h"	// for SArray_compareKey()

#define INSTANTIATE_NGRAMCOUNTS(CountT) \
	INSTANTIATE_TRIE(VocabIndex,CountT); \
	template class NgramCounts<CountT>

/*
 * Debug levels used
 */
#define DEBUG_READ_GOOGLE	1

template <class CountT>
void
NgramCounts<CountT>::dump()
{
    cerr << "order = " << order << endl;
    counts.dump();
}

template <class CountT>
void
NgramCounts<CountT>::memStats(MemStats &stats)
{
    stats.total += sizeof(*this) - sizeof(counts) - sizeof(vocab);
    vocab.memStats(stats);
    counts.memStats(stats);
}

template <class CountT>
NgramCounts<CountT>::NgramCounts(Vocab &vocab, unsigned int maxOrder)
    : LMStats(vocab), intersect(false), order(maxOrder)
{
}

template <class CountT>
unsigned int
NgramCounts<CountT>::countSentence(const VocabString *words, const char *factor)
{
    CountT factorCount;

    /*
     * Parse the weight string as a count
     */
    if (!stringToCount(factor, factorCount)) {
	return 0;
    }

    return countSentence(words, factorCount);
}

template <class CountT>
unsigned int
NgramCounts<CountT>::countSentence(const VocabString *words, CountT factor)
{
    VocabIndex *wids = TLSW_GET_ARRAY(countSentenceWidsTLS);
    unsigned int howmany;

    if (openVocab) {
	howmany = vocab.addWords(words, wids + 1, maxWordsPerLine + 1);
    } else {
	howmany = vocab.getIndices(words, wids + 1, maxWordsPerLine + 1,
					    vocab.unkIndex());
    }

    /*
     * Check for buffer overflow
     */
    if (howmany == maxWordsPerLine + 1) {
	return 0;
    }

    /*
     * update OOV count
     */
    if (!openVocab) {
	for (unsigned i = 1; i <= howmany; i++) {
	    if (wids[i] == vocab.unkIndex()) {
		stats.numOOVs ++;
	    }
	}
    }

    /*
     * Insert begin/end sentence tokens if necessary
     */
    VocabIndex *start;
    
    if (addSentStart && wids[1] != vocab.ssIndex()) {
	wids[0] = vocab.ssIndex();
	start = wids;
    } else {
	start = wids + 1;
    }

    if (addSentEnd && wids[howmany] != vocab.seIndex()) {
	wids[howmany + 1] = vocab.seIndex();
	wids[howmany + 2] = Vocab_None;
    }

    return countSentence(start, factor);
}

/*
 * Incrememt counts indexed by words, starting at node.
 */
template <class CountT>
void
NgramCounts<CountT>::incrementCounts(const VocabIndex *words,
					unsigned minOrder, CountT factor)
{
    NgramNode *node = &counts;

    for (unsigned i = 0; i < order; i++) {
	VocabIndex wid = words[i];

	/*
	 * check of end-of-sentence
	 */
        if (wid == Vocab_None) {
	    break;
	} else {
	    node = node->insertTrie(wid);
	    if (i + 1 >= minOrder) {
		node->value() += factor;
	    }
	}
    }
}

template <class CountT>
unsigned int
NgramCounts<CountT>::countSentence(const VocabIndex *words, CountT factor)
{
    unsigned int start;

    for (start = 0; words[start] != Vocab_None; start++) {
        incrementCounts(words + start, 1, factor);
    }

    /*
     * keep track of word and sentence counts
     */
    stats.numWords += start;
    if (words[0] == vocab.ssIndex()) {
	stats.numWords --;
    }
    if (start > 0 && words[start-1] == vocab.seIndex()) {
	stats.numWords --;
    }

    stats.numSentences ++;

    return start;
}

template <class CountT>
unsigned int
NgramCounts<CountT>::parseNgram(char *line,
		      VocabString *words,
		      unsigned int max,
		      CountT &count)
{
    unsigned howmany = Vocab::parseWords(line, words, max);

    if (howmany == max) {
	return 0;
    }

    /*
     * Parse the last word as a count
     */
    if (!stringToCount(words[howmany - 1], count)) {
	return 0;
    }

    howmany --;
    words[howmany] = 0;

    return howmany;
}

template <class CountT>
unsigned int
NgramCounts<CountT>::readNgram(File &file,
		      VocabString *words,
		      unsigned int max,
		      CountT &count)
{
    char *line;

    /*
     * Read next ngram count from file, skipping blank lines
     */
    line = file.getline();
    if (line == 0) {
	return 0;
    }

    unsigned howmany = parseNgram(line, words, max, count);

    if (howmany == 0) {
	file.position() << "malformed N-gram count or more than " << max - 1 << " words per line\n";
	return 0;
    }

    return howmany;
}

template <class CountT>
Boolean
NgramCounts<CountT>::read(File &file, unsigned int order, Boolean limitVocab)
{
    VocabString words[maxNgramOrder + 1];
    VocabIndex wids[maxNgramOrder + 1];
    CountT count;
    unsigned int howmany;

    /*
     * Check for binary format
     */
    char *firstLine = file.getline();

    if (!firstLine) {
    	return true;
    } else {
	if (strcmp(firstLine, NgramStats_BinaryFormatString) == 0) {
	    File binaryFile(file.name, "rb");
	    return readBinary(binaryFile, order, limitVocab);
	} else {
	    file.ungetline();
	}
    }

    while ((howmany = readNgram(file, words, maxNgramOrder + 1, count))) {
	/*
	 * Skip this entry if the length of the ngram exceeds our 
	 * maximum order
	 */
	if (howmany > order) {
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
	} else if (openVocab) {
	    vocab.addWords(words, wids, maxNgramOrder);
	} else {
	    vocab.getIndices(words, wids, maxNgramOrder, vocab.unkIndex());
	}

	/*
	 *  Update the count
	 */
        CountT *cnt = intersect ?
			counts.find(wids) :
			counts.insert(wids);
	
	if (cnt) {
	    *cnt += count;
	}
    }
    /*
     * XXX: always return true for now, should return false if there was
     * a format error in the input file.
     */
    return true;
}

/*
 * Binary count format:

 	magic string \n
	"maxorder" N \n
	index1 word1 \n
	index2 word2 \n
	...
	indexN wordN \n
	"." \n
	binary-count-trie
 */
template <class CountT>
Boolean
NgramCounts<CountT>::readBinary(File &file, unsigned order, Boolean limitVocab)
{
    char *line = file.getline();

    if (!line || strcmp(line, NgramStats_BinaryFormatString) != 0) {
	file.position() << "bad binary format\n";
	return false;
    }

    /*
     * Maximal count order
     */
    line = file.getline();
    unsigned maxOrder;
    if (sscanf(line, "maxorder %u", &maxOrder) != 1) {
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

    /* 
     * Count data
     */
    return readBinaryNode(counts, order, maxOrder, file, offset, limitVocab, vocabMap);
}

/*
 * Binary count-trie format:

	length-of-binary-trie (unsigned long long)
	word1
	count1
	[subtrie1]
	word2
	count2
	[subtrie2]
	...
 */
template <class CountT>
Boolean
NgramCounts<CountT>::readBinaryNode(NgramNode &node,
				    unsigned order, unsigned maxOrder,
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
	    while (offset < endOffset) {
		VocabIndex oldWid;

		nbytes = readBinaryCount(file, oldWid);
		if (!nbytes) {
		    return false;
		}
		offset += nbytes;

		if (oldWid >= vocabMap.size()) {
		    file.offset() << "word index " << oldWid
		                  << " out of range\n";
		    return false;
		}
		VocabIndex wid = vocabMap[oldWid];
		NgramNode *child = 0;

		if (wid != Vocab_None) {
		    child = intersect ?
				node.findTrie(wid) :
				node.insertTrie(wid);
		}

		if (child == 0) {
		    // skip count value and subtrie
		    CountT dummy;
		    nbytes = readBinaryCount(file, dummy);
		    if (!nbytes) {
		    	return false;
		    }
		    offset += nbytes;

		    if (!readBinaryNode(node, 0, maxOrder-1, file, offset,
							limitVocab, vocabMap)) {
			return false;
		    }
		} else {
		    // read count value and subtrie
		    CountT count;
		    nbytes = readBinaryCount(file, count);
		    if (!nbytes) {
			return false;
		    }
		    child->value() += count;
		    offset += nbytes;

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
 * Read counts stored in an indexed directory structure as proposed by 
 * Thorsten Brandts at Google.  From the documentation:
 *
 * a) top-level directory
 *    doc: documentation
 *    data: data [this is the directory passed as argument]
 *    (the top-level structure is required by LDC)
 * b) data directory
 *    one sub-directory per n-gram order: 1gms, 2gms, 3gms, 4gms, 5gms
 *    (separating the orders makes it easier for people to use smaller orders)
 * c) contents of sub-directory 1gms
 *    - file 'vocab.gz' contains the vocabulary sorted by word in unix
 *      sort-order. Each word is on its own line:
 *      WORD <tab> COUNT
 *    - file 'vocab_cs.gz' contains the same data as 'vocab.gz' but
 *      sorted by count.
 *    (need to be 8+3 file names)
 * d) contents of sub-directories 2gms, 3gms, 4gms, 5gms:
 *    - files 'Ngm-KKKK.gz' where N is the order of the n-grams
 *      and KKKK is the zero-padded number of the file. Each file contains
 *      10 million n-gram entries. N-grams are unix-sorted. Each
 *      n-gram occupies one line:
 *      WORD1 <space> WORD2 <space> ... WORDN <tab> COUNT
 *    - file 'Ngm.idx' where N is the order of the n-grams, with one line for
 *      each n-gram file:
 *      FILENAME <tab> FIRST_NGRAM_IN_FILE
 */
template <class CountT>
Boolean
NgramCounts<CountT>::readGoogle(const char *dir, unsigned order,
							Boolean limitVocab)
{
    makeArray(char, filename, strlen(dir) + 20);

    {
	sprintf(filename, "%s/1gms/vocab%s", dir, GZIP_SUFFIX);

	File countFile(filename, "r", 0);

	if (countFile.error() && sizeof(GZIP_SUFFIX) > 1) {
	    // also try uncompressed vocab file
	    sprintf(filename, "%s/1gms/vocab", dir);

	    countFile.reopen(filename, "r");
	}

	if (countFile.error()) {
	    perror(filename);
	    return false;
	}

	if (debug(DEBUG_READ_GOOGLE)) {
	    dout() << "reading " << filename << endl;
	}

	if (!read(countFile, 1, limitVocab)) {
	    return false;
	}
    }

    for (unsigned i = 2; i <= order; i ++) {

	/*
	 * Read index file
	 */

	sprintf(filename, "%s/%dgms/%dgm.idx", dir, i, i);
    	File indexFile(filename, "r", 0);
	if (indexFile.error()) {
	    perror(filename);
	    return false;
	}

	unsigned numEntries = 0;
	Array<VocabString *> indexEntries;

	while (char *line = indexFile.getline()) {
	    char *savedLine = strdup(line);
	    assert(savedLine != 0);

	    VocabString *indexEntry = new VocabString[i + 3];
	    assert(indexEntry != 0);

	    indexEntry[0] = savedLine;
	    unsigned howmany =
	    		Vocab::parseWords(savedLine, &indexEntry[1], i + 2);

	    if (howmany != i + 1) {
		indexFile.position() << "malformed index entry\n";
		return false;
	    }
	    indexEntries[numEntries++] = indexEntry;
	}

	if (indexFile.error()) {
	    perror(filename);
	    return false;
	}

	/*
	 * Read N-gram counts selectively
	 */

	for (unsigned k = 0; k < numEntries; k ++) {
	    // @kw false positive: SV.FORMAT_STR.PRINT_FORMAT_MISMATCH.BAD
	    sprintf(filename, "%s/%ugms/%s", dir, i, indexEntries[k][1]);

	    File countFile(filename, "r", 0);
	    if (countFile.error()) {
		perror(filename);
		return false;
	    }

	    /*
	     * If limitVocab is in effect check if any in-vocabulary ngrams
	     * are in this count file
	     */
	    if (!limitVocab ||
	        vocab.ngramsInRange(&indexEntries[k][2],
			k == numEntries - 1 ? 0 : &indexEntries[k+1][2]))
	    {
		if (debug(DEBUG_READ_GOOGLE)) {
		    dout() << "reading " << filename << endl;
		}

		if (!read(countFile, i, limitVocab)) {
		    return false;
		}
	    }

	    if (countFile.error()) {
		perror(filename);
		return false;
	    }

	    free((char *)indexEntries[k][0]);
	    delete [] indexEntries[k];
	}
    }

    return true;
}

/*
 * Helper function to record a set of meta-counts at a given prefix into
 * the ngram tree
 */
template <class CountT>
void
NgramCounts<CountT>::addCounts(const VocabIndex *prefix,
			       const LHash<VocabIndex, CountT> &set)
{
    NgramNode *node = intersect ?
			counts.findTrie(prefix) :
			counts.insertTrie(prefix);

    if (node) {
	LHashIter<VocabIndex, CountT> setIter(set);
	VocabIndex word;
	CountT *count;
	while ((count = setIter.next(word))) {
	    *node->insert(word) += *count;
	}
    }
}

/*
 * Read a counts file discarding counts below given minimum counts
 * Assumes that counts are merged, i.e., ngram order is generated by a
 * pre-order traversal of the ngram tree.
 */
template <class CountT>
Boolean
NgramCounts<CountT>::readMinCounts(File &file, unsigned order,
				   Count *minCounts, Boolean limitVocab)
{
    /*
     * Check for binary format
     */
    char *firstLine = file.getline();

    if (!firstLine) {
    	return true;
    } else {
	if (strcmp(firstLine, NgramStats_BinaryFormatString) == 0) {
	    cerr << "binary format not yet support in readMinCounts\n";
	    return false;
	} else {
	    file.ungetline();
	}
    }

    VocabString words[maxNgramOrder + 1];
    VocabIndex prefix[maxNgramOrder + 1];

    /*
     * Data for tracking deletable prefixes
     */
    LHash<VocabIndex, CountT> *metaCounts;
    
    metaCounts = new LHash<VocabIndex, CountT>[order];
    assert(metaCounts != 0);

    makeArray(Boolean, haveCounts, order);
    makeArray(VocabIndex *, lastPrefix, order);
    for (unsigned i = 0; i < order; i ++) {
	lastPrefix[i] = new VocabIndex[order + 1];
	assert(lastPrefix[i] != 0);
	lastPrefix[i][0] = Vocab_None;
	haveCounts[i] = false;
    }

    Vocab::setCompareVocab(0);		// do comparison by word index

    CountT count;
    unsigned int howmany;

    while ((howmany = readNgram(file, words, maxNgramOrder + 1, count))) {
	/*
	 * Skip this entry if the length of the ngram exceeds our 
	 * maximum order
	 */
	if (howmany > order) {
	    continue;
	}

	VocabIndex metaTag = Vocab_None;
	Boolean haveRealCount = false;

	/*
	 * Discard counts below mincount threshold
	 */
	if (count < minCounts[howmany - 1]) {
	    if ((unsigned)count == 0) {
		continue;
	    } else {
		/*
		 * See if metatag is defined, and if so, keep going
		 */
	    	metaTag = vocab.metaTagOfType((unsigned)count);

		if (metaTag == Vocab_None) {
		    continue;
		}
	    }
	}

	/* 
	 * Map words to indices (includes both N-gram prefix and last word)
	 */
	if (limitVocab) {
	    /*
	     * skip ngram if not in-vocabulary
	     */
	    if (!vocab.checkWords(words, prefix, maxNgramOrder)) {
	    	continue;
	    }
	} else if (openVocab) {
	    vocab.addWords(words, prefix, maxNgramOrder);
	} else {
	    vocab.getIndices(words, prefix, maxNgramOrder, vocab.unkIndex());
	}

	/* 
	 * Truncate Ngram for prefix comparison
	 */
	VocabIndex lastWord = prefix[howmany - 1];
	prefix[howmany - 1] = Vocab_None;

	/*
	 * If current ngram prefix differs from previous one assume that 
	 * we are done with all ngrams of that prefix (this assumption is 
	 * true for counts in prefix order).
	 * In this case, output saved meta-counts for the previous prefix,
	 * but only if there also were some real counts -- this saves a lot
	 * of space in the counts tree.
	 */
	if (Vocab::compare(prefix, lastPrefix[howmany-1]) != 0) {
	    if (haveCounts[howmany-1]) {
		addCounts(lastPrefix[howmany-1], metaCounts[howmany-1]);
		haveCounts[howmany-1] = false;
	    }
	    metaCounts[howmany-1].clear();
	    Vocab::copy(lastPrefix[howmany-1], prefix);
	}

	/*
	 *  Update the count (either meta or real)
	 */
	if (metaTag != Vocab_None) {
	    *metaCounts[howmany-1].insert(metaTag) += 1;
	} else {
	    CountT *cnt = intersect ?
				findCount(prefix, lastWord) :
				insertCount(prefix, lastWord);
	    if (cnt) {
		*cnt += count;
	    }
	    haveCounts[howmany-1] = true;
	}
    }

    /*
     * Record final saved meta-counts and deallocate prefix buffers
     */
    for (unsigned i = order; i > 0; i --) {
	if (haveCounts[i-1]) {
	    addCounts(lastPrefix[i-1], metaCounts[i-1]);
	}
	delete [] lastPrefix[i-1];
    }

    delete [] metaCounts;

    /*
     * XXX: always return true for now, should return false if there was
     * a format error in the input file.
     */
    return true;
}


template <class CountT>
unsigned int
NgramCounts<CountT>::writeNgram(File &file,
				const VocabString *words,
				CountT count)
{
    unsigned i = 0;

    if (words[0]) {
	file.fprintf("%s", words[0]);
	for (i = 1; words[i]; i++) {
	    file.fprintf(" %s", words[i]);
	}
    }
    file.fprintf("\t%s\n", countToString(count));

    return i;
}

/*
 * For reasons of efficiency the write() method doesn't use
 * writeNgram()  (yet).  Instead, it fills up a string buffer 
 * as it descends the tree recursively.  this avoid having to
 * lookup shared prefix words and buffer the corresponding strings
 * repeatedly.
 */
template <class CountT>
void
NgramCounts<CountT>::writeNode(
    NgramNode &node,		/* the trie node we're at */
    File &file,			/* output file */
    char *buffer,		/* output buffer */
    char *bptr,			/* pointer into output buffer */
    unsigned int level,		/* current trie level */
    unsigned int order,		/* target trie level */
    Boolean sorted)		/* produce sorted output */
{
    NgramNode *child;
    VocabIndex wid;

    TrieIter<VocabIndex,CountT> iter(node, sorted ? vocab.compareIndex() : 0);

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
	 * If this is the final level, print out the ngram and the count.
	 * Otherwise set up another level of recursion.
	 */
	if (order == 0 || level == order) {
	   file.fprintf("%s\t%s\n", buffer, countToString(child->value()));
	} 
	
	if (order == 0 || level < order) {
	   *(bptr + wordLen) = ' ';
	   writeNode(*child, file, buffer, bptr + wordLen + 1, level + 1,
			order, sorted);
	}
    }
}

template <class CountT>
void
NgramCounts<CountT>::write(File &file, unsigned int order, Boolean sorted)
{
    char *buffer = TLSW_GET_ARRAY(writeBufferTLS);
    writeNode(counts, file, buffer, buffer, 1, order, sorted);
}

template <class CountT>
Boolean
NgramCounts<CountT>::writeBinary(File &file, unsigned int order)
{
    /*
     * Magic string
     */
    file.fprintf("%s", NgramStats_BinaryFormatString);

    /*
     * Maximal count order
     */
    file.fprintf("maxorder %u\n", order > 0 ? order : this->order);

    /*
     * Vocabulary index
     */
    vocab.writeIndexMap(file);

    long long offset = file.ftell();

    // detect if file is not seekable
    if (offset < 0) {
	file.position() << srilm_ts_strerror(errno) << endl;
	return false;
    }

    /*
     * Count data 
     */
    return writeBinaryNode(counts, 1, order, file, offset);
}

template <class CountT>
Boolean
NgramCounts<CountT>::writeBinaryNode(NgramNode &node,
					unsigned level, unsigned order,
					File &file, long long &offset)
{
    unsigned effectiveOrder = order > 0 ? order : this->order;

    if (level > effectiveOrder) {
    	// when reaching the maximal order don't write an offset to save space
    	return true;
    } else {
	// guess number of bytes needed for storing subtrie rooted at node
	// based on its depth (if we guess wrong we need to redo the whole
	// subtrie later)
	unsigned subtrieDepth = effectiveOrder - level;
        unsigned offsetBytes = subtrieDepth == 0 ? 2 :
				subtrieDepth <= 3 ? 4 : 8;

	long long startOffset = offset;	// remember start offset

retry:
	// write placeholder value
	unsigned nbytes = writeBinaryCount(file, (unsigned long long)0,
								offsetBytes);
	if (!nbytes) return false;
	offset += nbytes;

	if (order == 0 || level <= order) {
	    NgramNode *child;
	    // write subtries -- always in index-sorted order to ensure fast
	    // reading regardless of data structure used
#ifdef USE_SARRAY_TRIE
	    TrieIter<VocabIndex,CountT> iter(node);
#else
	    TrieIter<VocabIndex,CountT> iter(node, SArray_compareKey);
#endif
	    VocabIndex wid;

	    while ((child = iter.next(wid))) {
		nbytes = writeBinaryCount(file, wid);
		if (!nbytes) return false;
		offset += nbytes;

		if (order > 0 && level < order) {
		    nbytes = writeBinaryCount(file, (CountT)0);
		} else {
		    nbytes = writeBinaryCount(file, child->value());
		}
		if (!nbytes) return false;
		offset += nbytes;

		if (!writeBinaryNode(*child, level + 1, order, file, offset)) {
		    return false;
		}
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
			  << " (order " << effectiveOrder << ","
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
}

template <class CountT>
CountT
NgramCounts<CountT>::sumNode(NgramNode &node, unsigned level, unsigned order)
{
    /*
     * For leaf nodes, or nodes beyond the maximum level we are summing,
     * return their count, leaving it unchanged.
     * For nodes closer to the root, replace their counts with the
     * the sum of the counts of all the children.
     */
    if (level > order || node.numEntries() == 0) {
	return node.value();
    } else {
	NgramNode *child;
	TrieIter<VocabIndex,CountT> iter(node);
	VocabIndex wid;

	CountT sum = 0;

	while ((child = iter.next(wid))) {
	    sum += sumNode(*child, level + 1, order);
	}

	node.value() = sum;

	return sum;
    }
}

template <class CountT>
CountT
NgramCounts<CountT>::sumCounts(unsigned int order)
{
    return sumNode(counts, 1, order);
}

/*
 * Prune ngram counts
 */
template <class CountT>
unsigned
NgramCounts<CountT>::pruneCounts(CountT minCount)
{
    unsigned npruned = 0;
    makeArray(VocabIndex, ngram, order + 1);

    for (unsigned i = 1; i <= order; i++) {
	CountT *count;
	NgramCountsIter<CountT> countIter(*this, ngram, i);

	/*
	 * This enumerates all ngrams
	 */
	while ((count = countIter.next())) {
	    if (*count < minCount) {
		removeCount(ngram);
		npruned ++;
	    }
	}
    }
    return npruned;
}

/*
 * Set ngram counts to constant value (default zero)
 */
template <class CountT>
void
NgramCounts<CountT>::setCounts(CountT value)
{
    makeArray(VocabIndex, ngram, order + 1);

    for (unsigned i = 1; i <= order; i++) {
	CountT *count;
	NgramCountsIter<CountT> countIter(*this, ngram, i);

	/*
	 * This enumerates all ngrams
	 */
	while ((count = countIter.next())) {
	    *count = value;
	}
    }
}

#endif /* _NgramStats_cc_ */
