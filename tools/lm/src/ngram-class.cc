/*
 * ngram-class --
 *	Induce class ngram models from counts
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1999-2010 SRI International, 2012-2016 Microsoft Corp., 2013-2014 Seppo Enarvi. All Rights Reserved.";
static char RcsId[] = "@(#)$Id: ngram-class.cc,v 1.43 2016/04/09 06:53:01 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <locale.h>
#include <assert.h>

#include "option.h"
#include "version.h"
#include "File.h"
#include "Debug.h"
#include "Prob.h"
#include "Vocab.h"
#include "SubVocab.h"
#include "TextStats.h"
#include "NgramStats.h"
#include "LHash.cc"
#include "Map2.cc"
#include "Array.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_MAP1(VocabIndex,VocabIndex,LogP);

#ifdef USE_SARRAY
// XXX: avoid multiple definitions with NgramLM
INSTANTIATE_LHASH(VocabIndex,LogP);
#endif

#endif

#define DEBUG_TEXTSTATS		1
#define DEBUG_TRACE_MERGE	2
#define DEBUG_PRINT_CONTRIBS	3	// in interactive mode

static int version = 0;
static char *vocabFile = 0;
static int toLower = 0;
static char *noclassFile = 0;
static char *countsFile = 0;
static char *textFile = 0;
static char *classesFile = 0;
static char *classCountsFile = 0;
static unsigned numClasses = 1;
static int fullMerge = 0;
static int interact = 0;
static int debug = 0;
static char *readClassesFile = 0;
static int saveFreq = 0;
static unsigned saveMaxClasses = 0;
static unsigned maxWordsPerClass = 0;

static Option options[] = {
    { OPT_TRUE, "version", &version, "print version information" },
    { OPT_UINT, "debug", &debug, "debugging level" },
    { OPT_STRING, "vocab", &vocabFile, "vocab file" },
    { OPT_TRUE, "tolower", &toLower, "map vocabulary to lowercase" },
    { OPT_STRING, "noclass-vocab", &noclassFile, "vocabulary not to be classed" },
    { OPT_STRING, "counts", &countsFile, "counts file to read" },
    { OPT_STRING, "text", &textFile, "text file to count" },
    { OPT_UINT, "numclasses", &numClasses, "number of classes to induce" },
    { OPT_TRUE, "full", &fullMerge, "perform full greedy merging" },
    { OPT_FALSE, "incremental", &fullMerge, "perform incremental greedy merging" },
    { OPT_TRUE, "interact", &interact, "perform interactive merging" },
    { OPT_STRING, "class-counts", &classCountsFile, "class N-gram count output" },
    { OPT_STRING, "classes", &classesFile, "class definitions output" },
    { OPT_STRING, "read", &readClassesFile, "read initial class definitions from this file" },
    { OPT_INT, "save", &saveFreq, "save classes/counts every this many iterations" },
    { OPT_UINT, "save-maxclasses", &saveMaxClasses, "maximum number of intermediate classes to save" },
    { OPT_UINT, "maxwordsperclass", &maxWordsPerClass, "maximum number of words per class" },
};

/*
 * Compute n \log n correctly even if n = 0.
 */
static inline LogP
NlogN(NgramCount count)
{
    if (count == 0) {
	return LogP_One;
    } else {
	return count * ProbToLogP(count);
    }
}

/*
 * Many-to-one class-to-word mapping
 */
class UniqueWordClasses: public Debug
{
public:
    UniqueWordClasses(Vocab &vocab, SubVocab &classVocab);
    virtual ~UniqueWordClasses() {};

    VocabIndex newClass();		// create a new class
    void initialize(NgramStats &counts, SubVocab &noclassVocab);
					// initialize classes from word counts
    void merge(VocabIndex c1, VocabIndex c2);	// merge classes
    LogP bestMerge(Vocab &mergeSet, VocabIndex &c1, VocabIndex &c2);
						// single best merge step
    void fullMerge(unsigned numClases);		// full greedy merging
    void incrementalMerge(unsigned numClases);	// incremental merging

    Boolean readClasses(File &file);	// read class definitions
    void writeClasses(File &file);	// write class definitions
    void writeCounts(File &file)	// write class ngram counts
	{ classNgramCounts.write(file, 0, true); };

    NgramCount getCount(VocabIndex c1)		// class unigram count
	{ return getCount(c1, Vocab_None); };		
    inline NgramCount getCount(VocabIndex c1, VocabIndex c2);
						// class bigram count
    inline NgramCount getCountR(VocabIndex c1, VocabIndex c2);
						// reverse bigram count

    LogP totalLogP();			// total log likelihood
    LogP diffLogP(VocabIndex c1, VocabIndex c2);
					// log likelihood difference for 
					// class merging

    void computeClassContribs();	// recompute classContribs vector
    void computeMergeContribs();	// recompute mergeContribs matrix

    void getStats(TextStats &stats);

    void writeContribs(File &file);	// dump contrib vector

    Vocab &vocab;
    Vocab &classVocab;

protected:
    LHash<VocabIndex,VocabIndex> wordToClass;	// word->class map
    LHash<VocabIndex,NgramCount> wordCounts;	// word counts
    NgramStats classNgramCounts;		// trie of (C,...,C) counts
    NgramStats classNgramCountsR;		// trie of reversed counts,
    LHash<VocabIndex,LogP> classContribs;	// class contributions to
						// total log likelihood
    Map2<VocabIndex,VocabIndex,LogP> mergeContribs;
						// merge-pair contributions
						// to delta log likelihood
    void computeMergeContrib(VocabIndex c1);	// recompute mergeContribs
    LogP computeClassContrib(VocabIndex c);	// recompute classContribs
    LogP computeMergeContrib(VocabIndex c1, VocabIndex c2);	// same

    void mergeCounts(NgramStats &counts, VocabIndex c1, VocabIndex c2);
    
    unsigned genSymCount;
};

UniqueWordClasses::UniqueWordClasses(Vocab &vocab, SubVocab &classVocab)
    : vocab(vocab), classVocab(classVocab),
      classNgramCounts(vocab, 2), classNgramCountsR(vocab, 2),
      genSymCount(0)
      
{
    /*
     * Make sure the classes are subset of base vocabulary
     */
    assert(&vocab == &classVocab.baseVocab());
};

VocabIndex
UniqueWordClasses::newClass()
{
    char className[30];

    do {
	sprintf(className, "CLASS-%05u", ++genSymCount);
    } while (vocab.getIndex(className) != Vocab_None);

    return classVocab.addWord(className);
}

void
UniqueWordClasses::initialize(NgramStats &counts, SubVocab &noclassVocab)
{
    VocabIndex ngram[3];

    /*
     * Make sure the noclassVocab is subset of base vocabulary
     */
    assert(&vocab == &noclassVocab.baseVocab());

    /*
     * Enumerate unigrams --
     *	use sorted iteration to generate deterministic results
     */
    NgramsIter iter1(counts, ngram, 1, vocab.compareIndex());
    NgramCount *count;

    while ((count = iter1.next())) {
	VocabIndex classUnigram[2];
	
	if ((noclassVocab.getWord(ngram[0]) != 0) &&
	    (wordToClass.find(ngram[0]) == 0)) {
	    /*
	     * A word that is not supposed to be classed, and is not yet classed
	     * (might happen that we read a class definition where the word has
	     * been classed anyway).
	     */
	    classUnigram[0] = ngram[0];
	} else {
	    /* 
	     * A word that is supposed to be classed: find or create class for
	     * it.
	     */
	    Boolean found;
	    VocabIndex *class1 = wordToClass.insert(ngram[0], found);
	    if (!found) {
		*class1 = newClass();

		if (debug(DEBUG_TRACE_MERGE)) {
		    dout() << "\tcreating " << classVocab.getWord(*class1)
			   << " for word " << vocab.getWord(ngram[0])
			   << endl;
		}
	    }
	    classUnigram[0] = *class1;
	}
	classUnigram[1] = Vocab_None;

	*classNgramCounts.insertCount(classUnigram) += *count;
	*classNgramCountsR.insertCount(classUnigram) += *count;

	*wordCounts.insert(ngram[0]) += *count;
    }

    /*
     * Enumerate bigrams
     */
    NgramsIter iter2(counts, ngram, 2);

    while ((count = iter2.next())) {
	VocabIndex classBigram[3];

	if ((noclassVocab.getWord(ngram[0]) != 0) &&
	    (wordToClass.find(ngram[0]) == 0)) {
	    classBigram[0] = ngram[0];
	} else {
	    VocabIndex *class1 = wordToClass.find(ngram[0]);
	    if (class1 == 0) {
		cerr << "word 1 in bigram \"" << (vocab.use(), ngram)
		     << "\" has no unigram count\n";
		exit(1);
	    }
	    classBigram[0] = *class1;
	}

	if (noclassVocab.getWord(ngram[1]) != 0) {
	    classBigram[1] = ngram[1];
	} else {
	    VocabIndex *class2 = wordToClass.find(ngram[1]);
	    if (class2 == 0) {
		cerr << "word 2 in bigram \"" << (vocab.use(), ngram)
		     << "\" has no unigram count\n";
		exit(1);
	    }
	    classBigram[1] = *class2;
	}

	classBigram[2] = Vocab_None;
	*classNgramCounts.insertCount(classBigram) += *count;

	Vocab::reverse(classBigram);
	*classNgramCountsR.insertCount(classBigram) += *count;
    }
}

void
UniqueWordClasses::mergeCounts(NgramStats &counts,
					VocabIndex c1, VocabIndex c2)
{
    VocabIndex unigram[2]; unigram[1] = Vocab_None;
    VocabIndex unigram2[2]; unigram2[1] = Vocab_None;
    VocabIndex bigram[3]; bigram[2] = Vocab_None;
    NgramCount *count;

    /*
     * Update Counts 
     * 1) add row c2 to c1 row
     */
    unigram[0] = c2;
    NgramsIter iter2(counts, unigram, unigram2, 1);
    while ((count = iter2.next())) {
	bigram[0] = c1;
	bigram[1] = unigram2[0];
	*counts.insertCount(bigram) += *count;
    }

    /*
     * 2) Remove row c2
     */
    unigram[0] = c1;
    unigram2[0] = c2;
    NgramCount removed;
    if (counts.removeCount(unigram2, &removed)) {
	*counts.insertCount(unigram) += removed;
    }

    /*
     * 3) add column c2 to column c1 and remove column c2
     */
    NgramsIter iter3(counts, unigram, 1);
    while ((count = iter3.next())) {
	bigram[0] = unigram[0];
	bigram[1] = c2;

	if (counts.removeCount(bigram, &removed)) {
	    bigram[1] = c1;
	    *counts.insertCount(bigram) += removed; 
	}
    }
}

void
UniqueWordClasses::merge(VocabIndex c1, VocabIndex c2)
{
    /*
     * Destructively merge c2 into c1 ...
     */
    assert(c1 != c2);

    /*
     * Make sure both c1 and c2 are classes
     */
    assert(classVocab.getWord(c1) != 0);
    assert(classVocab.getWord(c2) != 0);

    /*
     * Update class membership
     */
    LHashIter<VocabIndex,VocabIndex> iter1(wordToClass);
    VocabIndex *clasz;
    VocabIndex word;

    while ((clasz = iter1.next(word))) {
	if (*clasz == c2) {
	    *clasz = c1;
	}
    }

    /*
     * Update class contribs vector
     */
    classContribs.remove(c1);
    classContribs.remove(c2);
    {
	LHashIter<VocabIndex,LogP> iter(classContribs);
	VocabIndex clasz;
	LogP *logp;

	while ((logp = iter.next(clasz))) {
	    *logp += NlogN(getCount(clasz, c1) + getCount(clasz, c2)) 
		   + NlogN(getCount(c1, clasz) + getCount(c2, clasz))
		   - NlogN(getCount(clasz, c1)) - NlogN(getCount(clasz, c2))
		   - NlogN(getCount(c1, clasz)) - NlogN(getCount(c2, clasz));
	}
    }

    /*
     * Update merge contribs matrix
     */
    mergeContribs.remove(c1);
    mergeContribs.remove(c2);
    {
	Map2Iter<VocabIndex,VocabIndex,LogP> iter1(mergeContribs);
	VocabIndex class1;

	while ((iter1.next(class1))) {
	    mergeContribs.remove(class1, c1);
	    mergeContribs.remove(class1, c2);

	    Map2Iter2<VocabIndex,VocabIndex,LogP> iter2(mergeContribs, class1);
	    VocabIndex class2;
	    LogP *logp;

	    while ((logp = iter2.next(class2))) {
		*logp += NlogN(getCount(class1,c1) + getCount(class2,c1) +
			       getCount(class1,c2) + getCount(class2,c2))
		       + NlogN(getCount(c1,class1) + getCount(c1,class2) +
			       getCount(c2,class1) + getCount(c2,class2))
		       - NlogN(getCount(class1,c1) + getCount(class2,c1))
		       - NlogN(getCount(class1,c2) + getCount(class2,c2))
		       - NlogN(getCount(c1,class1) + getCount(c1,class2))
		       - NlogN(getCount(c2,class1) + getCount(c2,class2));
	    }
	}
    }

    /*
     * Update counts
     */
    mergeCounts(classNgramCounts, c1, c2);
    mergeCounts(classNgramCountsR, c1, c2);

    /*
     * Get rid of old class
     */
    classVocab.remove(c2);

}

Boolean
UniqueWordClasses::readClasses(File &file)
{
    while (char *line = file.getline()) {
	VocabString words[4];
	unsigned howmany = Vocab::parseWords(line, words, 4);
	if (howmany != 3) {
	    file.position() << "malformed class expansion\n";
	    return false;
	}

	VocabString &classString = words[0];
	VocabString &wordString = words[2];
	VocabIndex wordIndex = vocab.getIndex(wordString);
	if (wordIndex == Vocab_None) {
	    if (debug(DEBUG_TRACE_MERGE)) {
		dout() << "\tIgnoring unknown word " << wordString << endl;
	    }
	    continue;
	}

	/*
	 * Check that the second word is an expansion probability.
	 */
	char *endptr;
	Prob prob = strtod(words[1], &endptr);
	if (endptr == words[1]) {
	    file.position() << "malformed class expansion probability\n";
	    return false;
	}

	/*
	 * Create an entry in wordToClass.
	 */
	Boolean found;
	VocabIndex *class1 = wordToClass.insert(wordIndex, found);
	if (found) {
	    file.position() << "only single class expansion per word allowed\n";
	    return false;
	}

	/*
	 * Assign the word to the correct class, creating the class
	 * if necessary.
	 */
	*class1 = classVocab.addWord(classString);
	if (debug(DEBUG_TRACE_MERGE)) {
	    dout() << "\tassigning word " << wordString
	           << " to class " << classString
		   << endl;
	}
    }

    return true;
}

void
UniqueWordClasses::writeClasses(File &file)
{
    /*
     * Sort words by class and compute probabilities
     */
    Map2<VocabIndex,VocabIndex,Prob> classWordProbs;

    LHashIter<VocabIndex,NgramCount> wordIter(wordCounts);
    NgramCount *wordCount;
    VocabIndex word;

    while ((wordCount = wordIter.next(word))) {
	VocabIndex *clasz = wordToClass.find(word);

	/*
	 * Ignore words that are not classed
	 */
	if (clasz == 0) continue;

	/*
	 * get total class count
	 */
	VocabIndex unigram[2];
	unigram[0] = *clasz;
	unigram[1] = Vocab_None;

	NgramCount *classCount = classNgramCounts.findCount(unigram);
	assert(classCount != 0);

	assert(*classCount != 0 || *wordCount == 0);

	Prob prob = (*classCount == 0) ? 0.0 : ((Prob)*wordCount) / *classCount;

	*classWordProbs.insert(*clasz,word) = prob;
    }

    /*
     * Dump class expansion in sorted order
     */
    VocabIter classIter(classVocab, true);
    VocabIndex clasz;

    while (classIter.next(clasz)) {
	Map2Iter2<VocabIndex,VocabIndex,Prob> wordIter(classWordProbs, clasz);
	VocabIndex word;

	Prob *prob;
	while ((prob = wordIter.next(word))) {
	    file.fprintf("%s %.*lg %s\n", classVocab.getWord(clasz),
					 Prob_Precision, (double)*prob,
					 vocab.getWord(word));
	}
    }
}

void
UniqueWordClasses::writeContribs(File &file)
{
    file.fprintf("=== class contribs ===\n");

    LHashIter<VocabIndex,LogP> iter(classContribs);
    VocabIndex clasz;
    LogP *logp;

    while ((logp = iter.next(clasz))) {
	file.fprintf("%s %.*lg\n", vocab.getWord(clasz),
				   LogP_Precision, (double)*logp);
    }

    file.fprintf("=== merge contribs ===\n");

    Map2Iter<VocabIndex,VocabIndex,LogP> iter1(mergeContribs);
    VocabIndex class1;
    while ((iter1.next(class1))) {
	Map2Iter2<VocabIndex,VocabIndex,LogP> iter2(mergeContribs, class1);
	VocabIndex class2;
	while ((logp = iter2.next(class2))) {
	    file.fprintf("%s %s %.*lg\n", vocab.getWord(class1),
					vocab.getWord(class2),
					LogP_Precision, (double)*logp);
	}
    }
    file.fprintf("=== end of contribs ===\n");
}

void
UniqueWordClasses::getStats(TextStats &stats)
{
    LHashIter<VocabIndex,NgramCount> wordIter(wordCounts);
    NgramCount *count;
    VocabIndex word;

    stats.numWords = 0;

    while ((count = wordIter.next(word))) {
	if (word == vocab.seIndex()) {
	    stats.numSentences = *count;
	} else if (word != vocab.ssIndex()) {
	    stats.numWords += *count;
	}
    }

    stats.prob = totalLogP();
    stats.numOOVs = 0;
    stats.zeroProbs = 0;
}

inline NgramCount
UniqueWordClasses::getCount(VocabIndex c1, VocabIndex c2)
{
    VocabIndex bigram[3];
    bigram[0] = c1; bigram[1] = c2; bigram[2] = Vocab_None;

    NgramCount *count = classNgramCounts.findCount(bigram);
    return count ? *count : (NgramCount)0;
}

inline NgramCount
UniqueWordClasses::getCountR(VocabIndex c1, VocabIndex c2)
{
    VocabIndex bigram[3];
    bigram[0] = c1; bigram[1] = c2; bigram[2] = Vocab_None;

    NgramCount *count = classNgramCountsR.findCount(bigram);
    return count ? *count : (NgramCount)0;
}

LogP
UniqueWordClasses::totalLogP()
{
    /*
     * Total log likelihood =
     *	    \sum_{i} n(w_i) \log n(w_i)
     *	    + \sum_{i,j} n(c_i,c_j) \log n(c_i,c_j)
     *      - 2 \sum_{j} n(c_j) \log n(c_j)
     */

    LogP total = LogP_One;

    NgramCount *count;

    /*
     * summation over words
     */
    LHashIter<VocabIndex,NgramCount> wordIter(wordCounts);
    VocabIndex word;

    while ((count = wordIter.next(word))) {
	total += NlogN(*count);
    }

    /*
     * summation over class bigrams
     */
    VocabIndex classNgram[3];
    NgramsIter iter2(classNgramCounts, classNgram, 2);

    while ((count = iter2.next())) {
	total += NlogN(*count);
    }

    /*
     * summation over class unigrams
     */
    NgramsIter iter1(classNgramCounts, classNgram, 1);

    while ((count = iter1.next())) {
	total -= 2.0 * NlogN(*count);
    }

    return total;
}

LogP
UniqueWordClasses::diffLogP(VocabIndex c1, VocabIndex c2)
{
    assert(c1 != c2);

    return computeMergeContrib(c1, c2)
	   - computeClassContrib(c1)
	   - computeClassContrib(c2);
}

/*
 * Compute the contribution of each class in an auxiliary array
 * (a la the s_k(i) in Brown et al 1992).
 *
 * classContrib(i) = \sum_j n(c_i,c_j) \log n(c_i,c_j)
 *              + \sum_j n(c_j,c_i) \log n(c_j,c_i)
 *		- n(c_i,c_i) \log n(c_i,c_i)
 */
void
UniqueWordClasses::computeClassContribs()
{
    if (debug(DEBUG_TRACE_MERGE)) {
	dout() << "computing class contrib vector\n";
    }

    /*
     * Clear the classContribs array
     */
    classContribs.clear();

    /*
     * Recompute in a single pass over all counts
     */
    VocabIndex bigram[3];
    NgramsIter iter(classNgramCounts, bigram, 2);
    NgramCount *count;

    while ((count = iter.next())) {
	*classContribs.insert(bigram[0]) += NlogN(*count);
	if (bigram[0] != bigram[1]) {
	    *classContribs.insert(bigram[1]) += NlogN(*count);
	}
    }
}

LogP
UniqueWordClasses::computeClassContrib(VocabIndex c)
{
    Boolean found;
    LogP *logp = classContribs.insert(c, found);

    if (found) {
	return *logp;
    }

    LogP total = LogP_One;

    VocabIndex class1[2];
    class1[0] = c;
    class1[1] = Vocab_None;
    VocabIndex class2[2];

    NgramCount *count;
    NgramsIter iter1(classNgramCounts, class1, class2, 1);

    while ((count = iter1.next())) {
	total += NlogN(*count);
    }

    NgramsIter iter2(classNgramCountsR, class1, class2, 1);

    while ((count = iter2.next())) {
	if (class2[0] != c) {
	    total += NlogN(*count);
	}
    }

    /*
     * cache result
     */
    *logp = total;

    return total;
}

/*
 * Compute the contribution of a merge pair to log likelihood difference
 * in an auxiliary array
 * mergeContrib(c1, c2) =
 *	n(c_12,c_12) \log n(c_12,c_12)
 *    +	\sum_{i \neq 1,2}  n(c_i, c_12) \log n(c_i, c_12)
 *    + \sum_{j \neq 1,2}  n(c_12, c_j) \log n(c_12, c_j)
 *    + n(c_1, c_2) \log n(c_1, c_2) + n(c_2, c_1) \log n(c_2, c_1)
 *    - 2 n(c_12) \log n(c_12)
 */
void
UniqueWordClasses::computeMergeContribs()
{
    if (debug(DEBUG_TRACE_MERGE)) {
	dout() << "computing merge contrib matrix\n";
    }

    VocabIter iter1(classVocab);
    VocabIndex class1;

    while ((iter1.next(class1))) {
	VocabIter iter2(classVocab);
	VocabIndex class2;

	while ((iter2.next(class2))) {
	    if (class1 < class2) {
		(void)computeMergeContrib(class1, class2);
	    }
	}
    }
}

void
UniqueWordClasses::computeMergeContrib(VocabIndex c1)
{
    VocabIter iter(classVocab);
    VocabIndex c2;

    while ((iter.next(c2))) {
	if (c1 != c2) {
	    (void)computeMergeContrib(c1, c2);
	}
    }
}

LogP
UniqueWordClasses::computeMergeContrib(VocabIndex c1, VocabIndex c2)
{
    assert(c1 != c2);

    /*
     * For efficiency we only store and compute for c1 < c2
     */
    if (c1 > c2) {
	VocabIndex tmp = c2;
	c2 = c1;
	c1 = tmp;
    }

    Boolean found;
    LogP *logp = mergeContribs.insert(c1, c2, found);
    if (found) {
	return *logp;
    }

    VocabIndex unigram[2]; unigram[1] = Vocab_None;
    VocabIndex bigram[3]; bigram[2] = Vocab_None;
    NgramCount *count;

    LogP total = LogP_One;

    /*
     * n(c_12,c_12) \log n(c_12,c_12)
     */
    total += NlogN(getCount(c1, c1) + getCount(c1, c2) +
			getCount(c2, c1) + getCount(c2, c2));

    /*
     * + \sum_{i \neq 1,2}  n(c_i, c_12) \log n(c_i, c_12)
     */
    unigram[0] = c1;
    NgramsIter iter0(classNgramCountsR, unigram, &bigram[1], 1);
    while ((count = iter0.next())) {
	if (bigram[1] != c1 && bigram[1] != c2) {
	    total += NlogN(*count + getCountR(c2, bigram[1]));
	}
    }

    unigram[0] = c2;
    NgramsIter iter1(classNgramCountsR, unigram, &bigram[1], 1);
    while ((count = iter1.next())) {
	if (bigram[1] != c1 && bigram[1] != c2 &&
	    getCountR(c1, bigram[1]) == 0)
	{
	    total += NlogN(*count);
	}
    }

    /*
     * + \sum_{j \neq 1,2}  n(c_12, c_j) \log n(c_12, c_j)
     */
    unigram[0] = c1;
    NgramsIter iter2(classNgramCounts, unigram, &bigram[1], 1);
    while ((count = iter2.next())) {
	if (bigram[1] != c1 && bigram[1] != c2) {
	    total += NlogN(*count + getCount(c2, bigram[1]));
	}
    }

    unigram[0] = c2;
    NgramsIter iter3(classNgramCounts, unigram, &bigram[1], 1);
    while ((count = iter3.next())) {
	if (bigram[1] != c1 && bigram[1] != c2 &&
	    getCount(c1, bigram[1]) == 0)
	{
	    total += NlogN(*count);
	}
    }

    /*
     * + n(c_1, c_2) \log n(c_1, c_2) + n(c_2, c_1) \log n(c_2, c_1)
     */
    total += NlogN(getCount(c1, c2)) + NlogN(getCount(c2, c1));

    /*
     * - 2 n(c_12) \log n(c_12)
     */
    NgramCount n1 = getCount(c1);
    NgramCount n2 = getCount(c2);

    total -= 2 * NlogN(n1 + n2);

    /*
     * + 2 [ n(c_1) \log n(c1) + n(c_2) \log n(c_2)
     */
    total += 2 * NlogN(n1) + 2 * NlogN(n2);

    /*
     * Cache result
     */
    *logp = total;

    return total;
}

/*
 * Find and perform best merge pair
 */
LogP
UniqueWordClasses::bestMerge(Vocab &mergeSet, VocabIndex &b1, VocabIndex &b2)
{
    VocabIndex bestC1 = Vocab_None, bestC2 = Vocab_None;
    LogP bestDiff = 0.0;

    // use sorted iteration to generate deterministic results
    VocabIter iter1(mergeSet, true);
    VocabIndex c1;

    while (iter1.next(c1)) {
	VocabIter iter2(iter1);
	VocabIndex c2;

	while (iter2.next(c2)) {
	    LogP diff = diffLogP(c1, c2);

	    if (bestC1 == Vocab_None || diff > bestDiff) {
		bestC1 = c1;
		bestC2 = c2;
		bestDiff = diff;
	    }
	}
    }
    if (debug(DEBUG_TRACE_MERGE)) {
	dout() << "\tmerging " << mergeSet.getWord(bestC1)
	       << " and " << mergeSet.getWord(bestC2)
	       << " diff = " << bestDiff 
	       << endl;
    }

    merge(bestC1, bestC2);

    b1 = bestC1;
    b2 = bestC2;

    return bestDiff;
}

/*
 * Create a writable file if basename is defined and 
 * iter is a multiple of freq.  Used in saving preliminary results
 * during merging.
 */
static File *
logFile(const char *basename, unsigned freq, int iter)
{
    if (freq > 0 && iter >= 0 && basename != 0 && iter % freq == 0) {
	makeArray(char, filename, strlen(basename) + 10);

	if (stdio_filename_p(basename)) {
	    printf("*** SAVE FOR ITERATION %06d ***\n", iter);
	    fflush(stdout);
	    strcpy(filename, basename);
	} else {
	    sprintf(filename, "%s.%06d%s", basename, iter,
			    compressed_filename_p(basename) ? COMPRESS_SUFFIX :
			      gzipped_filename_p(basename) ? GZIP_SUFFIX : "");
	}

	File *file = new File(filename, "w");
	assert(file != 0);

	return file;
    } else {
	return 0;
    }
}

/*
 * Full greedy class merging
 *	This is the first, O(V^3) algorithm in Brown et al. (1992)
 */
void
UniqueWordClasses::fullMerge(unsigned numClasses)
{
    if (numClasses < 1) {
	numClasses = 1;
    }

    TextStats stats;
    getStats(stats);

    unsigned numTokens = (unsigned)(stats.numWords + stats.numSentences);

    unsigned iters = 0;
    int saveIters = -1;

    if (debug(DEBUG_TRACE_MERGE)) {
	dout() << "iter " << iters
	       << ": " << classVocab.numWords() << " classes, "
	       << "perplexity = " << LogPtoPPL(stats.prob/numTokens)
	       << endl;
    }

    while (classVocab.numWords() > numClasses) {
	iters ++;
	VocabIndex b1, b2;

	bestMerge(classVocab, b1, b2);

	if (debug(DEBUG_TRACE_MERGE)) {
	    dout() << "iter " << iters
		   << ": " << classVocab.numWords() << " classes, "
		   << "perplexity = " << LogPtoPPL(totalLogP()/numTokens)
		   << endl;
	}

	/*
	 * Save classes and counts if and when requested
	 */
	if (saveMaxClasses == 0) {
	    saveIters = iters;
	} else if (saveIters < 0) {
	    if (classVocab.numWords() <= saveMaxClasses) {
		saveIters = 0;		// start saving now
	    }
	} else {
	    saveIters ++;
	}

	File *cf = logFile(classesFile, saveFreq, saveIters);
	if (cf != 0) {
	    writeClasses(*cf);
	    delete cf;
	}

	cf = logFile(classCountsFile, saveFreq, saveIters);
	if (cf != 0) {
	    writeCounts(*cf);
	    delete cf;
	}
    }
}

/*
 * Order classes by count
 *	(break ties according to vocab index, to give strict ordering)
 */
static UniqueWordClasses *orderClasses;

static int
orderByCount(VocabIndex c1, VocabIndex c2)
{
    NgramCount diff = orderClasses->getCount(c2) - orderClasses->getCount(c1);

    if (diff != 0) {
	return (int)diff;
    } else {
	return (int)(c2 - c1);
    }
}

/*
 * Incremental greedy class merging
 *	This is the second, O(VC^2) algorithm in Brown et al. (1992)
 */
void
UniqueWordClasses::incrementalMerge(unsigned numClasses)
{
    if (numClasses < 1) {
	numClasses = 1;
    }

    TextStats stats;
    getStats(stats);

    unsigned numTokens = (unsigned)(stats.numWords + stats.numSentences);

    unsigned iters = 0;
    int saveIters = -1;

    if (debug(DEBUG_TRACE_MERGE)) {
	dout() << "iter " << iters
	       << ": " << classVocab.numWords() << " classes, "
	       << "perplexity = " << LogPtoPPL(stats.prob/numTokens)
	       << endl;
    }

    /*
     * Sort classes by count into listOfClasses.
     * nClasses = total number of classes
     */
    StaticArray<VocabIndex> listOfClasses(vocab.numWords());

    unsigned nClasses = 0;

    VocabIndex unigram[2];
    orderClasses = this;
    NgramsIter unigramIter(classNgramCounts, unigram, 1, orderByCount);
    NgramCount *classCount;

    while ((classCount = unigramIter.next())) {
	if (classVocab.getWord(unigram[0]) != 0) {
	    listOfClasses[nClasses ++] = unigram[0];
	}
    }

    /*
     * Construct the subset of classes undergoing merging
     */
    SubVocab mergeSet(classVocab);

    /*
     * Add the first numClasses to the merge set
     */
    unsigned i;
    for (i = 0; i < nClasses && i < numClasses; i++) {
	if (debug(DEBUG_TRACE_MERGE)) {
	    dout() << "\tadding " << classVocab.getWord(listOfClasses[i])
		   << endl;
	}
	mergeSet.addWord(listOfClasses[i]);
    }

    /*
     * Now add one extra class at a time, and merge after each addition
     */
    for ( ; i < nClasses; i ++) {
	if (debug(DEBUG_TRACE_MERGE)) {
	    dout() << "\tadding " << classVocab.getWord(listOfClasses[i])
		   << endl;
	}
	mergeSet.addWord(listOfClasses[i]);

	iters ++;
	VocabIndex b1, b2;

	bestMerge(mergeSet, b1, b2);

	if (debug(DEBUG_TRACE_MERGE)) {
	    dout() << "iter " << iters
		   << ": " << classVocab.numWords() << " classes, "
		   << "perplexity = " << LogPtoPPL(totalLogP()/numTokens)
		   << endl;
	}

	mergeSet.remove(b2);

	/*
	 * Remove the class from merge set if it contains maxWordsPerClass
	 * words (and add a new class into merge set after that)
	 */
	if (maxWordsPerClass > 0) {
	    LHashIter<VocabIndex,VocabIndex> iter(wordToClass);
	    VocabIndex *clasz;
	    VocabIndex word;

	    unsigned wordCount = 0;
	    while ((clasz = iter.next(word))) {
		if (*clasz == b1) {
		    wordCount += 1;
		}
	    }

	    if (wordCount >= maxWordsPerClass) {
		mergeSet.remove(b1);

		i += 1;
		if (i < nClasses) {
		    if (debug(DEBUG_TRACE_MERGE)) {
			dout() << "\tclass " << classVocab.getWord(b1)
			       << " now contains " << maxWordsPerClass << " words" << endl;
			dout() << "\tadding " << classVocab.getWord(listOfClasses[i])
			       << endl;
		    }
		    mergeSet.addWord(listOfClasses[i]);
		}
	    }
	}

	/*
	 * Save classes and counts if and when requested
	 */
	if (saveMaxClasses == 0) {
	    saveIters = iters;
	} else if (saveIters < 0) {
	    if (classVocab.numWords() <= saveMaxClasses) {
		saveIters = 0;		// start saving now
	    }
	} else {
	    saveIters ++;
	}

	File *cf = logFile(classesFile, saveFreq, saveIters);
	if (cf != 0) {
	    writeClasses(*cf);
	    delete cf;
	}

	cf = logFile(classCountsFile, saveFreq, saveIters);
	if (cf != 0) {
	    writeCounts(*cf);
	    delete cf;
	}
    }
}

/*
 * Simple interactive class merging
 */
void
interactiveMerge(UniqueWordClasses &classes)
{
    while (1) {
	char class1[30], class2[30];
	class1[0] = class2[0] = '\0';

	cout << "Enter two class names> ";
	// @kw N/A (non-library): SV.UNBOUND_STRING_INPUT.CIN
	cin >> class1 >> class2 ;

	if (!*class1) break;

	VocabIndex c1 = classes.classVocab.getIndex(class1);
	if (c1 == Vocab_None) {
	    cerr << class1 << " is not a valid class; try again.\n";
	    continue;
	}

	VocabIndex c2 = classes.classVocab.getIndex(class2);
	if (c2 == Vocab_None) {
	    cerr << class2 << " is not a valid class; try again.\n";
	    continue;
	}

	if (c1 == c2) {
	    cerr << "Classes must be distinct; try again.\n";
	    continue;
	}

	cout << "Merging class " << classes.classVocab.getWord(c1)
	     << " and " << classes.classVocab.getWord(c2) << endl;

	LogP delta = classes.diffLogP(c1, c2);
	cout << "Projected delta = " << delta << endl;

	classes.merge(c1, c2);

	{
	    if (classesFile) {
		File file(classesFile, "w");
		classes.writeClasses(file);
	    }

	    if (classCountsFile) {
		File file(classCountsFile, "w");
		classes.writeCounts(file);
	    }

	    if (debug >= DEBUG_PRINT_CONTRIBS) {
		File file(stdout);

		classes.writeContribs(file);

		classes.computeClassContribs();
		classes.computeMergeContribs();

		classes.writeContribs(file);
	    }

	    TextStats stats;
	    classes.getStats(stats);
	    cout << stats;
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

    Vocab vocab;
    vocab.toLower() = toLower ? true : false;

    SubVocab classVocab(vocab);
    SubVocab noclassVocab(vocab);

    UniqueWordClasses classes(vocab, classVocab);
    classes.debugme(debug);

    if (vocabFile) {
	File file(vocabFile, "r");
	vocab.read(file);
    }

    if (noclassFile) {
	File file(noclassFile, "r");
	noclassVocab.read(file);
    } else {
	/*
	 * Assume <s> and </s> are not to be classed
	 */
	noclassVocab.addWord(vocab.ssIndex());
	noclassVocab.addWord(vocab.seIndex());
    }

    if (countsFile || textFile) {
	NgramStats bigrams(vocab, 2);
	bigrams.debugme(debug);

	/*
	 * Restrict vocabulary if user specied one
	 */
	if (vocabFile) {
	    bigrams.openVocab = false;
	}

	if (textFile) {
	    File file(textFile, "r");
	    bigrams.countFile(file);
	}

	if (countsFile) {
	    File file(countsFile, "r");
	    bigrams.read(file);
	}

	if (readClassesFile) {
	    File file(readClassesFile, "r");
	    if (!classes.readClasses(file)) {
		exit(2);
	    }
	}

	classes.initialize(bigrams, noclassVocab);
    } else {
	cerr << "Specify counts or text file as input.\n";
	exit(1);
    }

    if (numClasses > 0) {
	if (fullMerge) {
	    classes.fullMerge(numClasses);
	} else {
	    classes.incrementalMerge(numClasses);
	}
    }

    if (classesFile) {
	File file(classesFile, "w");

	classes.writeClasses(file);
    }

    if (classCountsFile) {
	File file(classCountsFile, "w");

	classes.writeCounts(file);
    }

    if (debug >= DEBUG_TEXTSTATS) {
	TextStats stats;
	classes.getStats(stats);
	cerr << stats;
    }

    if (interact) {
	interactiveMerge(classes);
    }

    exit(0);
}


