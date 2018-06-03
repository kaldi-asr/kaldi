/*
 * WordMesh.cc --
 *	Word Meshes (aka Confusion Networks aka Sausages)
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2012 SRI International, 2012-2016 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/WordMesh.cc,v 1.56 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "WordMesh.h"
#include "WordAlign.h"

#include "TLSWrapper.h"
#include "Array.cc"
#include "LHash.cc"
#include "SArray.cc"

// Instantiate template types for external programs linking to the lib 
INSTANTIATE_LHASH(VocabIndex,NBestWordInfo);

typedef struct {
  double cost;			// minimal cost of partial alignment
  WordAlignType error;		// best predecessor
} ChartEntryDouble;

typedef struct {
  unsigned cost;		// minimal cost of partial alignment
  WordAlignType error;		// best predecessor
} ChartEntryUnsigned;

/*
 * Special token used to represent an empty position in an alignment column
 */
const VocabString deleteWord = "*DELETE*";

template <class CT>
void freeChart(CT **chart, unsigned maxRefLength)
{
    for (unsigned i = 0; i <= maxRefLength; i ++) {
        delete [] chart[i];
    }
    delete [] chart;
}

WordMesh::WordMesh(Vocab &vocab, const char *myname, VocabDistance *distance)
    : MultiAlign(vocab, myname), totalPosterior(0.0), numAligns(0),
      distance(distance)
{
    deleteIndex = vocab.addWord(deleteWord);
}

WordMesh::~WordMesh()
{
    if (name != 0) {
	free(name);
    }

    for (unsigned i = 0; i < numAligns; i ++) {
	delete aligns[i];

	LHashIter<VocabIndex,NBestWordInfo> infoIter(*wordInfo[i]);
	NBestWordInfo *winfo;
	VocabIndex word;
	while ((winfo = infoIter.next(word))) {
	    winfo->~NBestWordInfo();
	}
	delete wordInfo[i];

	LHashIter<VocabIndex,Array<HypID> > mapIter(*hypMap[i]);
	Array<HypID> *hyps;
	while ((hyps = mapIter.next(word))) {
	    hyps->~Array();
	}
	delete hypMap[i];
    }
}
   
Boolean
WordMesh::isEmpty()
{
    return numAligns == 0;
}

/*
 * alignment set to sort by posterior (parameter to comparePosteriors)
 */
static TLSW(void*, compareAlignTLS);

static int
comparePosteriors(VocabIndex w1, VocabIndex w2)
{
    LHash<VocabIndex, Prob>* compareAlign 
        = (LHash<VocabIndex, Prob>*)TLSW_GET(compareAlignTLS);

    Prob* p1 = compareAlign->find(w1);
    Prob* p2 = compareAlign->find(w2);
    if (!p1 || !p2) {
	// Unexpected error case with no meaningful return value
	return 0;
    }
    Prob diff = *p1 - *p2;

    if (diff < 0.0) {
	return 1;
    } else if (diff > 0.0) {
	return -1;
    } else {
	return 0;
    }
}

Boolean
WordMesh::write(File &file)
{
    void* &compareAlign = TLSW_GET(compareAlignTLS);

    if (name != 0) {
	file.fprintf("name %s\n", name);
    }
    file.fprintf("numaligns %u\n", numAligns);
    file.fprintf("posterior %.*lg\n", Prob_Precision, (double)totalPosterior);

    for (unsigned i = 0; i < numAligns; i ++) {
	file.fprintf("align %u", i);

	compareAlign = aligns[sortedAligns[i]];
	LHashIter<VocabIndex,Prob> alignIter(*((LHash<VocabIndex, Prob> *)compareAlign), comparePosteriors);

	Prob *prob;
	VocabIndex word;
	VocabIndex refWord = Vocab_None;

	while ((prob = alignIter.next(word))) {
	    file.fprintf(" %s %.*lg", vocab.getWord(word),
				      Prob_Precision, *prob);

	    /*
	     * See if this word is the reference one
	     */
	    Array<HypID> *hypList = hypMap[sortedAligns[i]]->find(word);
	    if (hypList) {
		for (unsigned j = 0; j < hypList->size(); j++) {
		    if ((*hypList)[j] == refID) {
			refWord = word;
			break;
		    }
		}
	    }
	}
	file.fprintf("\n");

	/*
	 * Print column and transition posterior sums,
	 * if different from total Posterior
	 */
	Prob myPosterior = columnPosteriors[sortedAligns[i]];

	if (myPosterior != totalPosterior) {
	    file.fprintf("posterior %u %.*lg\n", i, Prob_Precision, myPosterior);
	}

	Prob transPosterior = transPosteriors[sortedAligns[i]];

	if (transPosterior != totalPosterior) {
	    file.fprintf("transposterior %u %.*lg\n", i, Prob_Precision, transPosterior);
	}

	/* 
	 * Print reference word (if known)
	 */
	if (refWord != Vocab_None) {
	    file.fprintf("reference %u %s\n", i, vocab.getWord(refWord));
	}

	/*
	 * Dump hyp IDs if known
	 */
	LHashIter<VocabIndex,Array<HypID> >
			mapIter(*hypMap[sortedAligns[i]], comparePosteriors);
	Array<HypID> *hypList;

	while ((hypList = mapIter.next(word))) {
	    /*
	     * Only output hyp IDs if they are different from the refID
	     * (to avoid redundancy with "reference" line)
	     */
	    if (hypList->size() > (unsigned) (word == refWord)) {
		file.fprintf("hyps %u %s", i, vocab.getWord(word));

		for (unsigned j = 0; j < hypList->size(); j++) {
		    if ((*hypList)[j] != refID) {
			file.fprintf(" %d", (int)(*hypList)[j]);
		    }
		}
		file.fprintf("\n");
	    }
	}

	/*
	 * Dump word backtrace info if known
	 */
	LHashIter<VocabIndex,NBestWordInfo>
			infoIter(*wordInfo[sortedAligns[i]], comparePosteriors);
	NBestWordInfo *winfo;

	while ((winfo = infoIter.next(word))) {
	    file.fprintf("info %u %s ", i, vocab.getWord(word));
	    winfo->write(file);
	    file.fprintf("\n");
	}
    }

    return true;
}

Boolean
WordMesh::read(File &file)
{
    for (unsigned i = 0; i < numAligns; i ++) {
	delete aligns[i];
    }
   
    char *line;

    totalPosterior = 1.0;

    while ((line = file.getline())) {
	char arg1[100];
	double arg2;
	int parsed;
	unsigned index;

	if (sscanf(line, "numaligns %u", &index) == 1) {
	    if (numAligns > 0) {
		file.position() << "repeated numaligns specification\n";
		return false;
	    }
	    numAligns = index;

	    // @kw false positive: SV.TAINTED.LOOP_BOUND (this->numAligns)
	    for (unsigned i = 0; i < numAligns; i ++) {
		sortedAligns[i] = i;

		aligns[i] = new LHash<VocabIndex,Prob>;
		assert(aligns[i] != 0);

		wordInfo[i] = new LHash<VocabIndex,NBestWordInfo>;
		assert(wordInfo[i] != 0);

		hypMap[i] = new LHash<VocabIndex,Array<HypID> >;
		assert(hypMap[i] != 0);

		columnPosteriors[i] = transPosteriors[i] = totalPosterior;
	    }
	} else if (sscanf(line, "name %100s", arg1) == 1) {
	    if (name != 0) {
		free(name);
	    }
	    name = strdup(arg1);
	    assert(name != 0);
	} else if (sscanf(line, "posterior %100s %lg", arg1, &arg2) == 2 &&
	           // scan node index with %s so we fail if only one numerical
		   // arg is given (which case handled below)
		   sscanf(arg1, "%u", &index) == 1)
	    {
	    if (index >= numAligns) {
		file.position() << "position index exceeds numaligns\n";
		return false;
	    }

	    columnPosteriors[index] = arg2;
	} else if (sscanf(line, "transposterior %u %lg", &index, &arg2) == 2) {
	    if (index >= numAligns) {
		file.position() << "position index exceeds numaligns\n";
		return false;
	    }

	    transPosteriors[index] = arg2;
	} else if (sscanf(line, "posterior %lg", &arg2) == 1) {
	    totalPosterior = arg2;
	    for (unsigned j = 0; j < numAligns; j ++) {
		columnPosteriors[j] = transPosteriors[j] = arg2;
	    }
	} else if (sscanf(line, "align %u%n", &index, &parsed) == 1) {
	    if (index >= numAligns) {
		file.position() << "position index exceeds numaligns\n";
		return false;
	    }

	    // @kw false positive: SV.TAINTED.INDEX_ACCESS (parsed)
	    char *cp = line + parsed;
	    while (sscanf(cp, "%100s %lg%n", arg1, &arg2, &parsed) == 2) {
		VocabIndex word = vocab.addWord(arg1);

		*aligns[index]->insert(word) = arg2;
		
		cp += parsed;
	    }
	} else if (sscanf(line, "reference %u %100s", &index, arg1) == 2) {
	    if (index >= numAligns) {
		file.position() << "position index exceeds numaligns\n";
		return false;
	    }

	    VocabIndex refWord = vocab.addWord(arg1);

	    /*
	     * Records word as part of the reference string
	     */
	    Array<HypID> *hypList = hypMap[index]->insert(refWord);
	    (*hypList)[hypList->size()] = refID;
	} else if (sscanf(line, "hyps %u %100s%n", &index, arg1, &parsed) == 2){
	    if (index >= numAligns) {
		file.position() << "position index exceeds numaligns\n";
		return false;
	    }

	    VocabIndex word = vocab.addWord(arg1);
	    Array<HypID> *hypList = hypMap[index]->insert(word);

	    /*
	     * Parse and record hyp IDs
	     */
	    char *cp = line + parsed;
	    unsigned hypID;
	    while (sscanf(cp, "%u%n", &hypID, &parsed) == 1) {
		(*hypList)[hypList->size()] = hypID;
		*allHyps.insert(hypID) = hypID;

		cp += parsed;
	    }
	} else if (sscanf(line, "info %u %100s%n", &index, arg1, &parsed) == 2){
	    if (index >= numAligns) {
		file.position() << "position index exceeds numaligns\n";
		return false;
	    }

	    VocabIndex word = vocab.addWord(arg1);
	    NBestWordInfo *winfo = wordInfo[index]->insert(word);

	    winfo->word = word;
	    if (!winfo->parse(line + parsed)) {
		file.position() << "invalid word info\n";
		return false;
	    }
	} else {
	    file.position() << "unknown keyword\n";
	    return false;
	}
    }
    return true;
}

/*
 * Compute expected error from aligning a word to an alignment column
 * if column == 0 : compute insertion cost
 * if word == deleteIndex : compute deletion cost
 */
double
WordMesh::alignError(const LHash<VocabIndex,Prob>* column,
		     Prob columnPosterior,
		     VocabIndex word)
{
    if (column == 0) {
	/*
	 * Compute insertion cost for word
	 */
	if (word == deleteIndex) {
	    return 0.0;
	} else {
	    if (distance) {
		return columnPosterior * distance->penalty(word);
	    } else {
		return columnPosterior;
	    }
	}
    } else {
	if (word == deleteIndex) {
	    /* 
	     * Compute deletion cost for alignment column
	     */
	    if (distance) {
		double deletePenalty = 0.0;

		LHashIter<VocabIndex,Prob> iter(*column);
		Prob *prob;
		VocabIndex alignWord;
		while ((prob = iter.next(alignWord))) {
		    if (alignWord != deleteIndex) {
			deletePenalty += *prob * distance->penalty(alignWord);
		    }
		}
		return deletePenalty;
	    } else {
		Prob *deleteProb = column->find(deleteIndex);
		return  columnPosterior - (deleteProb ? *deleteProb : 0.0);
	    }
	} else {
	    /*
	     * Compute "substitution" cost of word in column
	     */
	    if (distance) {
		/*
		 * Compute distance to existing alignment as a weighted 
		 * combination of distances
		 */
		double totalDistance = 0.0;

	    	LHashIter<VocabIndex,Prob> iter(*column);
		Prob *prob;
		VocabIndex alignWord;
		while ((prob = iter.next(alignWord))) {
		    if (alignWord == deleteIndex) {
			totalDistance +=
			    *prob * distance->penalty(word);
		    } else {
			totalDistance +=
			    *prob * distance->distance(alignWord, word);
		    }
		}

		return totalDistance;
	    } else {
	        Prob *wordProb = column->find(word);
		return columnPosterior - (wordProb ? *wordProb : 0.0);
	    }
	}
    }
}

/*
 * Compute expected error from aligning two alignment columns
 * if column1 == 0 : compute insertion cost
 * if column2 == 0 : compute deletion cost
 */
double
WordMesh::alignError(const LHash<VocabIndex,Prob>* column1, 
		     Prob columnPosterior,
		     const LHash<VocabIndex,Prob>* column2,
		     Prob columnPosterior2)
{
    if (column2 == 0) {
	return alignError(column1, columnPosterior, deleteIndex);
    } else {
	/*
	 * compute weighted sum of aligning each of the column2 entries,
	 * using column2 posteriors as weights
	 */
	double totalDistance = 0.0;

	LHashIter<VocabIndex,Prob> iter(*column2);
	Prob *prob;
	VocabIndex word2;
	while ((prob = iter.next(word2))) {
	    double error = alignError(column1, columnPosterior, word2);

	    /*
	     * Handle case where one of the entries has posterior 1, but 
	     * others have small nonzero posteriors, too.  The small ones
	     * can be ignored in the sum total, and this shortcut makes the
	     * numerical computation symmetric with respect to the case
	     * where posterior 1 occurs in column1 (as well as speeding things
	     * up).
	     */
	    if (*prob == columnPosterior2) {
		return *prob * error;
	    } else {
		totalDistance += *prob * error;
	    }
	}
	return totalDistance;
    }
}

/*
 * Align new words to existing alignment columns, expanding them as required
 * (derived from WordAlign())
 * If hypID != 0, then *hypID will record a sentence hyp ID for the 
 * aligned words.
 */
void
WordMesh::alignWords(const VocabIndex *words, Prob score,
			    Prob *wordScores, const HypID *hypID)
{
    unsigned numWords = Vocab::length(words);
    NBestWordInfo *winfo = new NBestWordInfo[numWords + 1];
    assert(winfo != 0);

    /*
     * Fill word info array with word IDs and dummy info
     * Note: loop below also handles the final Vocab_None.
     */
    for (unsigned i = 0; i <= numWords; i ++) {
	winfo[i].word = words[i];
	winfo[i].wordPosterior = 0.0;
	winfo[i].transPosterior = 0.0;
    }

    alignWords(winfo, score, wordScores, hypID);

    delete [] winfo;
}

/*
 * This is the generalized version of alignWords():
 *	- merges NBestWordInfo into the existing alignment
 *	- aligns word string between any two existing alignment positions
 *	- optionally returns the alignment positions assigned to aligned words
 *	- optionally returns the posterior probabilities of aligned words
 * Alignment positions are integers corresponding to word equivalence classes.
 * They are assigned in increasing order; hence numerical order does not
 * correspond to topological (temporal) order.
 *
 *	- 'from' specifies the alignment position just BEFORE the first word.
 *	  A value of numAligns means the first word should start the alignment.
 *	- 'to' specifies the alignment position just AFTER the last word.
 *	  A value of numAligns means the last word should end the alignment.
 *
 * Returns false if the 'from' position does not strictly precede the 'to'.
 */

static TLSW(unsigned, alignWordsMaxHypLengthTLS);
static TLSW(unsigned, alignWordsMaxRefLengthTLS);
static TLSW(ChartEntryDouble**, alignWordsChartTLS);

Boolean
WordMesh::alignWords(const NBestWordInfo *winfo, Prob score,
		    	Prob *wordScores, const HypID *hypID,
			unsigned from, unsigned to, unsigned *wordAlignment) 
{
    /*
     * Compute word string length
     */
    unsigned hypLength = 0;
    for (unsigned i = 0; winfo[i].word != Vocab_None; i ++) hypLength ++;

    /*
     * Locate start and end positions to align to
     */
    unsigned fromPos = 0;
    unsigned refLength = 0;

    if (numAligns > 0) {
	unsigned toPos = numAligns - 1;

	for (unsigned p = 0; p < numAligns; p ++) {
	    if (sortedAligns[p] == from) fromPos = p + 1;
	    if (sortedAligns[p] == to) toPos = p - 1;
	}

    	refLength = toPos - fromPos + 1;

        if (toPos + 1 < fromPos) {
	    
	    return false;
	}
    }
    Boolean fullAlignment = (refLength == numAligns);

    /* 
     * Allocate chart statically, enlarging on demand
     */
    unsigned &maxHypLength    = TLSW_GET(alignWordsMaxHypLengthTLS);
    unsigned &maxRefLength    = TLSW_GET(alignWordsMaxRefLengthTLS);
    ChartEntryDouble** &chart = TLSW_GET(alignWordsChartTLS);

    if (chart == 0 || hypLength > maxHypLength || refLength > maxRefLength) {
	/*
	 * Free old chart
	 */
        if (chart !=0)
            freeChart<ChartEntryDouble>(chart, maxRefLength);
	/*
	 * Allocate new chart
	 */
	maxHypLength = hypLength;
	maxRefLength = refLength;
    
	chart = new ChartEntryDouble*[maxRefLength + 1];
	assert(chart != 0);

	for (unsigned i = 0; i <= maxRefLength; i ++) {
	    chart[i] = new ChartEntryDouble[maxHypLength + 1];
	    assert(chart[i] != 0);
	}
    }

    /*
     * Initialize the 0'th row
     */
    chart[0][0].cost = 0.0;
    chart[0][0].error = CORR_ALIGN;

    for (unsigned j = 1; j <= hypLength; j ++) {
	chart[0][j].cost = chart[0][j-1].cost +
				alignError(0, totalPosterior, winfo[j-1].word);
	chart[0][j].error = INS_ALIGN;
    }

    /*
     * Fill in the rest of the chart, row by row.
     */
    for (unsigned i = 1; i <= refLength; i ++) {
	double deletePenalty =
		    alignError(aligns[sortedAligns[fromPos+i-1]],
			       columnPosteriors[sortedAligns[fromPos+i-1]],
			       deleteIndex);

	chart[i][0].cost = chart[i-1][0].cost + deletePenalty;
	chart[i][0].error = DEL_ALIGN;

	for (unsigned j = 1; j <= hypLength; j ++) {
	    double minCost = chart[i-1][j-1].cost +
			alignError(aligns[sortedAligns[fromPos+i-1]],
				   columnPosteriors[sortedAligns[fromPos+i-1]],
				   winfo[j-1].word);
	    WordAlignType minError = SUB_ALIGN;

	    double delCost = chart[i-1][j].cost + deletePenalty;
	    if (delCost + Prob_Epsilon < minCost) {
		minCost = delCost;
		minError = DEL_ALIGN;
	    }

	    double insCost = chart[i][j-1].cost +
			alignError(0,
			           transPosteriors[sortedAligns[fromPos+i-1]],
				   winfo[j-1].word);
	    if (insCost + Prob_Epsilon < minCost) {
		minCost = insCost;
		minError = INS_ALIGN;
	    }

	    chart[i][j].cost = minCost;
	    chart[i][j].error = minError;
	}
    }

    /*
     * Backtrace and add new words to alignment columns.
     * Store word posteriors if requested.
     */
    {
	unsigned i = refLength;
	unsigned j = hypLength;

	while (i > 0 || j > 0) {

	    Prob wordPosterior = score;
	    Prob transPosterior = score;

	    /*
	     * Use word- and transition-specific posteriors if supplied.
	     * Note: the transition posterior INTO the first word is
	     * given on the Vocab_None item at winfo[hypLength].
	     */
	    if (j > 0 && winfo[j-1].wordPosterior != 0.0) {
		wordPosterior = winfo[j-1].wordPosterior;
	    }
	    if (j > 0 && winfo[j-1].transPosterior != 0.0) {
		transPosterior = winfo[j-1].transPosterior;
	    } else if (j == 0 && winfo[hypLength].transPosterior != 0.0) {
		transPosterior = winfo[hypLength].transPosterior;
	    }

	    switch (chart[i][j].error) {
	    case END_ALIGN:
		assert(0);
		break;
	    case CORR_ALIGN:
	    case SUB_ALIGN:
		*aligns[sortedAligns[fromPos+i-1]]->insert(winfo[j-1].word) +=
								wordPosterior;
		columnPosteriors[sortedAligns[fromPos+i-1]] += wordPosterior;
		transPosteriors[sortedAligns[fromPos+i-1]] += transPosterior;

		/*
		 * Check for preexisting word info and merge if necesssary
		 */
		if (winfo[j-1].valid()) {
		    Boolean foundP;
		    NBestWordInfo *oldInfo =
		    		wordInfo[sortedAligns[fromPos+i-1]]->
						insert(winfo[j-1].word, foundP);
		    if (foundP) {
			oldInfo->merge(winfo[j-1], wordPosterior);
		    } else {
			*oldInfo = winfo[j-1];
		    }
		}

		if (hypID) {
		    /*
		     * Add hyp ID to the hyp list for this word 
		     */
		    Array<HypID> &hypList = 
		    *hypMap[sortedAligns[fromPos+i-1]]->insert(winfo[j-1].word);
		    hypList[hypList.size()] = *hypID;
		}

		if (wordAlignment) {
		    wordAlignment[j-1] = sortedAligns[fromPos+i-1];
		}
		if (wordScores) {
		    Prob* p = aligns[sortedAligns[fromPos+i-1]]->find(winfo[j-1].word);
		    // For unexpected NULL error case, use LogP_Zero
		    wordScores[j-1] = p?*p:LogP_Zero;
		}

		i --; j --;
		break;
	    case DEL_ALIGN:
		*aligns[sortedAligns[fromPos+i-1]]->insert(deleteIndex) +=
								transPosterior;
		columnPosteriors[sortedAligns[fromPos+i-1]] += transPosterior;
		transPosteriors[sortedAligns[fromPos+i-1]] += transPosterior;

		if (hypID) {
		    /*
		     * Add hyp ID to the hyp list for this word 
		     */
		    Array<HypID> &hypList = 
			*hypMap[sortedAligns[fromPos+i-1]]->insert(deleteIndex);
		    hypList[hypList.size()] = *hypID;
		}

		i --;
		break;
	    case INS_ALIGN:
		/*
		 * Make room for new alignment column in sortedAligns
		 * and position new column
		 */
		for (unsigned k = numAligns; k > fromPos + i; k --) {
		    // use an intermediate variable to avoid bug in MVC.
		    unsigned a = sortedAligns[k-1];
		    sortedAligns[k] = a;
		}

		sortedAligns[fromPos + i] = numAligns;

		aligns[numAligns] = new LHash<VocabIndex,Prob>;
		assert(aligns[numAligns] != 0);

		wordInfo[numAligns] = new LHash<VocabIndex,NBestWordInfo>;
		assert(wordInfo[numAligns] != 0);

		hypMap[numAligns] = new LHash<VocabIndex,Array<HypID> >;
		assert(hypMap[numAligns] != 0);

		/*
		 * The transition posterior from the preceding to the following
		 * position becomes the posterior for *delete* at the new
		 * position (that's why we need transition posteriors!).
		 */
		Prob nullPosterior = totalPosterior;
		if (fromPos+i > 0) {
		    nullPosterior = transPosteriors[sortedAligns[fromPos+i-1]];
		}
		if (nullPosterior != 0.0) {
		    *aligns[numAligns]->insert(deleteIndex) = nullPosterior;
		}
		*aligns[numAligns]->insert(winfo[j-1].word) = wordPosterior;
		columnPosteriors[numAligns] = nullPosterior + wordPosterior;
		transPosteriors[numAligns] = nullPosterior + transPosterior;

		/*
		 * Record word info if given
		 */
		if (winfo[j-1].valid()) {
		    *wordInfo[numAligns]->insert(winfo[j-1].word) = winfo[j-1];
		}

		/*
		 * Add all hypIDs previously recorded to the *DELETE*
		 * hypothesis at the newly created position.
		 */
		{
		    Array<HypID> *hypList = 0;

		    SArrayIter<HypID,HypID> myIter(allHyps);
		    HypID id;
		    while (myIter.next(id)) {
			/*
			 * Avoid inserting *DELETE* in hypMap unless needed
			 */
			if (hypList == 0) {
			    hypList = hypMap[numAligns]->insert(deleteIndex);
			}
			(*hypList)[hypList->size()] = id;
		    }
		}

		if (hypID) {
		    /*
		     * Add hyp ID to the hyp list for this word 
		     */
		    Array<HypID> &hypList = 
				*hypMap[numAligns]->insert(winfo[j-1].word);
		    hypList[hypList.size()] = *hypID;
		}

		if (wordAlignment) {
		    wordAlignment[j-1] = numAligns;
		}
		if (wordScores) {
		    wordScores[j-1] = wordPosterior;
		}

		numAligns ++;
		j --;
		break;
	    }
	}

	/* 
	 * Add the transition posterior INTO the FIRST hyp word to
	 * to the transition posterior into the first alignment position
	 * This only applies when doing a partial alignment that doesn't
	 * start at the 0th position.
	 */
	if (fromPos+i > 0) {
	    Prob transPosterior = score;

	    if (winfo[hypLength].transPosterior != 0.0) {
		transPosterior = winfo[hypLength].transPosterior;
	    }

	    transPosteriors[sortedAligns[fromPos+i-1]] += transPosterior;
	}
    }

    /*
     * Add hyp to global list of IDs
     */
    if (hypID) {
	*allHyps.insert(*hypID) = *hypID;
    }

    /*
     * Only change total posterior if the alignment spanned the whole mesh.
     * This ensure totalPosterior reflects the initia/final node posterior
     * in a lattice.
     */
    if (fullAlignment) {
	totalPosterior += score;
    }
    return true;
}


NBestWordInfo* 
WordMesh::wordInfoFromUnsortedColumn(unsigned unsortedColumnIndex, VocabIndex word)
{
    if (unsortedColumnIndex >= numAligns) {
        return NULL;
    } 
    else {
        LHash<VocabIndex,NBestWordInfo>* vocabToInfo = wordInfo[unsortedColumnIndex];
        return vocabToInfo->find(word);
    }
}


Prob
WordMesh::wordProbFromUnsortedColumn(unsigned unsortedColumnIndex, VocabIndex word) {
    if (unsortedColumnIndex < numAligns) {
        LHash<VocabIndex,Prob>* vocabToProb = aligns[unsortedColumnIndex];
	Prob* probability = vocabToProb->find(word); 
        if (probability == NULL) {
            return 0;
        }
        else {
            return *probability;
        }
    } 
    else {
	return 0;
    }
}


static TLSW(unsigned, alignAlignmentMaxHypLengthTLS);
static TLSW(unsigned, alignAlignmentMaxRefLengthTLS);
static TLSW(ChartEntryDouble**, alignAlignmentChartTLS);

void
WordMesh::alignAlignment(MultiAlign &alignment, Prob score, Prob *alignScores)
{
    WordMesh &other = (WordMesh &)alignment;

    unsigned refLength = numAligns;
    unsigned hypLength = other.numAligns;

    /* 
     * Allocate chart statically, enlarging on demand
     */
    unsigned &maxHypLength    = TLSW_GET(alignAlignmentMaxHypLengthTLS);
    unsigned &maxRefLength    = TLSW_GET(alignAlignmentMaxRefLengthTLS);
    ChartEntryDouble** &chart = TLSW_GET(alignAlignmentChartTLS);

    if (chart == 0 || hypLength > maxHypLength || refLength > maxRefLength) {
	/*
	 * Free old chart
	 */
        if (chart !=0)
            freeChart<ChartEntryDouble>(chart, maxRefLength);

	/*
	 * Allocate new chart
	 */
	maxHypLength = hypLength;
	maxRefLength = refLength;
    
	chart = new ChartEntryDouble*[maxRefLength + 1];
	assert(chart != 0);

	for (unsigned i = 0; i <= maxRefLength; i ++) {
	    chart[i] = new ChartEntryDouble[maxHypLength + 1];
	    assert(chart[i] != 0);
	}
    }

    /*
     * Initialize the 0'th row
     */
    chart[0][0].cost = 0.0;
    chart[0][0].error = CORR_ALIGN;

    for (unsigned j = 1; j <= hypLength; j ++) {
	unsigned pos = other.sortedAligns[j-1];
	chart[0][j].cost = chart[0][j-1].cost +
		   alignError(0,
			      totalPosterior,
			      other.aligns[other.sortedAligns[j-1]],
			      other.columnPosteriors[other.sortedAligns[j-1]]);
	chart[0][j].error = INS_ALIGN;
    }

    /*
     * Fill in the rest of the chart, row by row.
     */
    for (unsigned i = 1; i <= refLength; i ++) {
	double deletePenalty =
			    alignError(aligns[sortedAligns[i-1]],
				       columnPosteriors[sortedAligns[i-1]],
				       deleteIndex);

	chart[i][0].cost = chart[i-1][0].cost + deletePenalty;
	chart[i][0].error = DEL_ALIGN;

	for (unsigned j = 1; j <= hypLength; j ++) {
	    double minCost = chart[i-1][j-1].cost +
		     alignError(aligns[sortedAligns[i-1]],
				columnPosteriors[sortedAligns[i-1]],
				other.aligns[other.sortedAligns[j-1]],
				other.columnPosteriors[other.sortedAligns[j-1]]);
	    WordAlignType minError = SUB_ALIGN;

	    double delCost = chart[i-1][j].cost + deletePenalty;
	    if (delCost + Prob_Epsilon < minCost) {
		minCost = delCost;
		minError = DEL_ALIGN;
	    }

	    double insCost = chart[i][j-1].cost +
		     alignError(0,
				transPosteriors[sortedAligns[i-1]],
				other.aligns[other.sortedAligns[j-1]],
				other.columnPosteriors[other.sortedAligns[j-1]]);
	    if (insCost + Prob_Epsilon < minCost) {
		minCost = insCost;
		minError = INS_ALIGN;
	    }

	    chart[i][j].cost = minCost;
	    chart[i][j].error = minError;
	}
    }

    /*
     * Backtrace and add new words to alignment columns.
     * Store word posteriors if requested.
     */
    {
	unsigned i = refLength;
	unsigned j = hypLength;

	while (i > 0 || j > 0) {

	    switch (chart[i][j].error) {
	    case END_ALIGN:
		assert(0);
		break;
	    case CORR_ALIGN:
	    case SUB_ALIGN:
		/*
		 * merge all words in "other" alignment column into our own
		 */
		{
		    double totalScore = 0.0;

		    LHashIter<VocabIndex,Prob>
				iter(*other.aligns[other.sortedAligns[j-1]]);
		    Prob *otherProb;
		    VocabIndex otherWord;
		    while ((otherProb = iter.next(otherWord))) {
			totalScore +=
			    (*aligns[sortedAligns[i-1]]->insert(otherWord) +=
							    score * *otherProb);
			/*
			 * Check for preexisting word info and merge if
			 * necesssary
			 */
			NBestWordInfo *otherInfo =
				    other.wordInfo[other.sortedAligns[j-1]]->
								find(otherWord);
			if (otherInfo) {
			    Boolean foundP;
			    NBestWordInfo *oldInfo =
					wordInfo[sortedAligns[i-1]]->
						    insert(otherWord, foundP);
			    if (foundP) {
				oldInfo->merge(*otherInfo);
			    } else {
				*oldInfo = *otherInfo;
			    }
			}

			Array<HypID> *otherHypList =
					other.hypMap[other.sortedAligns[j-1]]->
								find(otherWord);
			if (otherHypList) {
			    /*
			     * Add hyp IDs to the hyp list for this word 
			     * XXX: hyp IDs should be disjoint but there is no
			     * checking of this!
			     */
			    Array<HypID> &hypList =
						*hypMap[sortedAligns[i-1]]->
							insert(otherWord);
			    for (unsigned h = 0; h < otherHypList->size(); h ++)
			    {
				hypList[hypList.size()] = (*otherHypList)[h];
			    }
			}
		    }

		    if (alignScores) {
			alignScores[j-1] = totalScore;
		    }
		}

		columnPosteriors[sortedAligns[i-1]] +=
			score * other.columnPosteriors[other.sortedAligns[j-1]];
		transPosteriors[sortedAligns[i-1]] +=
			score * other.transPosteriors[other.sortedAligns[j-1]];

		i --; j --;

		break;
	    case DEL_ALIGN:
		*aligns[sortedAligns[i-1]]->insert(deleteIndex) +=
						score * other.totalPosterior;
		columnPosteriors[sortedAligns[i-1]] +=
						score * other.totalPosterior;
		transPosteriors[sortedAligns[i-1]] +=
						score * other.totalPosterior;

		/*
		 * Add all hyp IDs from "other" alignment to our delete hyp
		 */
		if (other.allHyps.numEntries() > 0) {
		    Array<HypID> &hypList = 
			*hypMap[sortedAligns[i-1]]->insert(deleteIndex);

		    SArrayIter<HypID,HypID> otherHypsIter(other.allHyps);
		    HypID id;
		    while (otherHypsIter.next(id)) {
			hypList[hypList.size()] = id;
		    }
		}

		i --;
		break;
	    case INS_ALIGN:
		/*
		 * Make room for new alignment column in sortedAligns
		 * and position new column
		 */
		for (unsigned k = numAligns; k > i; k --) {
		    // use an intermediate variable to avoid bug in MVC.
		    unsigned a = sortedAligns[k-1];
		    sortedAligns[k] = a;
		}
		sortedAligns[i] = numAligns;

		aligns[numAligns] = new LHash<VocabIndex,Prob>;
		assert(aligns[numAligns] != 0);

		wordInfo[numAligns] = new LHash<VocabIndex,NBestWordInfo>;
		assert(wordInfo[numAligns] != 0);

		hypMap[numAligns] = new LHash<VocabIndex,Array<HypID> >;
		assert(hypMap[numAligns] != 0);

		/*
		 * The transition posterior from the preceding to the following
		 * position becomes the posterior for *delete* at the new
		 * position (that's why we need transition posteriors!).
		 */
		Prob nullPosterior = totalPosterior;
		if (i > 0) {
		    nullPosterior = transPosteriors[sortedAligns[i-1]];
		}
		if (nullPosterior != 0.0) {
		    *aligns[numAligns]->insert(deleteIndex) = nullPosterior;
		}
		columnPosteriors[numAligns] = nullPosterior;
		transPosteriors[numAligns] = nullPosterior;

		/*
		 * Add all hypIDs previously recorded to the *DELETE*
		 * hypothesis at the newly created position.
		 */
		{
		    Array<HypID> *hypList = 0;

		    SArrayIter<HypID,HypID> myIter(allHyps);
		    HypID id;
		    while (myIter.next(id)) {
			/*
			 * Avoid inserting *DELETE* in hypMap unless needed
			 */
			if (hypList == 0) {
			    hypList = hypMap[numAligns]->insert(deleteIndex);
			}
			(*hypList)[hypList->size()] = id;
		    }
		}

		/*
		 * insert all words in "other" alignment at current position
		 */
		{
		    LHashIter<VocabIndex,Prob>
				iter(*other.aligns[other.sortedAligns[j-1]]);
		    Prob *otherProb;
		    VocabIndex otherWord;
		    while ((otherProb = iter.next(otherWord))) {
			*aligns[numAligns]->insert(otherWord) +=
							score * *otherProb;

			/*
			 * Record word info if given
			 */
			NBestWordInfo *otherInfo =
				other.wordInfo[other.sortedAligns[j-1]]->
								find(otherWord);
			if (otherInfo) {
			    *wordInfo[numAligns]->insert(otherWord) =
								*otherInfo;
			}

			Array<HypID> *otherHypList =
				other.hypMap[other.sortedAligns[j-1]]->
								find(otherWord);
			if (otherHypList) {
			    /*
			     * Add hyp IDs to the hyp list for this word 
			     */
			    Array<HypID> &hypList = 
				      *hypMap[numAligns]->insert(otherWord);
			    for (unsigned h = 0; h < otherHypList->size(); h ++)
			    {
				hypList[hypList.size()] = (*otherHypList)[h];
			    }
			}
		    }

		    if (alignScores) {
			alignScores[j-1] = score;
		    }
		}

		columnPosteriors[numAligns] +=
			score * other.columnPosteriors[other.sortedAligns[j-1]];
		transPosteriors[numAligns] +=
			score * other.transPosteriors[other.sortedAligns[j-1]];

		numAligns ++;
		j --;
		break;
	    }
	}
    }

    /*
     * Add hyps from "other" alignment to global list of our IDs
     */
    SArrayIter<HypID,HypID> otherHypsIter(other.allHyps);
    HypID id;
    while (otherHypsIter.next(id)) {
	*allHyps.insert(id) = id;
    }

    totalPosterior += score * other.totalPosterior;
}

/*
 * Incremental partial alignments using alignWords() can leave the 
 * posteriors of null entries defective (because not all transition posteriors
 * were added to the alignment).
 * This function normalized the null posteriors so that the total column
 * posteriors sum to the totalPosterior.
 */
void
WordMesh::normalizeDeletes()
{
    for (unsigned i = 0; i < numAligns; i ++) {

	LHashIter<VocabIndex,Prob> alignIter(*aligns[i]);

	Prob wordPosteriorSum = 0.0;

	Prob *prob;
	VocabIndex word;
	while ((prob = alignIter.next(word))) {
	    if (word != deleteIndex) {
		wordPosteriorSum += *prob;
	    }
	}

	Prob deletePosterior;
	if (wordPosteriorSum - totalPosterior > Prob_Epsilon) {
	    cerr << "WordMesh::normalizeDeletes: word posteriors exceed total: "
		 << wordPosteriorSum << endl;
	    deletePosterior = 0.0;
	} else {
	    deletePosterior = totalPosterior - wordPosteriorSum;
	}

	/*
	 * Delete null tokens with zero posterior.
	 * Insert nulls that should be there based on missing posterior mass.
	 * Avoid deleting nulls with small positive posterior that were
	 * already present in the alignment.
	 */
	if (deletePosterior <= 0.0) {
	    aligns[i]->remove(deleteIndex);
	} else if (deletePosterior > Prob_Epsilon) {
	    *aligns[i]->insert(deleteIndex) = deletePosterior;
	}

	columnPosteriors[i] = totalPosterior;
	transPosteriors[i] = totalPosterior;
    }
}

/*
 * Compute minimal word error with respect to existing alignment columns
 * (derived from WordAlign())
 */
static TLSW(unsigned, wordErrorMaxHypLengthTLS);
static TLSW(unsigned, wordErrorMaxRefLengthTLS);
static TLSW(ChartEntryUnsigned **, wordErrorChartTLS);

unsigned
WordMesh::wordError(const VocabIndex *words,
				unsigned &sub, unsigned &ins, unsigned &del)
{
    unsigned hypLength = Vocab::length(words);
    unsigned refLength = numAligns;

    /* 
     * Allocate chart statically, enlarging on demand
     */
    unsigned &maxHypLength      = TLSW_GET(wordErrorMaxHypLengthTLS);
    unsigned &maxRefLength      = TLSW_GET(wordErrorMaxRefLengthTLS);
    ChartEntryUnsigned** &chart = TLSW_GET(wordErrorChartTLS);

    if (chart == 0 || hypLength > maxHypLength || refLength > maxRefLength) {
	/*
	 * Free old chart
	 */
	if (chart != 0)
            freeChart<ChartEntryUnsigned>(chart, maxRefLength);

	/*
	 * Allocate new chart
	 */
	maxHypLength = hypLength;
	maxRefLength = refLength;
    
	chart = new ChartEntryUnsigned*[maxRefLength + 1];
	assert(chart != 0);

	for (unsigned i = 0; i <= maxRefLength; i ++) {
	    chart[i] = new ChartEntryUnsigned[maxHypLength + 1];
	    assert(chart[i] != 0);
	}

	/*
	 * Initialize the 0'th row, which never changes
	 */
	chart[0][0].cost = 0;
	chart[0][0].error = CORR_ALIGN;

	for (unsigned j = 1; j <= maxHypLength; j ++) {
	    chart[0][j].cost = chart[0][j-1].cost + INS_COST;
	    chart[0][j].error = INS_ALIGN;
	}
    }

    /*
     * Fill in the rest of the chart, row by row.
     */
    for (unsigned i = 1; i <= refLength; i ++) {

	/*
	 * deletion error only if alignment column doesn't have *DELETE*
	 */
	Prob *delProb = aligns[sortedAligns[i-1]]->find(deleteIndex);
	unsigned THIS_DEL_COST = delProb && *delProb > 0.0 ? 0 : DEL_COST;
	
	chart[i][0].cost = chart[i-1][0].cost + THIS_DEL_COST;
	chart[i][0].error = DEL_ALIGN;

	for (unsigned j = 1; j <= hypLength; j ++) {
	    unsigned minCost;
	    WordAlignType minError;
	
	    if (aligns[sortedAligns[i-1]]->find(words[j-1])) {
		minCost = chart[i-1][j-1].cost;
		minError = CORR_ALIGN;
	    } else {
		minCost = chart[i-1][j-1].cost + SUB_COST;
		minError = SUB_ALIGN;
	    }

	    unsigned delCost = chart[i-1][j].cost + THIS_DEL_COST;
	    if (delCost < minCost) {
		minCost = delCost;
		minError = DEL_ALIGN;
	    }

	    unsigned insCost = chart[i][j-1].cost + INS_COST;
	    if (insCost < minCost) {
		minCost = insCost;
		minError = INS_ALIGN;
	    }

	    chart[i][j].cost = minCost;
	    chart[i][j].error = minError;
	}
    }

    /*
     * Backtrace and add new words to alignment columns
     */
    {
	unsigned i = refLength;
	unsigned j = hypLength;

	sub = ins = del = 0;

	while (i > 0 || j > 0) {

	    switch (chart[i][j].error) {
	    case END_ALIGN:
		assert(0);
		break;
	    case CORR_ALIGN:
		i --; j--;
		break;
	    case SUB_ALIGN:
		sub ++;
		i --; j --;
		break;
	    case DEL_ALIGN:
		/*
		 * deletion error only if alignment column doesn't 
		 * have *DELETE*
		 */
		{
		    Prob *delProb =
				aligns[sortedAligns[i-1]]->find(deleteIndex);
		    if (!delProb || *delProb == 0.0) {
			del ++;
		    }
		}
		i --;
		break;
	    case INS_ALIGN:
		ins ++;
		j --;
		break;
	    }
	}
    }

    return sub + ins + del;
}

double
WordMesh::minimizeWordError(VocabIndex *words, unsigned length,
				double &sub, double &ins, double &del,
				unsigned flags, double delBias)
{
    NBestWordInfo *winfo = new NBestWordInfo[length];
    assert(winfo != 0);

    double result =
		minimizeWordError(winfo, length, sub, ins, del, flags, delBias);

    for (unsigned i = 0; i < length; i ++) {
	words[i] = winfo[i].word;
	if (words[i] == Vocab_None) break;
    }

    delete [] winfo;

    return result;
}

double
WordMesh::minimizeWordError(NBestWordInfo *winfo, unsigned length,
				double &sub, double &ins, double &del,
				unsigned flags, double delBias)
{
    unsigned numWords = 0;
    sub = ins = del = 0.0;

    for (unsigned i = 0; i < numAligns; i ++) {

	LHashIter<VocabIndex,Prob> alignIter(*aligns[sortedAligns[i]]);

	Prob deleteProb = 0.0;
	Prob bestProb = 0.0;
	VocabIndex bestWord = Vocab_None;

	Prob *prob;
	VocabIndex word;
	while ((prob = alignIter.next(word))) {
	    Prob effectiveProb = *prob; // prob adjusted for deletion bias

	    if (word == deleteIndex) {
		effectiveProb *= delBias;
		deleteProb = effectiveProb;
	    }
	    if (bestWord == Vocab_None || effectiveProb > bestProb) {
		bestWord = word;
		bestProb = effectiveProb;
	    }
	}

	if (bestWord != deleteIndex) {
	    if (numWords < length) {
		NBestWordInfo *thisInfo =
				    wordInfo[sortedAligns[i]]->find(bestWord);
		if (thisInfo) {
		    winfo[numWords] = *thisInfo;
		} else {
		    winfo[numWords].word = bestWord;
		    winfo[numWords].invalidate();
		}

		/* 
		 * Always return the word posterior into the NBestWordInfo
		 */
		winfo[numWords].wordPosterior = bestProb;
		winfo[numWords].transPosterior = bestProb;

		numWords += 1;
	    }
	    ins += deleteProb;
	    sub += totalPosterior - deleteProb - bestProb;
	} else {
	    del += totalPosterior - deleteProb;
	}
    }
    if (numWords < length) {
	winfo[numWords].word = Vocab_None;;
	winfo[numWords].invalidate();
    }

    return sub + ins + del;
}

/*
 * Return confusion set for a given position
 */
LHash<VocabIndex,Prob> *
WordMesh::wordColumn(unsigned columnNumber) {
    if (columnNumber < numAligns) {
	return aligns[sortedAligns[columnNumber]];
    } else {
	return 0;
    }
}

/*
 * Return word info set for a given position
 */
LHash<VocabIndex,NBestWordInfo> *
WordMesh::wordinfoColumn(unsigned columnNumber) {
    if (columnNumber < numAligns) {
	return wordInfo[sortedAligns[columnNumber]];
    } else {
	return 0;
    }
}

void
WordMesh::freeThread()
{
    ChartEntryDouble**   &chart1 = TLSW_GET(alignWordsChartTLS);
    ChartEntryDouble**   &chart2 = TLSW_GET(alignAlignmentChartTLS);
    ChartEntryUnsigned** &chart3 = TLSW_GET(wordErrorChartTLS);

    unsigned &len1 = TLSW_GET(alignWordsMaxRefLengthTLS);
    unsigned &len2 = TLSW_GET(alignAlignmentMaxRefLengthTLS);
    unsigned &len3 = TLSW_GET(wordErrorMaxRefLengthTLS);

    if (chart1 != 0) freeChart<ChartEntryDouble>(chart1, len1);
    if (chart2 != 0) freeChart<ChartEntryDouble>(chart2, len2);
    if (chart3 != 0) freeChart<ChartEntryUnsigned>(chart3, len3);

    TLSW_FREE(compareAlignTLS);
    TLSW_FREE(alignWordsMaxHypLengthTLS);
    TLSW_FREE(alignWordsMaxRefLengthTLS);
    TLSW_FREE(alignWordsChartTLS);
    TLSW_FREE(alignAlignmentMaxHypLengthTLS);
    TLSW_FREE(alignAlignmentMaxRefLengthTLS);
    TLSW_FREE(alignAlignmentChartTLS);
    TLSW_FREE(wordErrorMaxHypLengthTLS);
    TLSW_FREE(wordErrorMaxRefLengthTLS);
    TLSW_FREE(wordErrorChartTLS);
}

void WordMesh::alignAlignment(MultiAlign &other_alignment, vector<int>& src2other_col_map)
{
    WordMesh &other = (WordMesh &)other_alignment;

    unsigned refLength = this->numAligns;
    unsigned hypLength = other.numAligns;

    typedef struct {
	double cost;			// minimal cost of partial alignment
	WordAlignType error;		// best predecessor
    } ChartEntry;

    /* 
     * Allocate chart statically, enlarging on demand
     */
    static unsigned maxHypLength = 0;
    static unsigned maxRefLength = 0;
    static ChartEntry **chart = 0;

    if (chart == 0 || hypLength > maxHypLength || refLength > maxRefLength) {
	/*
	 * Free old chart
	 */
	if (chart != 0) {
	    for (unsigned i = 0; i <= maxRefLength; i ++) {
		delete [] chart[i];
	    }
	    delete [] chart;
	}

	/*
	 * Allocate new chart
	 */
	maxHypLength = hypLength;
	maxRefLength = refLength;
    
	chart = new ChartEntry*[maxRefLength + 1];
	assert(chart != 0);

	for (unsigned i = 0; i <= maxRefLength; i ++) {
	    chart[i] = new ChartEntry[maxHypLength + 1];
	    assert(chart[i] != 0);
	}
    }

    /*
     * Initialize the 0'th row
     */
    chart[0][0].cost = 0.0;
    chart[0][0].error = CORR_ALIGN;

    for (unsigned j = 1; j <= hypLength; j ++) {
	unsigned pos = other.sortedAligns[j-1];
	chart[0][j].cost = chart[0][j-1].cost +
		   this->alignError(0,
			      this->totalPosterior,
			      other.aligns[pos],
			      other.columnPosteriors[pos]);
	chart[0][j].error = INS_ALIGN;
    }

    /*
     * Fill in the rest of the chart, row by row.
     */
    for (unsigned i = 1; i <= refLength; i ++) {
	unsigned main_pos = this->sortedAligns[i-1];
	double deletePenalty =
			    this->alignError(this->aligns[main_pos],
				       this->columnPosteriors[main_pos],
				       this->deleteIndex);

	chart[i][0].cost = chart[i-1][0].cost + deletePenalty;
	chart[i][0].error = DEL_ALIGN;

	for (unsigned j = 1; j <= hypLength; j ++) {
	    unsigned other_pos = other.sortedAligns[j-1];
// TODO: save local errors (substitution error)
	    double minCost = chart[i-1][j-1].cost +
		     this->alignError(this->aligns[main_pos],
				this->columnPosteriors[main_pos],
				other.aligns[other_pos],
				other.columnPosteriors[other_pos]);
	    WordAlignType minError = SUB_ALIGN;

	    double delCost = chart[i-1][j].cost + deletePenalty;
	    if (delCost < minCost) {
		minCost = delCost;
		minError = DEL_ALIGN;
	    }

	    double insCost = chart[i][j-1].cost +
		     this->alignError(0,
				this->transPosteriors[main_pos],
				other.aligns[other_pos],
				other.columnPosteriors[other_pos]);
	    if (insCost < minCost) {
		minCost = insCost;
		minError = INS_ALIGN;
	    }

	    chart[i][j].cost = minCost;
	    chart[i][j].error = minError;
	}
    }

    /*
     * Backtrace and add new words to alignment columns.
     * Store word posteriors if requested.
     */
    {
	unsigned i = refLength;
	unsigned j = hypLength;

        src2other_col_map.resize(i, -1);

	while (i > 0 || j > 0) {
	    switch (chart[i][j].error) {
	    case END_ALIGN:
		assert(0);
		break;
	    case CORR_ALIGN:
	    case SUB_ALIGN:
		/*
		 * merge all words in "other" alignment column into our own
		 */
		{
// x.sortedAligns[] maps to the actual col in the sausage
                src2other_col_map[i-1] = j-1;
		i --; j --;
                }

		break;
	    case DEL_ALIGN:
		i --;
		break;
	    case INS_ALIGN:
		j --;
		break;
	    }
	}
    }
}

