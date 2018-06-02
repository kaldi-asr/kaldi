/*
 * ProductVocab.cc --
 *	Product vocabulary implementation
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/flm/src/ProductVocab.cc,v 1.15 2010/06/02 05:51:57 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "FNgramStats.h"		// define FNgramCount
#include "FNgramSpecs.h"
#include "ProductVocab.h"

#include "LHash.cc"
#include "Array.cc"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_LHASH(VocabIndex,Array<VocabIndex>);
#endif


/*
 * Initialize the word Vocab making sure indices will fit in the lower-order
 * bits.  Let tag indices start at one, so 0 can be used for the untagged
 * case.
 */
ProductVocab::ProductVocab(VocabIndex start, VocabIndex end)
    : Vocab(start, end)
{
}

void
ProductVocab::memStats(MemStats &stats) const
{
    Vocab::memStats(stats);
    // TODO: need to finish this function.
}

// addWord(): dynamically allocate indices for novel combinations of
// known factors, and allocate new factors.
VocabIndex
ProductVocab::addWord(VocabString name)
{
    // TODO: right now, we add the word in product form to the vocab,
    // thereby using extra memory. At some point, just store the indices
    // to the various factors and re-construct the word strings on the
    // fly based on the factors.

    // first add the entire word as an item, in product form. 
    VocabIndex wordId =  Vocab::addWord(name);

    // next, break it out into its factors and make sure each factor is
    // added. 
    VocabString word_factors[maxNumParentsPerChild+1];
    fvocab.loadWordFactor(name, word_factors);

    // add the words to the factored Vocabulary.
    unsigned numTags = fvocab.tagPosition.numEntries();
    Array<VocabIndex> wid_factors(0, numTags);

    fvocab.addWords(word_factors, wid_factors.data(), numTags);

    // check for start of sentence/end of sentence in word condition
    // in word position. Note: loadWordFactor() could do this, but
    // this is probably a bit faster since it checks the int index
    // rather than doing string compares.
    if (FNGRAM_WORD_TAG_POS < numTags) {
	if (wid_factors[FNGRAM_WORD_TAG_POS] == ssIndex()) {
	    for (unsigned i = 0; i< numTags; i++) {
		if (wid_factors[i] == fvocab.unkIndex() ||
		    wid_factors[i] == fvocab.nullIndex)
		    wid_factors[i] = fvocab.ssIndex();
	    }
	} else if (wid_factors[FNGRAM_WORD_TAG_POS] == seIndex()) {
	    for (unsigned i = 0; i< numTags; i++) {
		if (wid_factors[i] == fvocab.unkIndex() ||
		    wid_factors[i] == fvocab.nullIndex)
		    wid_factors[i] = fvocab.seIndex();
	    }
	}
    }

    unsigned j;
    for (j = 0; j < numTags && wid_factors[j] != Vocab_None; j ++) {
      fvocab.addTagWord(j, wid_factors[j]);
    }
    if (j == 0) {
      cerr << "No known factors found in " << name << endl;
      // don't create map entry so loadWidFactors() can try again.
    } else {
      // finally create a map from wordId to its factored form.
      *productIdToFactorIds.insert(wordId) = wid_factors;
    }

    return wordId;
}

VocabString
ProductVocab::getWord(VocabIndex index)
{
    // TODO: right now, we add the word in product form to the vocab,
    // thereby using extra memory. At some point, just store the indices
    // to the various factors and re-construct the word strings on the
    // fly based on the factors.

    return Vocab::getWord(index);
}

//
// getIndex(): dynamically allocate indices for combinations of
// factors.
//
// Possible Situations:
//  1)  word is known => all factors are already known
//        a) return appropriate index for word
//        b) create mapping from word to its factors
// 2)  word is unknown && all factors are already known
//        a) return new index for word
//        b) create mapping from new word index to factors,
//           word factor has unk index
// 3)  word is unknown && some factors are known, others are not
//        a) return a new index for word 
//        b) create mapping from new word index to factors,
//           unknown factors have unk index
// 4)  word is unknown && all factors are unknown
//        a) create a new index for word
//        b) create mapping from new word index to factrs,
//           all factors have unk index  
VocabIndex
ProductVocab::getIndex(VocabString name, VocabIndex unkIndex)
{
    // break it out into its factors and make sure each factor is
    // added. 

    VocabString word_factors[maxNumParentsPerChild+1];
    fvocab.loadWordFactor(name, word_factors);

    // add the words to the factored vocabulary.
    unsigned numTags = fvocab.tagPosition.numEntries();
    Array<VocabIndex> wid_factors(0, numTags);

    fvocab.getIndices(word_factors, wid_factors.data(), numTags, unkIndex);

    // check for start of sentence/end of sentence in word condition
    // in word position. Note: loadWordFactor() could do this, but
    // this is probably a bit faster since it checks the int index
    // rather than doing string compares.
    if (FNGRAM_WORD_TAG_POS < numTags) {
	if (wid_factors[FNGRAM_WORD_TAG_POS] == ssIndex()) {
	    for (unsigned i = 0; i< numTags; i++) {
		if (wid_factors[i] == fvocab.unkIndex() ||
		    wid_factors[i] == fvocab.nullIndex)
		    wid_factors[i] = fvocab.ssIndex();
	    }
	} else if (wid_factors[FNGRAM_WORD_TAG_POS] == seIndex()) {
	    for (unsigned i = 0; i< numTags; i++) {
		if (wid_factors[i] == fvocab.unkIndex() ||
		    wid_factors[i] == fvocab.nullIndex)
		    wid_factors[i] = fvocab.seIndex();
	    }
	}
    }

    // first add the entire word as an item, in product form. 
    VocabIndex wordId = Vocab::addWord(name);

    // finally create a map from wordId to its factored form.
    *productIdToFactorIds.insert(wordId) = wid_factors;

    return wordId;
}


//
// loads up the factors for a given product wid, returning false
// if wid is not found in vocab, true otherwise.
Boolean
ProductVocab::loadWidFactors(VocabIndex word, VocabIndex *factors)
{
    Array<VocabIndex> *res = productIdToFactorIds.find(word);

    // The word might not have an entry in productIdToFactorIds if it
    // was created during initialization of the Vocab base object.
    // Recreate it using our own addWord() to create the necessary mapping.
    // Note that we do this here rather than in the constructor since the
    // factors are not typically known until associated LMs have been read.
    if (res == NULL) {
	VocabString wordName = getWord(word);
	assert(wordName != NULL);

	(void)addWord(wordName);
        res = productIdToFactorIds.find(word);
    }

    assert(res != NULL);		// this shouldn't happen

    for (unsigned j = 0; j < res->size(); j ++) {
      factors[j] = (*res)[j];
    }
    return true;
}

