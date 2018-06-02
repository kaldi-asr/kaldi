/*
 * MEModel.cc --
 *	Maximum entropy Ngram models
 *
 *  Created on: Apr 5, 2010
 *      Author: tanel
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2009-2013 Tanel Alumae, Microsoft Corp. 2013-2016.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/MEModel.cc,v 1.13 2016/04/09 06:53:01 stolcke Exp $";
#endif

#include <assert.h>
#include <limits>

#include "Trie.h"
#include "LHash.cc"

#include "hmaxent.h"

#include "MEModel.h"
#include "Ngram.h"
#include "Array.h"

using namespace hmaxent;

#define DEBUG_NGRAM_HITS 2
#define DEBUG_PROGRESS 2

/* INSTANTIATE_LHASH(VocabIndex, size_t); */


int
simple_sort(VocabIndex k1 , VocabIndex k2)
{
    return k1-k2;
}

Trie<VocabIndex, NodeIndex>
reverseTrie(Trie<VocabIndex, NodeIndex> &inputTrie, unsigned order)
{
    Trie<VocabIndex, NodeIndex> result;
    makeArray(VocabIndex, ngram, order + 1);
    makeArray(VocabIndex, ngram2, order + 1);
    Boolean foundP;
    for (unsigned j=0; j <= order; j++) {
	TrieIter2<VocabIndex, NodeIndex> trieIter(inputTrie, ngram, j-1);
	while (Trie<VocabIndex, NodeIndex> *trie = trieIter.next()) {
	    Vocab::copy(ngram2, ngram);
	    Vocab::reverse(ngram2);
	    *(result.insert(ngram2, foundP)) = trie->value();
	}
    }
    return result;
}

/*
 * Inititializes Maximum Entropy LM
 */
MEModel::MEModel(Vocab & vocab, unsigned order)
    : LM(vocab), order(order), _skipOOVs(false), reverseContextIndex(), contextIndex(), vocabMap(),
      maxIterations(1000)
{
    if (order < 1) {
	order = 1;
    }
}

MEModel::~MEModel()
{
    clear();
}

LogP
MEModel::wordProb(VocabIndex word, const VocabIndex *context)
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
	    if (running() && debug(DEBUG_NGRAM_HITS)) {
		dout() << "[OOV context]";
	    }
	    return LogP_Zero;
	}
    }

    Boolean foundP;
    size_t * outcomeId = vocabMap.find(word, foundP);
    if (!foundP) {
	if (running() && debug(DEBUG_NGRAM_HITS)) {
	    dout() << "[OOV]";
	}
	return LogP_Zero;
    }

    unsigned length;
    Trie<VocabIndex, NodeIndex> *trieNode = (Trie<VocabIndex, NodeIndex> *) contextID(word, context, length);
    if (running() && debug(DEBUG_NGRAM_HITS)) {
	dout() << "[" << length + 1 << "gram]";
    }

    LogP result = m->log_prob_context(trieNode->value(), *outcomeId) / M_LN10;
    return result;
}

void
MEModel::clear()
{
    if (m != 0) {
	delete(m);
    }
    contextIndex.clear();
    reverseContextIndex.clear();
    vocabMap.clear();
}

void *
MEModel::contextID(VocabIndex word, const VocabIndex *context, unsigned  & length)
{
    Trie<VocabIndex, NodeIndex> *trieNode = &reverseContextIndex;
    unsigned int i = 0;
    while (i < order - 1 && context[i] != Vocab_None) {
    	Trie<VocabIndex, NodeIndex> *next = trieNode->findTrie(context[i]);
        if (next) {
            trieNode = next;
            i++;
        } else {
	    break;
        }
    }

    length = i;
    return (void *)trieNode;
}

Boolean
MEModel::read(File & file, Boolean limitVocab)
{
    char *line;
    line = file.getline();
    if (line && strncmp(line, "H-MAXENT 0.1", 12) == 0) {
	long int order, num_outcomes, num_contexts, num_features;
	line = file.getline();
	if (line && sscanf(line, "# %ld %ld %ld %ld", &order, &num_outcomes, &num_contexts, &num_features)) {
	    if ((order <= 0) || (num_contexts <= 0) || (num_features <= 0)) {
		// Illegal value
		return false;
	    }
	    if ((unsigned)order > maxNgramOrder) {
		// Illegal value
		return false;
	    }
	    // See if so large right at verge of overflow
	    if ((long int)(num_contexts + 1) <= 0) {
		// Used highest value possible - too large
		return false;
	    }
	    if ((long int)(num_features + 1) <= 0) {
		// Used highest value possible - too large
		return false;
	    }
	    structure_t *structure = new structure_t();
	    structure->order = order;
	    this->order = order;
	    structure->num_outcomes = num_outcomes;
	    structure->feature_contexts = new vector<feature_context_t>();
	    // @kw false positive: SV.TAINTED.ALLOC_SIZE (num_features)
	    structure->feature_outcome_ids = new valarray<size_t>(num_features);

	    if (debug(2)) {
		cerr << "Reading " << num_contexts << " contexts..." << endl;
	    }
	    unsigned context_id = 0;
	    unsigned feature_id = 0;
	    double *weights = new double[num_features];

	    // Initialize to 0 to remove any chance of uninitialized values
	    for (int i = 0; i < num_features; i++) {
		weights[i] = 0.0;
	    }

	    VocabString words[1 + maxNgramOrder + 1 + 1];
	    VocabIndex wids[1 + maxNgramOrder + 1 + 1];
	    for (int i = 0; i < num_contexts; i++) {
		line = file.getline();

		unsigned howmany = Vocab::parseWords(line, words, order);
		unsigned num_features = atoi(words[0]);

		unsigned contextLength = howmany - 1;
		vocab.addWords(words + 1, wids, howmany - 1);
		wids[contextLength] = Vocab_None;

		*(contextIndex.insert(wids)) = context_id;

		feature_context_t fc;
		fc.length = contextLength;
		if (contextLength > 0) {
		    unsigned int *ptr = contextIndex.find(wids + 1);
		    fc.parent_id = ptr?*ptr:0;
		} else {
		    fc.parent_id = 0;
		}
		fc.start_index = feature_id;
		fc.num_features = num_features;
		for (unsigned j = 0; j < num_features; j++) {
		    line = file.getline();
		    howmany = Vocab::parseWords(line, words, 2);

		    if (howmany != 2) {
			file.position() << "<word> <weight> expected" << endl;
			delete[] weights;
			delete structure->feature_outcome_ids;
			structure->feature_outcome_ids = 0;
			delete structure->feature_contexts;
			structure->feature_contexts = 0;
			delete structure;
			return false;
		    }
		    VocabIndex wid = vocab.addWord(words[0]);
		    Boolean foundP;
		    size_t *outcomeId = vocabMap.insert(wid, foundP);
		    if (!foundP) {
			*outcomeId = vocabMap.numEntries() - 1;
		    }
		    (*structure->feature_outcome_ids)[feature_id] = *outcomeId;
		    weights[feature_id] = atof(words[1]);
		    feature_id++;
		}
		structure->feature_contexts->push_back(fc);
		context_id++;
	    }
	    m = new model(structure);
	    valarray<double> *params = m->get_params();

	    for (int i = 0; i < num_features; i++) {
		(*params)[i] = weights[i];
	    }
	    delete [] weights;
	    reverseContextIndex = reverseTrie(contextIndex, order);

	    return true;
	}
    }

    file.position() << "format error in H-MAXENT file" << endl;
    return false;
}

Boolean
MEModel::write(File & file)
{
    LHashIter<VocabIndex, size_t> i(vocabMap);
    size_t *outcomeId;
    VocabIndex wid;
    while ((outcomeId = i.next(wid))) {
	*(reverseContextIndex.insert(*outcomeId)) = wid;
    }
    file.fprintf("H-MAXENT 0.1\n");
    structure_t * structure = m->get_structure();
    file.fprintf("# %ld %ld %ld %ld\n",
		(long)structure->order,
		(long)structure->num_outcomes,
		(long)structure->feature_contexts->size(),
		(long)structure->feature_outcome_ids->size());
    valarray<double> *params = m->get_params();
    makeArray(VocabIndex, ngram, order + 1);
    for (unsigned j=0; j < order; j++) {
	TrieIter2<VocabIndex, NodeIndex> trieIter(contextIndex, ngram, j);
	while (Trie<VocabIndex, NodeIndex> *trie = trieIter.next()) {
	    feature_context_t *fc = &(*structure->feature_contexts)[trie->value()];
	    file.fprintf("%ld", (long)fc->num_features);
	    for (unsigned i = 0; i < j; i++) {
		file.fprintf(" %s", vocab.getWord(ngram[i]));
	    }
	    file.fprintf("\n");
	    for (unsigned i = fc->start_index; i < fc->start_index + fc->num_features; i++) {
		unsigned int *ptr = reverseContextIndex.find((*structure->feature_outcome_ids)[i]);
		file.fprintf("%s %.*lg\n", ptr?vocab.getWord(*ptr):"NULL",
					  numeric_limits<double>::digits10 + 1, (*params)[i]);
	    }
	    file.fprintf("\n");
	}
    }
    return true;
}

/**
 * Converts MaxEnt model to ngram LM.
 * See Jun Wu's thesis, chap. 7.1.2  Mapping ME N-gram Model Parameters to ARPA Back-off Model Parameters
 * http://www.cs.jhu.edu/~junwu/publications/ch7.pdf
 */
Ngram *
MEModel::getNgramLM()
{
    Ngram *ngramLM = new Ngram(vocab, order);
    assert(ngramLM != 0);

    *ngramLM->insertProb(vocab.ssIndex(), &Vocab_None) = LogP_Zero;

    LHashIter<VocabIndex, size_t> i(vocabMap);
    size_t *outcomeId;
    VocabIndex wid;
    while ((outcomeId = i.next(wid))) {
	*(reverseContextIndex.insert(*outcomeId)) = wid;
    }

    valarray<double> param_sums = m->param_sums();
    valarray<double> *log_zetas = m->lognormconst();
    structure_t * structure = m->get_structure();

    makeArray(VocabIndex, ngram, order + 1);
    for (unsigned j=0; j < order; j++) {
	TrieIter2<VocabIndex, NodeIndex> trieIter(contextIndex, ngram, j);
	while (Trie<VocabIndex, NodeIndex> *trie = trieIter.next()) {
	    feature_context_t *fc = &(*structure->feature_contexts)[trie->value()];

	    makeArray(VocabIndex, context, j+1);
	    for (unsigned i = 0; i < j; i++) {
		context[j - i - 1] = ngram[i];
	    }
	    context[j] = Vocab_None;
	    LogP logBow = (*log_zetas)[fc->parent_id] - (*log_zetas)[trie->value()];

	    *ngramLM->insertBOW(context) = logBow/M_LN10;

	    for (unsigned i = fc->start_index; i < fc->start_index + fc->num_features; i++) {
		LogP logF = (param_sums[i] - (*log_zetas)[trie->value()])/M_LN10;

		unsigned int *ptr = reverseContextIndex.find((*structure->feature_outcome_ids)[i]);
		if (ptr) {
		    VocabIndex wordId = *ptr;

		    *ngramLM->insertProb(wordId, context) = logF;
		} // else unexpected error
	    }
	}
    }

    return ngramLM;
}

/**
 * Subtract higher order counts from lower order counts
 */
template <class CountT>
void
MEModel::modifyCounts(NgramCounts<CountT> &stats)
{
    makeArray(VocabIndex, ngram, order + 1);

    // remove counts containing  <unk> if <unk> is not in LM
    for (unsigned  i = order; i > 0; i--) {
	NgramCountsIter<CountT> countIter(stats, ngram, i);

	while (CountT *count = countIter.next()) {
	    for (unsigned j = 0; j < i; j++) {
		if (!vocab.unkIsWord() && (ngram[j] == vocab.unkIndex())) {
		    stats.removeCount(ngram);
		    break;
		}
	    }
	}
    }

    // subtract higher order counts from lower order counts
    for (unsigned j = 2; j <= order; j++) {
	NgramCountsIter<CountT> ngramIter(stats, ngram, j);

	while (CountT *n = ngramIter.next()) {
	    (*(stats.findCount((ngram + 1)))) -= *n;
	}
    }
}

template <class CountT>
data_t *
MEModel::createDataFromCounts(NgramCounts<CountT> &stats)
{
    makeArray(VocabIndex, ngram, order + 1);
    makeArray(VocabIndex, ngram2, order + 1);

    data_t * data = new data_t();

    unsigned num_counts = 0;

    Boolean warned = false;

    NgramCounts<CountT> nodeCounts(vocab, order);
    for (unsigned j = 0; j < order; j++) {
	NgramCountsIter<CountT> ngramIter(stats, ngram, j);

	while (CountT *n = ngramIter.next()) {
	    Boolean foundP;
	    NodeIndex *featureContextId = contextIndex.find(ngram, foundP);

	    NgramCountsIter<CountT> iter(stats, ngram, ngram2, 1, simple_sort);

	    while (CountT * count = iter.next()) {
		VocabIndex wid = ngram2[0];
		if ((vocab.unkIsWord() || (wid != vocab.unkIndex())) && !vocab.isNonEvent(wid)) {
		    if (featureContextId != 0) {
			*nodeCounts.insertCount(ngram, wid) += *count;
			num_counts += 1;
		    } else {
			if (!warned) {
			    cerr << "WARNING: Data contains n-grams that cannot be properly mapped the nodes of the Maximum Entropy model structure;" << endl;
			    cerr << "         If you are adapting a prior model, use also adaptation data (with weight 0) for creating the prior model" << endl;
			    warned = true;
			}
		    }
		}
	    }
	}
    }

    if (debug(DEBUG_PROGRESS)) {
	    cerr << "Creating count contexts..." << endl;
    }

    data->count_outcome_ids = new valarray<size_t>(num_counts);
    data->counts = new valarray<float>(num_counts);
    data->count_contexts = new vector<count_context_t>();

    unsigned count_index = 0;
    for (unsigned j = 0; j < order; j++) {
	NgramCountsIter<CountT> ngramIter(nodeCounts, ngram, j);

	while (CountT *n = ngramIter.next()) {
	    Boolean foundP;

	    NgramCountsIter<CountT> iter(nodeCounts, ngram, ngram2, 1, simple_sort);

	    NodeIndex *featureContextId = contextIndex.find(ngram, foundP);
	    if (foundP) {
		count_context_t count_context;
		count_context.feature_context_id = *featureContextId;
		count_context.start_index = count_index;

		while (CountT *count = iter.next()) {
		    if (*count > 0) {
			VocabIndex wid = ngram2[0];
			(*data->counts)[count_index] = (float)*count;
			(*data->count_outcome_ids)[count_index] = *(vocabMap.find(wid));
			count_index ++;
		    }
		}
		count_context.num_counts = count_index - count_context.start_index;
		if (count_context.num_counts > 0) {
		    data->count_contexts->push_back(count_context);
		}
	    }
	}
    }

    if (debug(10)) {
	File o("-", "w");
	nodeCounts.write(o, 0);
    }

    return data;
}

template <class CountT>
Boolean
MEModel::_estimate(NgramCounts<CountT> &stats, double alpha, double sigma2)
{
    //modifyCounts(stats, mincounts);
    // count all unigrams
    VocabIter vocabIter(vocab, true);
    VocabIndex wid;
    makeArray(VocabIndex, ngram, order + 1);
    makeArray(VocabIndex, ngram2, order + 1);
    unsigned num_features = 0;

    modifyCounts(stats);

    Boolean foundP;
    if (debug(DEBUG_PROGRESS)) {
	cerr << "Counting counts of order 1 " << endl;
    }

    unsigned num_outcomes = 0;
    unsigned num_contexts = 1;
    for (VocabIndex wid = 0; wid <= vocab.highIndex(); wid++) {
	if ((vocab.unkIsWord() || (wid != vocab.unkIndex())) &&
	    wid != vocab.pauseIndex() &&
	    !vocab.isNonEvent(wid))
	{
	    *(vocabMap.insert(wid)) = num_outcomes;
	    num_features++;
	    num_outcomes++;
	}
    }
    ngram[0] = Vocab_None;
    *(contextIndex.insert(ngram,  foundP)) = 0;

    // now count all higher order ngrams
    for (unsigned j = 2; j <= order; j++) {
	if (debug(1)) {
	    cerr << "Counting counts of order " << j << endl;
	}
	NgramCountsIter<CountT> countIter(stats, ngram, j-1, simple_sort);

	while (countIter.next()) {
	    NgramCountsIter<CountT> iter2(stats, ngram, ngram2, 1, simple_sort);

	    Boolean hasCounts = false;
	    while (CountT *count = iter2.next()) {
		if (!vocab.isNonEvent(ngram2[0])) {
		    num_features++;
		}
		hasCounts = true;
	    }
	    if (hasCounts) {
		NodeIndex *context_id = contextIndex.insert(ngram,  foundP);
		if (!foundP) {
		    *context_id = num_contexts;
		    num_contexts++;
		}
	    }
	}
    }

    if (debug(10)) {
	cerr << "Contexts:" << endl;
	contextIndex.dump();
    }


    structure_t *structure = new structure_t();
    structure->order = order;
    structure->num_outcomes = num_outcomes;
    structure->feature_contexts = new vector<feature_context_t>(num_contexts);
    structure->feature_outcome_ids = new valarray<size_t>(num_features);


    unsigned feature_id = 0;
    feature_context_t *feature_context = &(*structure->feature_contexts)[0];
    feature_context->length = 0;
    feature_context->start_index = 0;
    feature_context->num_features = num_outcomes;

    vocabIter.init();
    for (VocabIndex wid = 0; wid <= vocab.highIndex(); wid++) {
	if ((vocab.unkIsWord() || (wid != vocab.unkIndex())) &&
	    wid != vocab.pauseIndex() &&
	    !vocab.isNonEvent(wid))
	{
	    (*structure->feature_outcome_ids)[feature_id] = *(vocabMap.find(wid));

	    feature_id++;
	}
    }

    if (debug(DEBUG_PROGRESS)) {
	cerr << "Creating feature contexts..." << endl;
    }


    unsigned context_id = 1;

    for (unsigned j = 1; j < order; j++) {
	if (debug(DEBUG_PROGRESS)) {
		cerr << "Indexing contexts of order " << j+1 << endl;
	}

	TrieIter2<VocabIndex, NodeIndex> iter(contextIndex, ngram, j, simple_sort);

	while (Trie<VocabIndex, NodeIndex> *trie = iter.next()) {
	    unsigned start_index = feature_id;

	    context_id = trie->value();

	    feature_context_t *feature_context = &(*structure->feature_contexts)[context_id];
	    feature_context->length = j;
	    feature_context->parent_id = *(contextIndex.find(ngram + 1, foundP));
	    feature_context->start_index = start_index;

	    NgramCountsIter<CountT> iter2(stats, ngram, ngram2, 1, simple_sort);

	    while (CountT *count = iter2.next()) {
		if (!vocab.isNonEvent(ngram2[0])) {
		    (*structure->feature_outcome_ids)[feature_id] = *(vocabMap.find(ngram2[0]));
		    feature_id++;
		}
	    }
	    feature_context->num_features = feature_id - feature_context->start_index;
	}
    }


    m = new model(structure);

    data_t * data = createDataFromCounts(stats);

    m->set_sigma2(data->counts->sum() * sigma2);
    m->set_c1(alpha/data->counts->sum());
    m->set_max_iters(maxIterations);

    m->fit(data);
    delete data;


    if (debug(DEBUG_PROGRESS)) {
	cerr << "Creating reverse context index..." << endl;
    }
    reverseContextIndex = reverseTrie(contextIndex, order);
    return true;
}

template <class CountT>
Boolean
MEModel::_adapt(NgramCounts<CountT> &stats, double alpha, double sigma2)
{
    modifyCounts(stats);
    data_t *data = createDataFromCounts(stats);
    m->init_prior_params();
    *(m->get_params()) = 0.0;

    m->set_sigma2(data->counts->sum() * sigma2);
    m->set_c1(alpha/data->counts->sum());
    m->set_max_iters(maxIterations);

    m->fit(data);
    delete data;

    if (debug(2)) {
	cerr << "Creating reverse context index..." << endl;
    }
    reverseContextIndex = reverseTrie(contextIndex, order);
    return true;
}

