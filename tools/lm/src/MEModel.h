/*
 * MEModel.h --
 *	Maximum entropy Ngram models
 *
 *  Created on: Apr 5, 2010
 *      Author: tanel
 *
 * Copyright (c) 2009-2013 Tanel Alumae.  All Rights Reserved.
 *
 * @(#)$Header: /home/speech/stolcke/project/srilm/devel/lm/src/RCS/DFNgram.h,v 
1.5 1995/11/07 08:37:12 stolcke Exp $
 *
 */

#ifndef MEMODEL_H_
#define MEMODEL_H_

#include "LM.h"
#include "LHash.h"
#include "Trie.h"
#include "Ngram.h"

#include "hmaxent.h"

typedef unsigned int NodeIndex;


class MEModel: public LM {
public:
	MEModel(Vocab &vocab, unsigned order = defaultNgramOrder);
	virtual ~MEModel();

	/*
	 * LM interface
	 */
	virtual LogP wordProb(VocabIndex word, const VocabIndex *context);
	virtual void *contextID(const VocabIndex *context, unsigned &length) {
		return contextID(Vocab_None, context, length);
	}

	virtual void *contextID(VocabIndex word, const VocabIndex *context,
			unsigned &length);


	virtual Boolean read(File &file, Boolean limitVocab = false);
	virtual Boolean write(File &file);

	virtual Ngram *getNgramLM();

	virtual Boolean &skipOOVs() {
		return _skipOOVs;
	}


	/*
	 * Estimation
	 */
	virtual Boolean estimate(NgramCounts<FloatCount> &stats,  double alpha, double sigma2) {
		return _estimate(stats, alpha, sigma2);
	}
	virtual Boolean estimate(NgramStats &stats,  double alpha, double sigma2) {
		return _estimate(stats, alpha, sigma2);
	}

	/**
	 * Performs adaption according to [Chelba and Acero, 2004] --
	 * Adaptation of maximum entropy capitalizer: Little data can help a lot
	 */
	virtual Boolean adapt(NgramCounts<FloatCount> &stats,  double alpha, double sigma2) { return _adapt(stats, alpha, sigma2); }
	virtual Boolean adapt(NgramStats &stats,  double alpha, double sigma2) { return _adapt(stats, alpha, sigma2); }
	void setMaxIterations(unsigned max) { this->maxIterations = max; }
protected:
	template <class CountT> void modifyCounts(NgramCounts<CountT> &stats);
	template <class CountT> Boolean _estimate(NgramCounts<CountT> &stats,  double alpha, double sigma2);
	template <class CountT> Boolean _adapt(NgramCounts<CountT> &stats, double alpha, double sigma2);
	template <class CountT> hmaxent::data_t *createDataFromCounts(NgramCounts<CountT> &stats);
	unsigned int order;
	double alpha;
	double sigma2;
	Boolean _skipOOVs;
	hmaxent::model *m;
	Trie<VocabIndex, NodeIndex> reverseContextIndex;
	Trie<VocabIndex, NodeIndex> contextIndex;
	LHash<VocabIndex, size_t> vocabMap;
	unsigned maxIterations;
	VocabIndex otherIndex;
	void clear();
};

#endif /* MEMODEL_H_ */
