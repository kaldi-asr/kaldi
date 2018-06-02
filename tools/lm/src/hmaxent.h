/*
 * hmaxent.h
 *
 *  Created on: Mar 16, 2010
 *      Author: tanel
 *
 * Copyright (c) 2009-2013 Tanel Alumae.  All Rights Reserved.
 *
 * @(#)$Header: /home/speech/stolcke/project/srilm/devel/lm/src/RCS/DFNgram.h,v 
1.5 1995/11/07 08:37:12 stolcke Exp $
 */


#ifndef HMAXENT_HPP_
#define HMAXENT_HPP_

#include <cstdlib>
#include <valarray>
#include <vector>
#include <map>

namespace hmaxent {

struct feature_context_t {
    unsigned length;
    size_t parent_id;
    size_t start_index;
    size_t num_features;
};

struct count_context_t {
    size_t feature_context_id;
    size_t start_index;
    size_t num_counts;
};

struct structure_t {
    size_t order;
    size_t num_outcomes;
    size_t num_sets;
    std::valarray<size_t> *feature_outcome_ids;
    std::vector<feature_context_t> *feature_contexts;
};

struct data_t {
    std::valarray<size_t> *count_outcome_ids;
    std::valarray<float> *counts;
    std::vector<count_context_t> *count_contexts;
};

class model {

public:
	model(structure_t *structure);
	~model();
	std::valarray<double> *lognormconst();
	std::valarray<double> expectations();
	double dual();
	std::valarray<double> grad();
	double fit(data_t *data);
	double sum_logpmf();

	void log_pmf_context(size_t feature_context_id, std::valarray<double> &result);
	double log_prob_context(size_t feature_context_id, size_t outcome_id);

	std::valarray<double> param_sums();

	std::valarray<double> *get_params() { return params; }
	structure_t *get_structure() { return structure; }
	void clear_cache();
	void set_sigma2(double sigma2) { this->sigma2 = sigma2; }
	void set_c1(double c1) { this->c1 = c1; }
	void set_max_iters(unsigned max_iters) { this->max_iters = max_iters; }

	/**
	 * Initialize prior means from current parameters
	 */
	void init_prior_params() { this->prior_params = new std::valarray<double>(*params); };
private:
	void ensure_exp_params();

	data_t *data;
	std::valarray<double> *p_tilde_context;

	structure_t *structure;

	std::valarray<double> logZ;
	std::valarray<double> exp_params;
	double sigma2;
	double c1;
	bool exp_params_computed;
	bool logZ_computed;
	std::valarray<double> k;
	std::valarray<size_t> feature_parent_ids;
	std::valarray<size_t> *num_order_contexts;
	std::valarray<double> *params;
	std::valarray<double> *prior_params;
	unsigned max_iters;
};


template <class T>
inline std::ostream&
operator<< (std::ostream &out, const std::valarray<T> &v) {
    out << '[';

    for (std::size_t i = 0; (i < v.size()) && (i < 40); ++i) {
        out << v [i];
        if (i < v.size () - 1)
            out << ',';
    }
    if (v.size() > 40) {
    	out << " ... ";
    }

    return out << ']';
}

inline std::ostream&
operator<< (std::ostream &out, const std::vector<feature_context_t> &v){
    out << '[';

    for (std::size_t i = 0; (i < v.size()) && (i < 30); ++i) {
        out << "[" << (int)v[i].length << ":" << v[i].parent_id << ":" << v[i].start_index << ":" << v[i].num_features << "]";
        if (i < v.size () - 1)
            out << ',';
    }
    if (v.size() > 30) {
    	out << " ... ";
    }

    return out << ']';
}

}

#endif /* HMAXENT_HPP_ */

