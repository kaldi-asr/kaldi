/*
 * hmaxent.cpp
 *
 *  Created on: Mar 16, 2010
 *      Author: tanel
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2009-2012 Tanel Alumae.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/hmaxent.cc,v 1.6 2014-04-22 08:46:32 stolcke Exp $";
#endif

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <typeinfo>
#include <sstream>
#include <fstream>
#include <ios>
#include <iostream>

#include <valarray>
#include <vector>
#include <numeric>
#include <limits>
#include <algorithm>

#if defined(_OPENMP) && defined(_MSC_VER)
#include <omp.h>
#endif

#ifdef HAVE_LIBLBFGS
#include <lbfgs.h>

#ifdef _MSC_VER
#pragma comment(lib, "lbfgs.lib")
#endif
#endif

#include "Prob.h"

#include "hmaxent.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::valarray;
using std::slice_array;
using std::slice;

using namespace hmaxent;

/**
 * Compute the log of the sum of exponentials log(e^{a_1}+...e^{a_n})
 *   of the components of the array a, avoiding numerical overflow.
 */
template <class T>
T logsumexp(const valarray<T> &a) {
    T a_max = a.max();
    return a_max + log((exp(a-a_max)).sum());
}

template <class T>
T dot(valarray<T> &a1, valarray<T> &a2) {
	return (a1*a2).sum();
}

template <class T>
T norm(valarray<T> &a) {
	return sqrt((a*a).sum());
}

template <class T> T square(const T& x) { return x*x; }

valarray<size_t> slice_to_valarray(slice &s) {
	valarray<size_t> result(s.size());
	for (unsigned i = 0;  i < s.size(); i++) {
		result[i] = s.start() + i * s.stride();
	}
	return result;
}

/* logaddexp returns ln(exp(a) + exp(b)), but works even when both exp's
   would overflow and/or underflow  */
template <class T>
T logaddexp(T a, T b)
{
  T l, s;

  if (a > b) {
    l = a; s = b;
  } else {
    l = b; s = a;
  }

  /* log(a+b) = log(a) + log(1 + b/a)
              = log(a) + log(1 + exp(log(b)-log(a)))
              = l + log(1 + exp(s-l)) */
  if (s-l < -50.0) {
    /* exp(s-l) reduces to 0 */
    return l;
  } else {
    return (l + log(1.0 + exp(s-l)));
  }
}

template <class T>
T max(const T &a, const T &b) {
	return a > b ? a : b;
}

template <class T>
T min(const T &a, const T &b) {
	return a < b ? a : b;
}

/*
 * Calculate log(exp(a) - exp(b)), assume a > b
 */
template <class T>
valarray<T> logminusexp(const valarray<T> &a, const valarray<T> &b) {
	return b + log(exp(a-b) - 1);
}

template <class T>
void check_nans(valarray<T> &a, const std::string &name) {
	size_t num_nans = 0;
	size_t num_infs = 0;
	for (size_t i = 0; i< a.size(); i++) {
		if (isnan(a[i])) {
			num_nans++;
		}
		if (!isfinite(a[i])) {
			num_infs++;
		}
	}
	cerr << "  No of NaNs in " << name << ": " << num_nans << ", No infs: " << num_infs << endl;
}


template <class T>
std::valarray<size_t> search_sorted(const valarray<T> &data_from, const valarray<T> &data_indexes, const slice &source, const slice &values) {

	size_t N = source.size();
	std::valarray<size_t> result(values.size());
	size_t low = 0;
	size_t delta_source = source.start();
	size_t delta_values = values.start();
	for (size_t i = 0; i < values.size(); i++) {
		size_t high = N;
		T val = data_indexes[delta_values + i];
		while (low < high) {
			size_t mid = low + ((high - low) / 2);
			if (data_from[delta_source + mid] < val) {
				low = mid + 1;
			} else {
                //can't be high = mid-1: here A[mid] >= value,
                //so high can't be < mid if A[mid] == value
                high = mid;
			}
		}
		result[i] = low;
		low += 1;
	}
	return result;
}


/**
 * Searches sorted array for value
 * Return position of value, or position to insert value if value is not in the array
 */
template <class T>
size_t lower_bound(const valarray<T> &sorted_array, size_t first, size_t last, T value) {
	size_t mid = first;
	while (first <= last) {
	   mid = (first + last) / 2;  // compute mid point.
	   if (value > sorted_array[mid])
		   first = mid + 1;  // repeat search in top half.
	   else if (value < sorted_array[mid])
		   last = mid - 1; // repeat search in bottom half.
	   else
		   return mid;     // found it. return position /////
	}
	return mid;
}

model::model(structure_t *structure) :
	structure(structure), logZ(structure->feature_contexts->size()), exp_params(structure->feature_outcome_ids->size()), sigma2(0), c1(0),
	exp_params_computed(false), logZ_computed(false), k(0.0, structure->feature_outcome_ids->size()),
	feature_parent_ids(structure->feature_outcome_ids->size()), prior_params(0), max_iters(1000) {

	params = new valarray<double>(0.0, structure->feature_outcome_ids->size());

	num_order_contexts = new valarray<size_t>(structure->order - 1);
	for (size_t i = 1; i < structure->feature_contexts->size(); i++) {
		size_t length = (*structure->feature_contexts)[i].length;
		(*num_order_contexts)[length-1] += 1;
	}

	// find parent feature for all features of higher order contexts
	for (size_t i = 1; i < structure->feature_contexts->size(); i++) {
		slice s((*structure->feature_contexts)[i].start_index,
				(*structure->feature_contexts)[i].num_features, 1);

		size_t parent_id = (*structure->feature_contexts)[i].parent_id;

		slice parent_s((*structure->feature_contexts)[parent_id].start_index,
				(*structure->feature_contexts)[parent_id].num_features, 1);

		valarray<size_t> match_indices = search_sorted(*structure->feature_outcome_ids, *structure->feature_outcome_ids, parent_s, s) + parent_s.start();
		feature_parent_ids[s] = match_indices;
	}
}

model::~model() {
	delete(num_order_contexts);
	delete(params);
	if (prior_params != 0) {
		delete(prior_params);
	}
}

valarray<double> *model::lognormconst() {

	if (logZ_computed) {
		return &logZ;
	}
	ensure_exp_params();

	logZ[0] = logsumexp(
			static_cast<valarray<double> >((*params)[slice(0, (*structure->feature_contexts)[0].num_features, 1)]));

	size_t order_start_index = 1;
	for (size_t o = 1; o < structure->order; o++) {
#ifdef _OPENMP
		#pragma omp parallel for
#endif
		// MSVC openmp requires iteration variables to be signed
		for (long j = 0; j < (long)(*num_order_contexts)[o-1]; j++) {
			feature_context_t *fc = &(*structure->feature_contexts)[order_start_index + j];
			double parent_sum = logZ[fc->parent_id];

			slice s(fc->start_index, fc->num_features, 1);

			const valarray<double> & exp_level_params = (exp_params)[s];
			valarray<double> parent_level_sums(0.0, exp_level_params.size());

			size_t parent_id = (*structure->feature_contexts)[order_start_index + j].parent_id;

			valarray<size_t> current_indices(s.size());

			for (size_t k = 1; k < o + 1; k++) {

				if (k == 1) {
					current_indices = feature_parent_ids[s];
				} else {
					current_indices = feature_parent_ids[current_indices];
				}

				parent_level_sums += valarray<double>((*params)[current_indices]);


				parent_id = (*structure->feature_contexts)[parent_id].parent_id;
			}
			double current_sum = (((exp_level_params) - 1.0) * exp(parent_level_sums)).sum();

			if (current_sum != 0.0) {
				double sum = exp(parent_sum) + current_sum;
				logZ[order_start_index + j] = log(sum);

			} else {
				logZ[order_start_index + j] = parent_sum;
			}

		}
		order_start_index += (*num_order_contexts)[o-1];
	}



	logZ_computed = true;
	/*
	cerr << "Z=[";
	for (size_t context_id = 0; context_id < data->count_contexts->size(); context_id++) {
			size_t feature_context_id = (*data->count_contexts)[context_id].feature_context_id;
			cerr << exp(logZ[feature_context_id]) << ",";
	}
	cerr << "]" << endl;
	*/

	return &logZ;
}

/**
 * Calculates sums of parameters, where each sum is a sum of
 * a parameter and all its lower order corresponding parameters
 */
std::valarray<double> model::param_sums() {
	valarray<double> result(*params);
	size_t order_start_index = 1;
	for (size_t o = 1; o < structure->order; o++) {
#ifdef _OPENMP
		#pragma omp parallel for
#endif
		// MSVC openmp requires iteration variables to be signed
		for (long j = 0; j < (long)(*num_order_contexts)[o-1]; j++) {
			size_t feature_context_id = order_start_index + j;
			slice s((*structure->feature_contexts)[feature_context_id].start_index,
					(*structure->feature_contexts)[feature_context_id].num_features, 1);
			valarray<size_t> match_indices = feature_parent_ids[s];
			result[s] += valarray<double>(result[match_indices]);
		}
		order_start_index += (*num_order_contexts)[o-1];
	}
	return result;
}

std::valarray<double> model::expectations() {
	ensure_exp_params();
	valarray<double> result(0.0, k.size());

	const valarray<double> zz = exp(*(lognormconst()));

	valarray<double> sum_p_div_z_context(0.0, structure->feature_contexts->size());

	// accumulate sum(p_tilde/Z) for all contexts
	for (size_t context_id = 0; context_id < data->count_contexts->size(); context_id++) {
		size_t feature_context_id = (*data->count_contexts)[context_id].feature_context_id;
		double p_div_z = (*p_tilde_context)[context_id]/zz[feature_context_id];
		feature_context_t *feature_context = &(*structure->feature_contexts)[feature_context_id];
		while (feature_context != NULL) {
			sum_p_div_z_context[feature_context_id] += p_div_z;

			if (feature_context->length > 0) {
				feature_context_id = feature_context->parent_id;
				feature_context = &(*structure->feature_contexts)[feature_context_id];
			} else {
				feature_context = NULL;
			}
		}
	}

	for (size_t i = 0; i < structure->feature_contexts->size(); i++) {
		feature_context_t *feature_context = &(*structure->feature_contexts)[i];
		slice s = slice(feature_context->start_index, feature_context->num_features, 1);
		result[s] = sum_p_div_z_context[i];
	}


	valarray<double> product_exp_params(exp_params);

	size_t order_start_index = 1;

	valarray<double> lower_order_sum(0.0, k.size());

	for (size_t o = 1; o < structure->order; o++) {
#ifdef _OPENMP
		#pragma omp parallel for
#endif
		// MSVC openmp requires iteration variables to be signed
		for (long j = 0; j < (long)(*num_order_contexts)[o-1]; j++) {

			size_t feature_context_id = order_start_index + j;
			slice s((*structure->feature_contexts)[feature_context_id].start_index,
					(*structure->feature_contexts)[feature_context_id].num_features, 1);



			valarray<size_t> match_indices = feature_parent_ids[s];
			product_exp_params[s] *= valarray<double>(product_exp_params[match_indices]);
			double p_div_z = sum_p_div_z_context[feature_context_id];

			valarray<size_t> mmm(match_indices);

			for (size_t k = 1; k < o + 1; k++) {
				for (size_t l = 0; l < match_indices.size(); l++) {
					double add = p_div_z * (exp_params[s.start() + l] - 1) * product_exp_params[mmm[l]];
#ifdef _OPENMP
					#pragma omp atomic
#endif
					lower_order_sum[match_indices[l]] += add;
				}

				if (k < o) {
					match_indices = feature_parent_ids[match_indices];
				}
			}
		}
		order_start_index += (*num_order_contexts)[o-1];
	}
	result *= product_exp_params;
	result += lower_order_sum;

	return result;
}



double model::dual() {
	valarray<double> logZs((data->count_contexts->size()));
	for (size_t i = 0; i < data->count_contexts->size(); i++) {
		logZs[i] = (*(lognormconst()))[(*data->count_contexts)[i].feature_context_id];
	}

	check_nans(logZs, "logZs");
	double dot1 = dot(*p_tilde_context, logZs);
	double dot2 = dot(*params, k);
	double L  =  dot1 - dot2;
	cerr << "  dual is " << L << endl;

	// Use a Gaussian prior for smoothing if requested.
	// This adds the penalty term \sum_{i=1}^m \theta_i^2 / {2 \sigma_i^2}
	if (sigma2 > 0) {
		double penalty;
		if (prior_params == NULL) {
			penalty = 0.5 * ((*params) * (*params) ).sum() / sigma2;
		} else {
			std::valarray<double> params_with_priors(*params - *prior_params);
			penalty = 0.5 * (params_with_priors * params_with_priors).sum() / sigma2;
		}
		L += penalty;
		cerr << "  regularized dual is " <<  L << endl;
	}
	if (isnan(L)) {
		L = std::numeric_limits<double>::max();
	}
	return L;
}

/**
 * Computes the gradient of the entropy dual.
 */
valarray<double> model::grad() {
	valarray<double> g = expectations() - k;

	for (size_t i = 0; i < g.size(); i++) {
		if (isnan(g[i])) {
			g[i] = 0.0;
		}
	}

	cerr << "  norm of gradient =" <<  norm(g) << endl;

	// Use a Gaussian prior for smoothing if requested.  The ith
	// partial derivative of the penalty term is \params_i /
	// \sigma_i^2.  Define 0 / 0 = 0 here; this allows a variance term
	// of sigma_i^2==0 to indicate that feature i should be ignored.
	if (sigma2 > 0.0) {
		valarray<double> penalty(g.size());
		if (prior_params == NULL) {
			penalty = *params / sigma2;
		} else {
			penalty = (*params - *prior_params) / sigma2;
		}
		g += penalty;
		for (size_t i = 0; i < penalty.size(); i++) {
			if (isnan(penalty[i])) {
				g[i] = 0.0;
			}
		}

        double norm_g = norm(g);
        cerr << "  norm of regularized gradient =" << norm_g << endl;
	}
	return g;
}

#ifdef HAVE_LIBLBFGS
static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step) {

	model *m = reinterpret_cast<model*> (instance);
	m->clear_cache();
	valarray<double> *params = m->get_params();
	for (size_t i = 0; i < params->size(); i++) {
		(*params)[i] = x[i];
	}
	double dual = m->dual();
	valarray<double> grad = m->grad();
	for (size_t i = 0; i < grad.size(); i++) {
		g[i] = grad[i];
	}

    return dual;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    ) {
    cerr << "Iteration "<< k  << endl;
    return 0;
}


double model::fit(data_t *data) {
	cerr << "Starting fitting..." << endl;
	this->data = data;
	this->clear_cache();
	p_tilde_context = new valarray<double>(data->count_contexts->size());

	{
		double sum_counts = data->counts->sum();
		valarray<double> p_tilde(data->counts->size());
		for (size_t i = 0; i < data->counts->size(); i++) {
			p_tilde[i] = (double)(*data->counts)[i] / sum_counts;
		}


		for (size_t i = 0; i < data->count_contexts->size(); i++) {
			count_context_t & count_context = (*data->count_contexts)[i];
			slice sa(count_context.start_index, count_context.num_counts, 1);
			(*p_tilde_context)[i] = (static_cast<valarray<double> >(p_tilde[sa])).sum();
		}


		for (size_t i = 0; i < data->count_contexts->size(); i++) {
			count_context_t & count_context = (*data->count_contexts)[i];
			slice s(count_context.start_index, count_context.num_counts, 1);
			size_t feature_context_id = count_context.feature_context_id;
			feature_context_t *feature_context = &(*structure->feature_contexts)[feature_context_id];
			while (feature_context != NULL) {

				// TODO: optimize?
				for (size_t j = 0; j < count_context.num_counts; j++) {
					size_t lower = lower_bound(*structure->feature_outcome_ids,
							feature_context->start_index,
							feature_context->start_index + feature_context->num_features, (*data->count_outcome_ids)[count_context.start_index + j]);
					if ((lower < feature_context->start_index +feature_context->num_features) &&
							((*structure->feature_outcome_ids)[lower] == (*data->count_outcome_ids)[count_context.start_index + j])) {
						k[lower] += p_tilde[count_context.start_index + j];
					}
				}

				if (feature_context->length > 0) {
					feature_context = &(*structure->feature_contexts)[feature_context->parent_id];
				} else {
					feature_context = NULL;
				}
			}
		}
		//cerr << endl;
	}
	//cerr << *p_tilde_context << endl;
	//cerr << k << endl;


	int N = params->size();
	lbfgsfloatval_t *x = lbfgs_malloc(N);

	for (size_t i = 0; i < params->size(); i++) {
		x[i] = (*params)[i];
	}
	lbfgs_parameter_t param;
    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);

    param.epsilon = 1e-7;
    param.past = 5;
    param.delta = 1e-4;
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.orthantwise_c = c1;
    param.orthantwise_end = N;
    param.max_iterations = max_iters;

    cerr << "Starting OWL-BFGS with c1=" << c1 << ", sigma2=" << sigma2 << ", max_iters=" << max_iters << endl;
    time_t start_time = std::time(NULL);
    lbfgsfloatval_t result;

	int status = lbfgs(N, x, &result, evaluate, progress, this, &param);

	double duration = std::difftime(std::time(NULL), start_time);

	if (status == LBFGS_CONVERGENCE) {
		cerr << "OWL-BFGS resulted in convergence" << std::endl;
	} else if (status == LBFGS_STOP) {
		cerr << "OWL-BFGS terminated with the stopping criterion" << std::endl;
	} else if (status == LBFGSERR_MAXIMUMITERATION) {
		cerr << "OWL-BFGS terminated: maximum number of iterations reached" << std::endl;
	} else {
		cerr << "OWL-BFGS terminated with error code (" << status << ")" << std::endl;
	}

	cerr << "Duration: " << duration << " seconds" << endl;
	for (size_t i = 0; i < params->size(); i++) {
		(*params)[i] = x[i];
	}
	lbfgs_free(x);

	delete(p_tilde_context);

	return result;
}
#else /* !HAVE_LIBLBFGS */

double model::fit(data_t *data) {
	cerr << "maxent model estimation not supported (requires liblbfgs)\n";
	exit(2);
	return 0.0;
}
#endif /* HAVE_LIBLBFGS */

void model::clear_cache() {
	logZ_computed = false;
	exp_params_computed = false;
}


void model::log_pmf_context(size_t feature_context_id, valarray<double> &result) {
	double logZ = (*(lognormconst()))[feature_context_id];
	result = -logZ;

	feature_context_t *feature_context = &(*structure->feature_contexts)[feature_context_id];
	while (feature_context != NULL) {
		slice s(feature_context->start_index, feature_context->num_features, 1);
		result[(*structure->feature_outcome_ids)[s]] += valarray<double>((*params)[s]);

		if (feature_context->length > 0) {
			feature_context = &(*structure->feature_contexts)[feature_context->parent_id];
		} else {
			feature_context = NULL;
		}
	}
}

double model::log_prob_context(size_t feature_context_id, size_t outcome_id) {
	double logZ = (*(lognormconst()))[feature_context_id];
	double result = -logZ;

	feature_context_t *feature_context = &(*structure->feature_contexts)[feature_context_id];
	while (feature_context != NULL) {


		size_t pos = lower_bound(*structure->feature_outcome_ids, feature_context->start_index, feature_context->start_index + feature_context->num_features - 1, outcome_id);
		if ((*structure->feature_outcome_ids)[pos] == outcome_id) {
			result  += (*params)[pos];
		}
		if (feature_context->length > 0) {
			feature_context = &(*structure->feature_contexts)[feature_context->parent_id];
		} else {
			feature_context = NULL;
		}
	}
	return result;
}



double model::sum_logpmf() {
	valarray<double> &logZs = *(lognormconst());

	valarray<double> result(1.0, data->count_outcome_ids->size());

	for (size_t i = 0; i < data->count_contexts->size(); i++) {
		slice s((*data->count_contexts)[i].start_index, (*data->count_contexts)[i].num_counts, 1);
		result[s] = -logZs[(*data->count_contexts)[i].feature_context_id];
	}
	for (size_t i = 0; i < data->count_contexts->size(); i++) {
		count_context_t & count_context = (*data->count_contexts)[i];
		slice s(count_context.start_index, count_context.num_counts, 1);
		size_t feature_context_id = count_context.feature_context_id;
		feature_context_t *feature_context = &(*structure->feature_contexts)[feature_context_id];
		while (feature_context != NULL) {
			slice s2(feature_context->start_index, feature_context->num_features, 1);
			valarray<size_t> matched_indices = search_sorted(*structure->feature_outcome_ids, *data->count_outcome_ids, s2, s);

			result[s] += valarray<double>((*params)[matched_indices + (size_t)feature_context->start_index]);

			if (feature_context->length > 0) {
				feature_context = &(*structure->feature_contexts)[feature_context->parent_id];
			} else {
				feature_context = NULL;
			}
		}
	}

	double sum = 0;
	for (size_t i = 0; i < data->counts->size(); i++) {
		sum += result[i] * (double)((*data->counts)[i]);
	}
	return sum;
}



void model::ensure_exp_params() {

	if (!exp_params_computed) {
#ifdef _OPENMP
		#pragma omp parallel for
		// MSVC openmp requires iteration variables to be signed
		for (long i=0; i < (long)params->size(); i++) {
			exp_params[i] = (*params)[i] == 0.0 ? 1.0 : exp((*params)[i]);
		}
#else
		exp_params = exp(*params);
#endif
	}
	exp_params_computed = true;
}
