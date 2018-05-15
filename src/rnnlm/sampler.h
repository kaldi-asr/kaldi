// rnnlm/sampler.h

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_RNNLM_SAMPLER_H_
#define KALDI_RNNLM_SAMPLER_H_

#include <vector>
#include "base/kaldi-common.h"

namespace kaldi {
namespace rnnlm {


/**
   Sample without replacement from a distribution, with provided 1st order inclusion
   probabilities.  For example, if probs[i] is 1.0, i will definitely be included
   in the list 'sample', and probs[i] is 0.0, i will definitely not be included.

   @params [in] probs   The input vector of inclusion probabilities, with
                        0.0 <= probs[i] <= 1.0, and the sum of 'probs' should
                        be close to an integer.  (specifically: within 1.0e-03 of
                        a whole number; this should be easy to ensure in double
                        precision).  Let 'k' be this sum, rounded to
                        the nearest integer.
   @params [out] sample  The vector 'sample' will be set to an unsorted list
                        of 'k' distinct samples with first order inclusion
                        probabilities given by 'probs'.
 */
void SampleWithoutReplacement(const std::vector<double> &probs,
                              std::vector<int32> *sample);



/**
   This function samples from part of a distribution expressed as a cdf
   (cumulative density function).  It is a utility function used in class
   Sampler, but we make it a namespace function so that we can test it.
   It does the sampling in O(log(cdf_end - cdf_start)) time, using
   binary search.

    @param [in] cdf_start   The start of a range of a cdf.
    @param [in] cdf_end     The end of a range of a cdf.  We require that
                            cdf_end > cdf_start and *cdf_end > *cdf_start,
                            and there must be nondecreasing doubles in between.
                            As an example, suppose cdf_end - cdf_start == 3,
                            and the full range including the ends
                            is [ 0.50, 0.55, 0.65, 0.70 ].
                            This is interpreted as representing the probabilities
                            of 3 elements, which are difference between the
                            elements, meaning probabilities [ 0.05, 0.1, 0.05 ].
                            What this function does is to renormalize those values
                            to sum to 1.0 and then saple from the resulting distribution,
                            i.e. the distribution [ 0.25, 0.5, 0.25 ] in this
                            example, so we'd return 'cdf_start' with proability 0.25,
                            'cdf_start + 1' with probability 0.5, and
                            'cdf_start + 2' with probability 0.25.
     @return                Returns a pointer cdf_start <= p < cdf_end, with probability
                            proportional to p[1] - p[0].
*/
const double* SampleFromCdf(const double *cdf_start,
                            const double *cdf_end);


/**
   This class allows us to sample a set of words from a distribution over
   words, where the distribution (which ultimately comes from an ARPA-style
   language model) is given as a combination of a unigram distriubution
   with a sparse component represented as a list of (word-index, probability)
   pairs.
 */
class Sampler {
 public:
  // Initialize the class.  The unigram probabilities (which you can think
  // of as the probability for each word if we don't know the history) are given.
  // each element of unigram_probs should be >= 0, and they should sum to
  // a value close to 1.
  // This class does not retain a reference to 'unigram_probs' after
  // the constructor exits.
  explicit Sampler(const std::vector<BaseFloat> &unigram_probs);


  /// Sample words from the supplied distribution, appropriately scaled.
  /// Let the unnormalized distribution be as follows:
  ///    p(i)  = unigram_weight * u(i) + h(i)
  /// where u(i) is the 'unigram_probs' vector this class was constructed
  /// with, and h(i) is the probability that word i is given (if any) in
  /// the sparse vector that 'higher_order_probs' represents.
  /// Notice that we are adding to the unigram distribution, we are not
  /// backing off to it.  Doing it this way makes a lot of things simpler.
  ///
  /// We define the first-order inclusion probabilities:
  ///   q(i) = min(alpha p(i), 1.0)
  /// where alpha is chosen so that the sum of q(i) equals 'num_words_to_sample'.
  /// Then we generate a sample whose first-order inclusion probabilities
  /// are q(i).  We do all this without explicitly iterating over the unigram
  /// distribution, so this is fairly fast.
  ///
  ///   @param [in] num_words_to_sample    The number of words that we are
  ///                            directed sample; must be > 0 and less than
  ///                            the number of nonzero elements of the 'unigram_probs'
  ///                            that this class was constructed with.
  ///   @param [in] unigram_weight     Must be > 0.0.  Search above for p(i) to
  ///                             see what effect it has.
  ///   @param [in] higher_order_probs   A vector of pairs (i, p)  where
  ///                            0 <= i < unigram_probs.size() (referring to the
  ///                            unigram_probs vector used in the constructor),
  ///                            and p > 0.0.  This vector must be sorted and
  ///                            unique w.r.t. i.  Note: the probabilities
  ///                            here will be added to the unigram probabilities
  ///                            of the words concerned.
  ///   @param [out] sample      The sampled list of words, represented as pairs
  ///                            (i, p), where 0 <= i < unigram_probs.size() is
  ///                            the word index and 0 < p <= 1 is the probabilitity
  ///                            with which that word was included in the set.
  ///                            The list will not be sorted, but it will be unique
  ///                            on the int.  Its size will equal num_words_to_sample.
  void SampleWords(int32 num_words_to_sample,
                   BaseFloat unigram_weight,
                   const std::vector<std::pair<int32, BaseFloat> > &higher_order_probs,
                   std::vector<std::pair<int32, BaseFloat> > *sample) const;

  /// This is an alternative version of SampleWords() which allows you to
  /// specify a list of words that must be sampled (i.e. after scaling, they
  /// must have probability 1.0.).  It does this by adding them to the
  /// distribution with sufficiently large probability and then calling the
  /// other version of SampleWords().
  ///
  /// The vector 'words_we_must_sample' must be sorted and unique, and all
  /// elements i must satisfy 0 <= i < unigram_probs.size(), where unigram_probs
  /// was the vector supplied to the constructor.
  /// See the comments for the other version of SampleWords() to understand the
  /// interface, which is otherwise the same.
  void SampleWords(int32 num_words_to_sample,
                   BaseFloat unigram_weight,
                   const std::vector<std::pair<int32, BaseFloat> > &higher_order_probs,
                   const std::vector<int32> &words_we_must_sample,
                   std::vector<std::pair<int32, BaseFloat> > *sample) const;


 private:

  // This structure represents a contiguous range of symbols; 'start' and 'end'
  // are pointers into the contents of 'unigram_cdf_'.  Let
  //  start_i = start - &(unigram_cdf_[0])
  //  end_i = end - &(unigram_cdf_[0])
  // and we require that end_i > start_i.
  // Then this struct Interval represents the set of words from
  // start_i to end_i - 1, with a total probability mass given
  // by 'prob'.  If end > start + 1, the probability mass is
  // apportioned among the words proportional to the 'unigram_probs'
  // vector passed to the constructor, with cdf given by unigram_cdf_.
  //
  // Note: although we call the total mass 'prob', and we require that
  // prob > 0.0, there is not necessarily a constraint that prob < 1.0
  // in early stages of processing.  Ultimately these will be processed
  // into inclusion probabilities, which will sum to num_words_to_sample.
  // Search for q(i) above for more explanation.
  struct Interval {
    double prob;
    const double *start;
    const double *end;
    // this operator < allows us to include Interval in a priority_queue
    // from which we can select the largest one (used when computing alpha).
    bool operator < (const Interval &other) const {
      return prob < other.prob;
    }
    Interval(double p, const double *s, const double *e):
        prob(p), start(s), end(e) { }
  };

  /// Given a distribution over words which we will write as p(i)
  /// (this is the initial state of 'intervals'), produces
  ///   q(i) = min(alpha p(i), 1.0)
  /// where alpha is chosen so that sum_i q(i) == num_words_to_sample.
  /// This involves choosing alpha in an algorithm involving a queue
  /// of Intervals.  We may have to split some Intervals.
  /// After this function is called, 'intervals' will contain q(i),
  /// so each Interval will have prob <= 1.0 and the sum of the
  /// probs will equal num_words_to_sample.
  /// 'total_p' is the sum of (*intervals)[i].prob; it is provided
  /// to this function so that it doesn't have to compute it itself.
  static void NormalizeIntervals(int32 num_words_to_sample,
                                 double total_p,
                                 std::vector<Interval> *intervals);

  /// Sample from the distribution q(i) represented by 'intervals'.
  ///  @param [in] intervals  The vector of Intervals to sample from.
  ///                    The number of things to sample are given by the sum of
  ///                    intervals[i].prob, and will equal the original
  ///                    'num_words_to_sample' that was passed into the function
  ///                    SampleWords().  Below, we refer to this number as
  ///                    'num_words_to_sample', although it is not passed in
  ///                    explicitly.
  ///  @param [out] sample   The sampled words and the probabilities 0 < p <= 1
  ///                    with which they were included in the sample, are written
  ///                    to here.  The size of this vector will equal
  ///                    'num_words_to_sample' at exit.  This vector will not
  ///                    be sorted.
  void SampleFromIntervals(const std::vector<Interval> &intervals,
                           std::vector<std::pair<int32, BaseFloat> > *sample) const;

  // This helper function, used inside SampleWords(), combines the unigram and
  // higher-order portions of the distribution into a single unified format
  // based on Intervals.  The intervals that it outputs represent the p(i) in
  // the comment above, i.e. this is before we compute alpha and normalize it so
  // that it sums to num_words_to_sample.
  // Returns the sum of the probabilities of the intervals.
  double GetInitialIntervals(BaseFloat unigram_weight,
                             const std::vector<std::pair<int32, BaseFloat> > &higher_order_probs,
                             std::vector<Interval> *intervals) const;



  // the cdf (cumulative density function) of the unigram distribution, indexed
  // from 0 to N where N is unigram_probs.size() given in the constructor (so
  // the dimension of unigram_cdf_ is N+1).  We make sure the unigram
  // distribution is normalized, so unigram_cdf_[0] == 0.0 and
  // unigram_cdf_.back() == 1.0
  std::vector<double> unigram_cdf_;
};


// A 'Distribution' represents an, unnormalized distribution
// over a discrete space.  E.g., [ (5, 0.5), (10, 0.4) ]
// represents: choose 5 with probability 0.5, choose 10 with probability
// 0.4.  (we said that it's unnormalized, so these things won't
// necessarily sum to one).
// A valid Distribution must be sorted and unique on the .first
// element, and all the .second elements must be > 0.
typedef std::vector<std::pair<int32, BaseFloat> > Distribution;


// Check that a Distribution is sorted and unique on its
// .first values, and that all of its .second values are > 0.
void CheckDistribution(const Distribution &d);


// Weights a Distribution by multiplying all the .second elements by
// 'weight'.  'weight' must be > 0.0.
void WeightDistribution(BaseFloat weight,
                        Distribution *d);

// Returns the sum of the .second elements of a Distribution.
BaseFloat TotalOfDistribution(const Distribution &d);


// Merges two distributions, summing the probabilities of any elements that
// occur in both.
void MergeDistributions(const Distribution &d1,
                        const Distribution &d2,
                        Distribution *d);






}  // namespace rnnlm
}  // namespace kaldi
#endif  // KALDI_RNNLM_SAMPLER_H_
