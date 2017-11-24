// rnnlm/sampler.cc

// Copyright 2017  Daniel Povey

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

#include <algorithm>
#include <numeric>
#include <queue>
#include "rnnlm/sampler.h"
#include "base/kaldi-math.h"
#include "util/stl-utils.h"

namespace kaldi {
namespace rnnlm {


void SampleWithoutReplacement(const std::vector<double> &probs,
                              std::vector<int32> *sample) {

  // This outer loop over 't' will *almost always* just run for t == 0.  The
  // loop is necessary only to handle a pathological case.
  for (int32 t = 0; t < 10; t++) {
    sample->clear();
    int32 n = probs.size();

#define DO_SHUFFLE 0

#if DO_SHUFFLE
    // Removing the random shuffling for now because it turns out that
    // with multiple threads, it has to do locking inside the call to rand(),
    // which is quite slow; and using re-entrant random number generators with
    // this is quite complicated.  Anyway this wasn't necessary.
    // Maybe at some later point we can redo the data structures.


    // We randomize the order in which we process the indexes,
    // in order to reduce correlations.. not that this will
    // matter for most applications.
    std::vector<int32> order(n);
    for (int32 i = 0; i < n; i++) order[i] = i;
    std::random_shuffle(order.begin(), order.end());
#endif

    double r = RandUniform();  // r <= 0 <= 1.

    double c = -r;  // c is a kind of counter, to which we add the probabilities
                    // we we process them..  Whenever it becomes >= 0, we add something
                    // to the sample and subtract 1 from c.
    for (int32 i = 0; i < n; i++) {
#if DO_SHUFFLE
      int32 j = order[i];
#else
      int32 j = i;
#endif
      double p = probs[j];
      c += p;
      if (c >= 0) {
        sample->push_back(j);
        c -= 1.0;
      }
    }

    // you can verify by looking at the few lines of code above that
    // 'total_prob' is the total of the 'probs' array.
    double total_prob = c + sample->size() + r;
    int32 k = std::round(total_prob);
    if (std::abs(total_prob - k) > 1.0e-04) {
      // If this happened then the preconditions for this function are
      // violated-- the probs were not well enough normalized.  In double
      // precision the relative rounding error is about 10^-16, so to get an
      // error of 1.0e-04 from rounding we'd have to have 10^12 numbers which in
      // double precision would take 8000 G of memory-- unlikely.
      KALDI_ERR << "The sum of the inputs was " << k << " + "
                << (total_prob - k)
                << " which is too far from a whole number.";
    }
    if (sample->size() == k) {
      return;
    } else {
      // The only possible situations where the sample-size != k and we didn't
      // already crash, are when
      // c + r < 0.9999 or c + r > 0.9999.
      // Since -1 <= c < 0 and 0 <= r <= 1, this is only possible when r < 0.0001
      // and c < -0.9999, or r > 0.9999 and c >= -0.0001.

      // give it a bit of extra space in the assertion.
      KALDI_ASSERT((r < 0.00011 && c < -0.99985) ||
                   (r > 0.99985 && c > -0.00011));

      // .. and continue around the loop.
      // Having 'r' take these values is extremely improbable, so
      // it will be rare to go around the loop more than once.
    }
  }
  KALDI_ERR << "Looped too many times: likely bug.";
}


void CheckDistribution(const Distribution &d) {
  Distribution::const_iterator iter = d.begin(),
      endm1 = d.end() - 1;
  if (d.empty())
    return;
  for (; iter != endm1; ++iter) {
    KALDI_ASSERT(iter->second > 0.0 &&
                 iter->first < (iter+1)->first);
  }
  KALDI_ASSERT(d.back().second > 0.0);
}

void WeightDistribution(BaseFloat weight,
                        Distribution *d) {
  Distribution::iterator iter = d->begin(),
      end = d->end();
  for (; iter != end; ++iter)
    iter->second *= weight;
}


BaseFloat TotalOfDistribution(const Distribution &d) {
  double tot = 0.0;
  Distribution::const_iterator iter = d.begin(),
      end = d.end();
  for (; iter != end; ++iter)
    tot += iter->second;
  return tot;
}


const double* SampleFromCdf(const double *cdf_start,
                            const double *cdf_end) {
  double tot_prob = *cdf_end - *cdf_start;
  KALDI_ASSERT(cdf_end > cdf_start && tot_prob > 0.0);
  double cutoff = *cdf_start + tot_prob * RandUniform();
  if (cutoff >= *cdf_end) {
    // Mathematically speaking this should not happen; if it happens it is due
    // to roundoff.  It should be extremely rare in any case.
    cutoff = *cdf_start;
  }
  // With respect to the sample where [*cdf_start.. *cdf_end]
  // is [ 0.50, 0.55, 0.65, 0.70 ], suppose the randomly
  // sampled 'cutoff' is 0.68.  The std::upper_bound call
  // below finds the first location in the range
  // [ 0.55, 0.65, 0.70 ] that is > 0.68, which
  // in this case is '0.70'.  We then return that
  // pointer minus one, which is the pointer to 0.65.
  const double *ans = std::upper_bound(cdf_start + 1,
                                       cdf_end + 1,
                                       cutoff) - 1;
  // if the following assertion fails, it means that upper_bound returned
  // cdf_end + 1, which means that *cdf_end was not > 'cutoff'.  But we
  // ensured above that *cdf_end was > cutoff, so this should not happen.
  KALDI_ASSERT(ans != cdf_end);
  // If the following fails, it would be an error in our logic or in
  // std::upper_bound.
  KALDI_ASSERT(ans[1] != ans[0]);
  return ans;
}


// Merges two distributions, summing the probabilities of any elements that
// occur in both.
void MergeDistributions(const Distribution &d1,
                        const Distribution &d2,
                        Distribution *d) {
  if (GetVerboseLevel() >= 2) {
    CheckDistribution(d1);
    CheckDistribution(d2);
  }
  d->resize(d1.size() + d2.size());
  // we could write a single function that does the jobs of the
  // two things below, which might improve speed slightly, if
  // this becomes a bottleneck.
  std::merge(d1.begin(), d1.end(), d2.begin(), d2.end(), d->begin());
  MergePairVectorSumming(d);
  if (GetVerboseLevel() >= 2) {
    CheckDistribution(*d);
  }

}


void Sampler::SampleWords(
    int32 num_words_to_sample,
    BaseFloat unigram_weight,
    const std::vector<std::pair<int32, BaseFloat> > &higher_order_probs,
    const std::vector<int32> &words_we_must_sample,
    std::vector<std::pair<int32, BaseFloat> > *sample) const {
  CheckDistribution(higher_order_probs);  // TODO: delete this.
  int32 vocab_size = unigram_cdf_.size();
  KALDI_ASSERT(IsSortedAndUniq(words_we_must_sample) &&
               num_words_to_sample > 0 && num_words_to_sample < vocab_size);

  int32 num_words_we_must_sample = words_we_must_sample.size();
  if (num_words_we_must_sample > 0) {
    KALDI_ASSERT(num_words_we_must_sample < vocab_size &&
                 num_words_we_must_sample < num_words_to_sample);
    KALDI_ASSERT(words_we_must_sample.front() >= 0 &&
                 words_we_must_sample.back() < vocab_size);
  }

  BaseFloat total_existing_weight = unigram_weight +
      TotalOfDistribution(higher_order_probs);
  // To ensure that all the words we must sample actually get sampled,
  // we need to make sure that they are sampled with probability 1.0.
  // i.e. that after computing alpha, alpha p(i) for all of those words
  // is >= 1.0.  See comment for one of the versions of SampleWords()
  // in the header for more explanation.
  //
  // We can ensure this by making sure that after adding these words
  // each with probability p, each of these words has at least
  // weight T / num_words_to_sample, where 'T' is the new total
  // of the distribution.  I.e., we require that
  //   p > T / num_words_to_sample
  // Since T = total_existing_weight + p * num_words_we_must_sample, we have:
  //   p * num_words_to_sample > total_existing_weight + p * num_words_we_must_sample.
  // i.e.
  //  p > total_existing_weight / (num_words_to_sample - num_words_we_must_sample).
  // To minimize roundoff problems we make p just a little bigger than
  // that, bigger by a factor of 0.1.  So we'll set
  // p = 1.1 * total_existing_weight / (num_words_to_sample - num_words_we_must_sample).

  BaseFloat p = 1.1 * total_existing_weight /
      (num_words_to_sample - num_words_we_must_sample);

  std::vector<std::pair<int32, BaseFloat> > words_we_must_sample_distribution(
      num_words_we_must_sample);
  for (int32 i = 0 ; i < num_words_we_must_sample; i++) {
    words_we_must_sample_distribution[i].first = words_we_must_sample[i];
    words_we_must_sample_distribution[i].second = p;
  }

  std::vector<std::pair<int32, BaseFloat> > merged_distribution;
  MergeDistributions(higher_order_probs,
                     words_we_must_sample_distribution,
                     &merged_distribution);

  SampleWords(num_words_to_sample, unigram_weight,
              merged_distribution,
              sample);
  if (GetVerboseLevel() >= 2) {
    std::vector<int32> merged_list(words_we_must_sample);
    for (size_t i = 0; i < sample->size(); i++)
      merged_list.push_back((*sample)[i].first);
    SortAndUniq(&merged_list);
    // if the following assert fails, it means that one of the words
    // that we were required to sample, was not in fact sampled.
    // This implies there was a bug somewhere, or a flaw in
    // our reasoning.
    KALDI_ASSERT(merged_list.size() == sample->size());
  }
}

Sampler::Sampler(const std::vector<BaseFloat> &unigram_probs) {
  KALDI_ASSERT(!unigram_probs.empty());
  double total = std::accumulate(unigram_probs.begin(),
                                 unigram_probs.end(),
                                 0.0);
  KALDI_ASSERT(std::abs(total - 1.0) < 1.0e-02);
  double inv_total = 1.0 / total;

  double sum = 0.0;
  size_t n = unigram_probs.size();
  unigram_cdf_.resize(n + 1);
  unigram_cdf_[0] = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += unigram_probs[i];
    unigram_cdf_[i + 1] = sum * inv_total;
  }
}



void Sampler::SampleWords(
    int32 num_words_to_sample,
    BaseFloat unigram_weight,
    const std::vector<std::pair<int32, BaseFloat> > &higher_order_probs,
    std::vector<std::pair<int32, BaseFloat> > *sample) const {
  int32 vocab_size = unigram_cdf_.size() - 1;
  KALDI_ASSERT(num_words_to_sample > 0 &&
               num_words_to_sample + 1 < unigram_cdf_.size() &&
               unigram_weight > 0.0);
  if (!higher_order_probs.empty()) {
    KALDI_ASSERT(higher_order_probs.front().first >= 0 &&
                 higher_order_probs.back().first < vocab_size);
  }
  if (GetVerboseLevel() >= 2) {
    CheckDistribution(higher_order_probs);
  }

  std::vector<Interval> intervals;
  double total_p = GetInitialIntervals(unigram_weight, higher_order_probs,
                                       &intervals);
  // you can interpret total_p as sum_i p(i), with reference to the
  // math in the header next to the declaration of SampleWords().
  if (GetVerboseLevel() >= 2) {
    AssertEqual(total_p,
                unigram_weight + TotalOfDistribution(higher_order_probs));
  }
  NormalizeIntervals(num_words_to_sample, total_p, &intervals);
  SampleFromIntervals(intervals, sample);
}



// This hacked version of std::priority_queue allows us to extract all elements
// of the priority queue to a supplied vector, in an efficient way.  It relies
// on the fact that std::priority<queue> stores the underlying container as a
// protected member 'c'.  The only way to do this using the supplied interface
// of std::priority_queue is to repeatedly pop() the element from the queue, but
// that is too slow, and it actually had an impact on the speed of the
// application.
template <typename T>
class hacked_priority_queue: public std::priority_queue<T> {
 public:
  void append_all_elements(std::vector<T> *output) const {
    output->insert(output->end(), this->c.begin(), this->c.end());
  }
  // we have to redeclare the constructor.
  template <typename InputIter> hacked_priority_queue(
      InputIter begin, const InputIter end): std::priority_queue<T>(begin, end) { }
};


// static
void Sampler::NormalizeIntervals(int32 num_words_to_sample,
                                 double total_p,
                                 std::vector<Interval> *intervals) {
  // as mentioned in the header, if the input probabilities of the words
  // (here represented as Intervals)/ are p(i), we define
  //  q(i) = min(alpha p(i), 1.0)
  // where alpha is chosen so that sum_i q(i) == num_words_to_sample.
  // 'current_alpha' is initialized to the alpha that we would
  // have if none of the quantities alpha p(i) were greater than 1.
  // This function computes q(i).
  double current_alpha = num_words_to_sample / total_p;

  // 'num_ones' is the number of times the expression min(alpha p(i), 1.0)
  // is >= 1.0.
  int32 num_ones = 0;
  // 'total_remaining_p' is total_p [which equals the sum of p(i)] minus the
  // total of the p(i) for which we already know that alpha p(i) >= 1.
  double total_remaining_p = total_p;

  // In general, we will have:
  //  current_alpha = (num_words_to_sample - num_ones) / total_remaining_p.
  // As we update 'num_ones' and 'total_remaining_p', we will continue
  // to update current_alpha, and it will keep getting larger.
  hacked_priority_queue<Interval> queue(intervals->begin(), intervals->end());

  // clear 'intervals'; we'll use the space to store the intervals that will
  // have a prob of exactly 1.0, and eventually we'll add the rest.
  intervals->clear();
  while (!queue.empty()) {  // note: we normally won't reach the condition where
                          // queue.empty() here; we'll normally break.
    Interval top = queue.top();
    if (current_alpha * top.prob < 1.0) {
      break;  // we're done.
    } else {
      queue.pop();
      size_t interval_size = top.end - top.start;
      if (interval_size > 1) {
        // it's a range containing more than one thing -> we can split the
        // range.
        size_t half_size = interval_size / 2;
        double start_cdf = top.start[0],
            mid_cdf = top.start[half_size],
            end_cdf = top.end[0],
            total_unigram_prob = end_cdf - start_cdf,
            first_half_unigram_prob = mid_cdf - start_cdf,
            second_half_unigram_prob = (total_unigram_prob -
                                        first_half_unigram_prob),
            top_prob = top.prob;
        KALDI_ASSERT(total_unigram_prob > 0.0 && top_prob > 0.0);
        if (first_half_unigram_prob > 0.0) {
          queue.push(Interval(top_prob * first_half_unigram_prob /
                              total_unigram_prob,
                              top.start,
                              top.start + half_size));
        }
        if (second_half_unigram_prob > 0.0) {
          queue.push(Interval(top_prob * second_half_unigram_prob /
                              total_unigram_prob,
                              top.start + half_size, top.end));
        }
      } else {
        // It's an interval of one thing, we can't split it.
        // It's one of those things that takes the "1.0" in the expression
        // min(alpha p(i), 1.0).
        total_remaining_p -= top.prob;
        num_ones++;
        double new_alpha = (num_words_to_sample - num_ones) / total_remaining_p;
        top.prob = 1.0;
        intervals->push_back(top);
        KALDI_ASSERT(queue.empty() ||
                     (total_remaining_p > 0 && new_alpha > current_alpha));
        current_alpha = new_alpha;
      }
    }
  }
#if 0
  // The following code is a bit slow but has the advantage of not assuming
  // anything about the internals of class std::priority_queue.
  while (!queue.empty()) {
    Interval top = queue.top();
    top.prob *= current_alpha;
    queue.pop();
    intervals->push_back(top);
  }
#else
  { // This code is faster but relies on the fact that priority_queue
    // has a protected member 'c' which is the underlying container.
    size_t cur_size = intervals->size();
    queue.append_all_elements(intervals);
    // the next loop scales the 'prob' members of the elements we just
    // added to 'intervals', by current_alpha.
    std::vector<Interval>::iterator iter = intervals->begin() + cur_size,
        end = intervals->end();
    for (; iter != end; ++iter) iter->prob *= current_alpha;
  }
#endif

  if (GetVerboseLevel() >= 2) {
    double tot_prob = 0.0;
    for (size_t i = 0; i < intervals->size(); i++) {
      double p = (*intervals)[i].prob;
      KALDI_ASSERT(p > 0.0 && p <= 1.0);
      tot_prob += p;
    }
    KALDI_ASSERT(tot_prob - num_words_to_sample < 1.0e-04);
  }
}

void Sampler::SampleFromIntervals(const std::vector<Interval> &intervals,
                                  std::vector<std::pair<int32, BaseFloat> > *samples)  const {
  size_t num_intervals = intervals.size();
  std::vector<double> probs(num_intervals);
  for (size_t i = 0; i < num_intervals; i++)
    probs[i] = intervals[i].prob;
  // 'raw_samples' will contain indexes into the 'intervals' vector,
  // which we need to convert into actual words.
  std::vector<int32> raw_samples;
  SampleWithoutReplacement(probs, &raw_samples);
  size_t num_samples = raw_samples.size();
  samples->resize(num_samples);
  const double *cdf_start = &(unigram_cdf_[0]);
  for (size_t i = 0; i < num_samples; i++) {
    int32 j = raw_samples[i];  // j is interval index.
    const Interval &interval = intervals[j];
    if (interval.end == interval.start + 1) {
      // handle this simple case simply, even though
      // the general case on the other side of the if-statement
      // would give the correct expression.
      int32 word = interval.start - cdf_start;
      (*samples)[i].first = word;
      (*samples)[i].second = interval.prob;
    } else {
      const double *word_ptr = SampleFromCdf(interval.start,
                                             interval.end);
      int32 word = word_ptr - cdf_start;
      // the probability with which this word was sampled is: the probability of
      // sampling from this interval of the unigram, times the probability of
      // this word given the unigram distribution (which equals word[1] -
      // word[0]), divided by the probability of this whole unigram interval.
      // Actually we could more simply compute this, as unigram_weight * alpha *
      // (word_ptr[1] - word_ptr[0]), but alpha and unigram_weight not passed
      // into this function and I'd rather not make linkages between different
      // parts of the code.
      BaseFloat prob = interval.prob *
          (word_ptr[1] - word_ptr[0]) / (*interval.end - *interval.start);
      (*samples)[i].first = word;
      (*samples)[i].second = prob;
    }
  }
}


double Sampler::GetInitialIntervals(
    BaseFloat unigram_weight,
    const std::vector<std::pair<int32, BaseFloat> > &higher_order_probs,
    std::vector<Interval> *intervals) const {
  double ans = 0.0;
  intervals->clear();
  intervals->reserve(higher_order_probs.size() * 2 + 1);

  std::vector<std::pair<int32, BaseFloat> >::const_iterator
      h_iter = higher_order_probs.begin(),
      h_end = higher_order_probs.end();

  size_t vocab_size = unigram_cdf_.size() - 1;
  size_t cur_start = 0;
  const double *cdf = &(unigram_cdf_[0]);

  for (; h_iter != h_end; ++h_iter) {
    int32 w = h_iter->first;
    // include the unigram part of the probability in 'p': we add the unigram,
    // we don't back off to it.
    double p = h_iter->second +
        unigram_weight * (cdf[w + 1] - cdf[w]);
    KALDI_ASSERT(p > 0);
    if (w > cur_start && cdf[w] > cdf[cur_start]) {
      // Before we can add an Interval for (w, p), we have some lower-numbered
      // words to deal with.
      double range_p = unigram_weight * (cdf[w] - cdf[cur_start]);
      intervals->push_back(Interval(range_p,
                                    cdf + cur_start,
                                    cdf + w));
      ans += range_p;
    }
    intervals->push_back(Interval(p, cdf + w, cdf + w + 1));
    ans += p;
    cur_start = w + 1;
  }
  KALDI_ASSERT(cur_start <= vocab_size);
  double range_p = unigram_weight * (cdf[vocab_size] - cdf[cur_start]);
  if (range_p > 0) {
    intervals->push_back(Interval(range_p, cdf + cur_start, cdf + vocab_size));
    ans += range_p;
  }
  return ans;
}



}  // namespace rnnlm
}  // namespace kaldi
