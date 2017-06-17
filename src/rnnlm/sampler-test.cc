// rnnlm/sampler-test.cc
//
// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "base/kaldi-math.h"
#include <limits>
#include <numeric>
#include "rnnlm/sampler.h"
#include "util/stl-utils.h"

namespace kaldi {
namespace rnnlm {


// returns true if |(a/|a| - b/|b|)| < threshold,
// where |x| is 1-norm.
bool NormalizedSquaredDiffLessThanThreshold(
    const std::vector<double> &a,
    const std::vector<double> &b,
    double threshold) {
  KALDI_ASSERT(a.size() == b.size());
  double a_sum = 0.0, b_sum = 0.0;
  size_t s = a.size();
  for (size_t i = 0; i < s; i++) {
    a_sum += a[i];
    b_sum += b[i];
  }
  if (a_sum == 0.0 || b_sum == 0.0) {
    return (a_sum == 0.0 && b_sum == 0.0);
  }
  double a_scale = 1.0 / a_sum,
      b_scale = 1.0 / b_sum;
  double diff_sum = 0.0;
  for (size_t i = 0; i < s; i++) {
    double a_norm = a[i] * a_scale,
        b_norm = b[i] * b_scale,
        diff = std::abs(a_norm - b_norm);
    diff_sum += diff;
  }
  return (diff_sum < threshold);
}

void UnitTestSampleWithoutReplacement() {
  int32 num_tries = 50;
  for (int32 t = 0; t < num_tries; t++) {
    std::vector<double> prob;
    int32 num_elements = RandInt(1, 100);
    prob.resize(num_elements);
    double total = 0.0;
    for (int32 i = 0; i + 1 < num_elements; i++) {
      if (WithProb(0.2)) {
        prob[i] = RandInt(0, 1);  // 0 or 1.
      } else {
        prob[i] = RandUniform(); // Uniform between 0 and 1.
      }
      total += prob[i];
    }
    int32 total_ceil = std::ceil(total);
    prob[num_elements - 1] =  total_ceil - total;
    std::random_shuffle(prob.begin(), prob.end());

    std::vector<double> sample_total(prob.size());
    size_t l = 0;
    while (true) {
      // this will loop forever if the normalized samples don't approach 'prob'
      // closely enough.
      std::vector<int32> samples;
      SampleWithoutReplacement(prob, &samples);
      KALDI_ASSERT(samples.size() == size_t(total_ceil));
      std::sort(samples.begin(), samples.end());
      KALDI_ASSERT(IsSortedAndUniq(samples));
      for (size_t i = 0; i < samples.size(); i++) {
        sample_total[samples[i]] += 1.0;
      }
      if (NormalizedSquaredDiffLessThanThreshold(prob, sample_total,
                                                 0.1)) {
        KALDI_LOG << "Converged after " << l << " iterations.";
        break;
      }
      l++;  // for debugging purposes, in case it fails.
    }
  }
}


void UnitTestSampleFromCdf() {
  int32 num_tries = 50;
  for (int32 t = 0; t < num_tries; t++) {
    std::vector<double> prob;
    int32 num_elements = RandInt(1, 100);


    prob.resize(num_elements);
    double total = 0.0;
    for (int32 i = 0; i < num_elements; i++) {
      if (WithProb(0.2)) {
        prob[i] = RandInt(0, 1);  // 0 or 1.
      } else {
        prob[i] = RandUniform(); // Uniform between 0 and 1.
      }
      total += prob[i];
    }
    if (total == 0.0)
      continue;  // if all the probs are zero, we can't do the test; try again.

    std::vector<double> cdf(num_elements + 1);
    cdf[0] = RandUniform();
    for (int32 i = 0; i < num_elements; i++) {
      cdf[i+1] = cdf[i] + prob[i];
    }


    std::vector<double> sample_total(prob.size());
    size_t l = 0;
    while (true) {
      // this will loop forever if the samples don't approach 'prob'
      // closely enough.
      const double *sampled_location = SampleFromCdf(&(cdf[0]),
                                                     &(cdf[num_elements]));
      int32 i = sampled_location - &(cdf[0]);
      sample_total[i] += 1.0;

      if (l % 20 == 0) {
        if (NormalizedSquaredDiffLessThanThreshold(prob, sample_total,
                                                   0.1)) {
          KALDI_LOG << "Converged after " << l << " iterations.";
          break;
        }
      }
      l++;
    }
  }
}

// Given a list of unnormalized probabilities p(i), compute
//    q(i) = min(alpha p(i), 1.0),
// where alpha is chosen to ensure that sum_i q(i) equals
// num_words_to_sample.
void NormalizeProbs(int32 num_words_to_sample,
                    std::vector<double> *probs) {
  double sum = std::accumulate(probs->begin(), probs->end(), 0.0);
  for (size_t i = 0; i < probs->size(); i++) {
    // normalize so it sums to num_words_to_sample.
    (*probs)[i] *= num_words_to_sample / sum;
  }
  int32 num_ones = 0;
  for (size_t i = 0; i < probs->size(); i++) {
    if ((*probs)[i] >= 1.0) {
      (*probs)[i] = 1.0;
      num_ones++;
    }
  }
  while (true) {
    double sum = std::accumulate(probs->begin(), probs->end(), 0.0);
    double scale = (num_words_to_sample - num_ones) / (sum - num_ones);
    KALDI_ASSERT(scale > 0.9999);
    if (scale < 1.00001) return;  // we're done.
    // apply the scale.
    for (size_t i = 0; i < probs->size(); i++) {
      if ((*probs)[i] != 1.0) {
        (*probs)[i] *= scale;
        if ((*probs)[i] >= 1.0) {
          (*probs)[i] = 1.0;
          num_ones++;
        }
      }
    }
  }
}

void UnitTestSampleWords() {
  int32 num_tries = 50;
  for (int32 t = 0; t < num_tries; t++) {
    int32 vocab_size = RandInt(200, 300);
    std::vector<BaseFloat> unigram_probs(vocab_size);
    unigram_probs.resize(vocab_size);
    double total = 0.0;
    for (int32 i = 0; i < vocab_size; i++) {
      if (WithProb(0.2)) {
        unigram_probs[i] = RandInt(0, 1);  // 0 or 1.
      } else {
        unigram_probs[i] = RandUniform(); // Uniform between 0 and 1.
      }
      total += unigram_probs[i];
    }
    double inv_total = 1.0 / total;
    for (int32 i = 0; i < vocab_size; i++)
      unigram_probs[i] *= inv_total;

    // add this many extra elements to the unigram distribution.
    int32 num_sparse = RandInt(0, 10);
    std::vector<std::pair<int32, BaseFloat> > higher_order_probs(num_sparse);
    for (int32 i = 0; i < num_sparse; i++) {
      higher_order_probs[i].first = RandInt(0, vocab_size - 1);
      higher_order_probs[i].second = 0.01 + RandUniform();
    }
    std::sort(higher_order_probs.begin(), higher_order_probs.end());
    // remove duplicate words.
    MergePairVectorSumming(&higher_order_probs);
    num_sparse = higher_order_probs.size();

    BaseFloat unigram_weight = RandInt(1, 3);
    int32 num_words_to_sample = RandInt(20, 40);


    // in addition to the unigram and sparse components, the interface
    // allows you to specify words that must be sampled with probability one.
    // this is to test that part of the interface.
    std::vector<int32> words_we_must_sample(RandInt(0, num_words_to_sample / 8));
    for (size_t i = 0; i < words_we_must_sample.size(); i++)
      words_we_must_sample[i] = RandInt(0, vocab_size - 1);
    SortAndUniq(&words_we_must_sample);

    // full_distribution will be an unnormalized distribution proportional to
    // unigram_probs * unigram_weight plus the sparse vector
    // 'higher_order_probs'.
    std::vector<double> full_distribution(vocab_size);
    for (int32 i = 0 ; i < vocab_size; i++) {
      full_distribution[i] = unigram_weight * unigram_probs[i];
    }
    for (int32 i = 0; i < num_sparse; i++) {
      int32 w = higher_order_probs[i].first;
      BaseFloat p = higher_order_probs[i].second;
      KALDI_ASSERT(w >= 0 && w < vocab_size);
      full_distribution[w] += p;
    }
    for (size_t i = 0; i < words_we_must_sample.size(); i++) {
      // here, 100 is just a "large enough number".
      full_distribution[words_we_must_sample[i]] = 100.0;
    }
    NormalizeProbs(num_words_to_sample,
                   &full_distribution);

    Sampler sampler(unigram_probs);
    std::vector<double> sample_total(vocab_size);
    size_t l = 0;
    while (true) {
      // this will loop forever if the normalized samples don't approach
      // 'full_distribution' closely enough.
      std::vector<std::pair<int32, BaseFloat> > sample;
      sampler.SampleWords(num_words_to_sample, unigram_weight,
                          higher_order_probs, words_we_must_sample,
                          &sample);

      KALDI_ASSERT(sample.size() == size_t(num_words_to_sample));
      std::sort(sample.begin(), sample.end());
      KALDI_ASSERT(IsSortedAndUniq(sample));
      for (size_t i = 0; i < sample.size(); i++) {
        sample_total[sample[i].first] += 1.0;
        AssertEqual(sample[i].second, full_distribution[sample[i].first]);
      }
      if (NormalizedSquaredDiffLessThanThreshold(full_distribution,
                                                 sample_total,
                                                 0.1)) {
        KALDI_LOG << "Converged after " << l << " iterations.";
        break;
      }
      l++;  // for debugging purposes, in case it fails.
    }
  }
}


}  // end namespace rnnlm.
}  // end namespace kaldi.

int main() {
  kaldi::SetVerboseLevel(2);  // activates extra testing code.
  using namespace kaldi::rnnlm;
  UnitTestSampleWithoutReplacement();
  UnitTestSampleFromCdf();
  UnitTestSampleWords();
}
