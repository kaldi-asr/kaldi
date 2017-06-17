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
  int32 num_tries = 500;
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


void UnitTestSampleWords() {
  int32 num_tries = 50;
  for (int32 t = 0; t < num_tries; t++) {
    int32 vocab_size = RandInt(200, 300);
    std::vector<BaseFloat> unigram_probs(vocab_size);
    prob.resize(vocab_size);
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

    int32 num_sparse = RandInt(0, 10);
    std::vector<std::pair<int32, BaseFloat> > higher_order_probs(num_sparse);
    for (int32 i = 0; i < num_sparse; i++) {
      higher_order_probs[i].first = RandInt(0, vocab_size - 1);
      higher_order_probs[i].second = 0.01 + RandUniform();
    }
    std::sort(higher_order_probs.begin(), higher_order_probs.end());
    // remove duplicates.
    MergePairVectorSumming(higher_order_probs);

    BaseFloat unigram_weight = RandInt(1, 3);
    int32 num_samples = RandInt(20, 40);

    std::vector<BaseFloat> full_distribution(unigram_probs);
    for (int32 i = 0; i < num_sparse; i++) {
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


}  // end namespace rnnlm.
}  // end namespace kaldi.

int main() {
  using namespace kaldi::rnnlm;
  UnitTestSampleWithoutReplacement();
  UnitTestSampleFromCdf();
  UnitTestSampleWords();
}

