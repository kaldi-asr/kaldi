// nnet2/nnet-stats.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-stats.h"

namespace kaldi {
namespace nnet2 {

void NnetStats::StatsElement::PrintStats(std::ostream &os) {
  BaseFloat c = (count == 0 ? 1 : count), // prevent division by zero.
      deriv_mean = deriv_sum/c,
      deriv_stddev = std::sqrt(deriv_sumsq/c - deriv_mean*deriv_mean),
      abs_value_mean = abs_value_sum/c,
      abs_value_stddev = std::sqrt(abs_value_sumsq/c -
                                   abs_value_mean*abs_value_mean);

  os << '[' << deriv_begin << ':' << deriv_end << "] count=" << count
     << ", deriv mean,stddev=" << deriv_mean << ',' << deriv_stddev
     << ", abs-avg-value mean,stddev=" << abs_value_mean << ','
     << abs_value_stddev;
}
  
void NnetStats::StatsElement::AddStats(BaseFloat avg_deriv, BaseFloat avg_value) {
  count++;
  deriv_sum += avg_deriv;
  deriv_sumsq += avg_deriv * avg_deriv;
  abs_value_sum += std::abs(avg_value);
  abs_value_sumsq += avg_value * avg_value;
}

int32 NnetStats::BucketFor(BaseFloat avg_deriv) {
  KALDI_ASSERT(avg_deriv >= 0.0);
  KALDI_ASSERT(bucket_width_ > 0.0);
  // cast ratio to int.  Since we do +0.5, this rounds down.
  int32 index = static_cast<int32>(avg_deriv / bucket_width_ + 0.5);
  while (index >= static_cast<int32>(buckets_.size()))
    buckets_.push_back(StatsElement(buckets_.size() * bucket_width_,
                                    (buckets_.size() + 1) * bucket_width_));
  return index;
}

void NnetStats::AddStats(BaseFloat avg_deriv, BaseFloat avg_value) {
  global_.AddStats(avg_deriv, avg_value);
  buckets_[BucketFor(avg_deriv)].AddStats(avg_deriv, avg_value);
}

void NnetStats::AddStatsFromNnet(const Nnet &nnet) {
  const AffineComponent *ac = dynamic_cast<const AffineComponent*>(
      &(nnet.GetComponent(affine_component_index_)));
  KALDI_ASSERT(ac != NULL); // would be an error in calling code.
  const NonlinearComponent *nc = dynamic_cast<const NonlinearComponent*>(
      &(nnet.GetComponent(affine_component_index_ + 1)));
  KALDI_ASSERT(nc != NULL); // would be an error in calling code.

  double count = nc->Count();
  if (count == 0) {
    KALDI_WARN << "No stats stored with nonlinear component";
    return;
  }
  const CuVector<double> &value_sum = nc->ValueSum();
  const CuVector<double> &deriv_sum = nc->DerivSum();
  if (value_sum.Dim() != deriv_sum.Dim())
    KALDI_ERR << "Error computing nnet stats: probably you are "
              << "trying to compute stats for a sigmoid layer.";
  for (int32 i = 0; i < value_sum.Dim(); i++) {
    BaseFloat avg_value = value_sum(i) / count,
        avg_deriv = deriv_sum(i) / count;
    AddStats(avg_deriv, avg_value);
  }
}

void NnetStats::PrintStats(std::ostream &os) {
  os << "Stats for buckets:" << std::endl;
  for (size_t i = 0; i < buckets_.size(); i++) {
    buckets_[i].PrintStats(os);
    os << std::endl;
  }
  os << "Global stats: ";
  global_.PrintStats(os);
  os << std::endl;
}

void GetNnetStats(const NnetStatsConfig &config,
                  const Nnet &nnet,
                  std::vector<NnetStats> *stats) {
  KALDI_ASSERT(stats->size() == 0);
  for (int32 c = 0; c + 1 < nnet.NumComponents(); c++) {
    const AffineComponent *ac = dynamic_cast<const AffineComponent*>(
        &(nnet.GetComponent(c)));
    if (ac == NULL) continue;
    const NonlinearComponent *nc = dynamic_cast<const NonlinearComponent*>(
        &(nnet.GetComponent(c + 1)));
    if (nc == NULL) continue;
    // exclude softmax.
    const SoftmaxComponent *sc = dynamic_cast<const SoftmaxComponent*>(
        &(nnet.GetComponent(c + 1)));
    if (sc != NULL) continue;
    stats->push_back(NnetStats(c, config.bucket_width));
    stats->back().AddStatsFromNnet(nnet);
  }
}



} // namespace nnet2
} // namespace kaldi
