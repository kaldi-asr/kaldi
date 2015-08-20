// nnet2/nnet-stats.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_NNET_STATS_H_
#define KALDI_NNET2_NNET_STATS_H_

#include "nnet2/nnet-nnet.h"

namespace kaldi {
namespace nnet2 {

/* This program computes various statistics from a neural net.  These are
   summaries of certain quantities already present in the network as
   stored on disk, especially regarding certain average values and
   derivatives of the sigmoids.   
*/

struct NnetStatsConfig {  
  BaseFloat bucket_width;
  NnetStatsConfig(): bucket_width(0.025) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("bucket-width", &bucket_width, "Width of bucket in average-derivative "
                   "stats for analysis.");
  }
};

class NnetStats {
 public:
  NnetStats(int32 affine_component_index, BaseFloat bucket_width):
      affine_component_index_(affine_component_index),
      bucket_width_(bucket_width), global_(0, -1) { }
  
  // Use default copy constructor and assignment operator.
  
  void AddStats(BaseFloat avg_deriv, BaseFloat avg_value);

  void AddStatsFromNnet(const Nnet &nnet);
  
  void PrintStats(std::ostream &os);  
 private:

  struct StatsElement {
    BaseFloat deriv_begin; // avg-deriv, beginning of bucket.
    BaseFloat deriv_end;   // avg-deriv, end of bucket.
    BaseFloat deriv_sum;   // sum of avg-deriv within bucket.
    BaseFloat deriv_sumsq;   // Sum-squared of avg-deriv within bucket.
    BaseFloat abs_value_sum; // Sum of abs(avg-value).  Tells us whether it's
    // saturating at one or both ends.
    BaseFloat abs_value_sumsq; // Sum-squared of abs(avg-value).
    int32 count;      // Number of nonlinearities in this bucket.

    StatsElement(BaseFloat deriv_begin,
                 BaseFloat deriv_end):
        deriv_begin(deriv_begin), deriv_end(deriv_end), deriv_sum(0.0),
        deriv_sumsq(0.0), abs_value_sum(0.0), abs_value_sumsq(0.0), count(0) { }
    void AddStats(BaseFloat avg_deriv, BaseFloat avg_value);
    // Outputs stats for this bucket; no newline
    void PrintStats(std::ostream &os); 
  };
  int32 BucketFor(BaseFloat avg_deriv); // returns the bucket
  // for this avg-derivative value, and makes sure it is allocated.
  
  int32 affine_component_index_; // Component index of the affine component
                                // associated with this nonlinearity.
  BaseFloat bucket_width_; // width of buckets of stats we store (in derivative values).
  
  std::vector<StatsElement> buckets_; // Stats divided into buckets by avg_deriv.
  StatsElement global_; // All the stats.
  
};

void GetNnetStats(const NnetStatsConfig &config,
                  const Nnet &nnet,
                  std::vector<NnetStats> *stats);


} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_STATS_H_
