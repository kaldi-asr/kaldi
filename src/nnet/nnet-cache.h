// nnet/nnet-cache.h

// Copyright 2012  Karel Vesely

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


#ifndef KALDI_NNET_NNET_CACHE_H_
#define KALDI_NNET_NNET_CACHE_H_

#include "base/kaldi-math.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

/**
 * The feature-target pair cache
 */
class Cache {
  typedef enum { EMPTY, FILLING, FULL, EMPTYING } State;

 public:
  Cache() : state_(EMPTY), filling_pos_(0), emptying_pos_(0), 
            cachesize_(0), bunchsize_(0), randomized_(false) 
  { }
  ~Cache() { }
 
  /// Initialize the cache
  void Init(int32 cachesize, int32 bunchsize);

  /// Add data to cache
  void AddData(const CuMatrix<BaseFloat> &features, const std::vector<int32> &targets);
  /// Randomizes the cache
  void Randomize();
  /// Get the bunch of training data from cache
  void GetBunch(CuMatrix<BaseFloat> *features, std::vector<int32> *targets);


  /// Returns true if the cache was completely filled
  bool Full() { 
    return (state_ == FULL); 
  }
  
  /// Returns true if the cache is empty
  bool Empty() { 
    return (state_ == EMPTY || filling_pos_ < bunchsize_); 
  }

  /// Returns true if the cache is empty
  bool Randomized() { 
    return randomized_;
  }


 private:
  struct GenerateRandom { 
    int32 operator()(int32 max) const {
      // return lrand48() % max; 
      return RandInt(0, max-1); 
    }
  };
  
  State state_; ///< Current state of the cache

  int32 filling_pos_;  ///< Number of frames filled to cache by AddData
  int32 emptying_pos_; ///< Number of frames given by cache by GetBunch
  
  int32 cachesize_; ///< Size of cache
  int32 bunchsize_; ///< Size of bunch

  bool randomized_;

  CuMatrix<BaseFloat> features_; ///< Feature cache
  CuMatrix<BaseFloat> features_random_; ///< Feature cache
  CuMatrix<BaseFloat> features_leftover_; ///< Feature cache
  
  std::vector<int32> targets_;  ///< Desired vector cache
  std::vector<int32> targets_random_;  ///< Desired vector cache
  std::vector<int32> targets_leftover_;  ///< Desired vector cache

  std::vector<int32> randmask_;
  CuStlVector<int32> randmask_device_;

}; 
 
  
} // namespace nnet1
} // namespace kaldi

#endif
