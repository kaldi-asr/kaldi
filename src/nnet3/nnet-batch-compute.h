// nnet3/nnet-batch-compute.h

// Copyright 2012-2018  Johns Hopkins University (author: Daniel Povey)
//           2018       Hang Lyu

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

#ifndef KALDI_NNET3_NNET_BATCH_COMPUTE_H_
#define KALDI_NNET3_NNET_BATCH_COMPUTE_H_

#include <vector>
#include <string>
#include <list>
#include <utility>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "util/stl-utils.h"


namespace kaldi {
namespace nnet3 {

/*
  This class handles the neural net computation in parallel.

  It can accept just input features, or input features plus iVectors.  */
class BatchNnetComputer {
 public:
  /**
     This class does neural net inference in a way that is optimized for GPU
     use: it combines chunks of multiple utterances into minibatche for more
     efficient computation.
     
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] opts   The options class.  Warning: it includes an acoustic
                        weight, whose default is 0.1; you may sometimes want to
                        change this to 1.0.
     @param [in] nnet   The neural net that we're going to do the computation with
     @param [in] priors Vector of priors-- if supplied and nonempty, we subtract
                        the log of these priors from the nnet output.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
     @param [in] ensure_exact_final_context If an utterance length is less than
                        opts_.frames_per_chunk, we call it "shorter-than-chunk
                        -size" utterance. This option is used to control whether
                        we deal with "shorter-than-chunk-size" utterances
                        specially. It is useful in some models, such as blstm.
                        If it is true, its "t" indexs will from 
                        "-opts_.extra_left_context_initial - nnet_left_context_"
                        to "chunk_length + nnet_right_context_ + 
                        opts_.extra_right_context_final". Otherwise, it will be
                        from "-opts_.extra_left_context_initial - nnet_left_context_"
                        to "opts_.frames_per_chunk + nnet_right_context_ + 
                        opts_.extra_right_context"
     @param [in] minibatch_size The capacity of the minibatch. In general, it 
                        means the number of chunks will be processed. The chunk
                        size comes from opts, and it may be adjusted slightly
                        by CheckAndFixConfigs() function according to 
                        nnet_modulus and frame_subsampling_factor.
  */
  BatchNnetComputer(const NnetSimpleComputationOptions &opts,
                    const Nnet &nnet,
                    const VectorBase<BaseFloat> &priors,
                    int32 online_ivector_period = 0,
                    bool  ensure_exact_final_context = false,
                    int32 minibatch_size = 128);
  ~BatchNnetComputer();

  // It takes features as input, and you can either supply a
  // single iVector input, estimated in batch-mode ('ivector'), or 'online'
  // iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
  // BatchNnetComputer takes the ownership of the three pointers, and they
  // will be released in function Clear().
  void AcceptInput(const std::string &utt_id,
                   const Matrix<BaseFloat> *feats,  // takes the ownership of
                                                    // the below pointers
                   const Vector<BaseFloat> *ivector = NULL,
                   const Matrix<BaseFloat> *online_ivectors = NULL);


  // Gets the output for a finished utterance. It will return quickly.
  // Note: The utterances which are going to be returned are in the same order
  // as they were provided to the class.
  bool GetFinishedUtterance(std::string *uttid,
      Matrix<BaseFloat> *output_matrix);

  // It completes the primary computation task.
  // If 'flush == true', it would ensure that even if a batch wasn't ready,
  // the computation would be run.
  // If 'flush == false', it would ensure that a batch was ready.
  void Compute(bool flush);

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BatchNnetComputer);

  // If true, it means the class has enough data in minibatch, so we can call
  // DoNnetComputation(). It is called from Compute() with "flush == false".
  inline bool Ready() const {
    for (BatchInfoMap::const_iterator iter =
        batch_info_.begin(); iter != batch_info_.end(); iter++) {
      if ((iter->second)->size() == minibatch_size_ )
        return true;
    }
    return false;
  }

  // If true, it means the class has no data need to be computed. It is called
  // from Compute() with "flush==true"
  inline bool Empty() const {
    for (BatchInfoMap::const_iterator iter =
        batch_info_.begin(); iter != batch_info_.end(); iter++) {
      if ((iter->second)->size() != 0)
        return false;
    }
    return true;
  }

  // When an utterance is taken by GetFinishedUtterance(), clear its information
  // in this class and release the memory on the heap.
  void Clear(std::string utt_id);

  // According to the information in batch_info_, prepare the 'batch' data,
  // ComputationRequest, compute and get the results out.
  void DoNnetComputation();
  // If ensure_exact_final_context is true, this function is used to deal with
  // "shorter than chunk size" utterances. In this function, we have to build
  // a new CompuationRequest.
  void DoNnetComputationOnes();

  // Gets the iVector that will be used for this chunk of frames, if we are
  // using iVectors (else does nothing).  note: the num_output_frames is
  // interpreted as the number of t value, which in the subsampled case is not
  // the same as the number of subsampled frames (it would be larger by
  // opts_.frame_subsampling_factor).
  void GetCurrentIvector(std::string utt_id,
                         int32 output_t_start,
                         int32 num_output_frames,
                         Vector<BaseFloat> *ivector);

  // called from constructor
  void CheckAndFixConfigs();

  // called from AcceptInput()
  void CheckInput(const Matrix<BaseFloat> *feats,
                  const Vector<BaseFloat> *ivector = NULL,
                  const Matrix<BaseFloat> *online_ivectors = NULL);

  // called from AcceptInput() or Compute(). Prepare the batch_info_ which
  // will be used to compute.
  void PrepareBatchInfo();

  // called from constructor. According to (tot_left_context,tot_right_context),
  // which equals to the model left/right context plus the extra left/right
  // context, we prepare the frequently-used ComputationRequest.
  // The CompuationRequest will be delete in deconstructor.
  // Otherwise, we will initialize the "batch_info_" map which is used to
  // maintain each information entry in (tot_left_context, tot_right_context)
  // batch. The "batch_info_" map will be delete in deconstructor.
  void PrepareComputationRequest();

  NnetSimpleComputationOptions opts_;
  const Nnet &nnet_;
  int32 nnet_left_context_;
  int32 nnet_right_context_;
  int32 output_dim_;
  // the log priors (or the empty vector if the priors are not set in the model)
  CuVector<BaseFloat> log_priors_;

  std::unordered_map<std::string, const Matrix<BaseFloat> *,
                     StringHasher> feats_;

  // ivector_ is the iVector if we're using iVectors that are estimated in batch
  // mode.
  std::unordered_map<std::string, const Vector<BaseFloat> *,
                     StringHasher> ivectors_;

  // online_ivector_feats_ is the iVectors if we're using online-estimated ones.
  std::unordered_map<std::string, const Matrix<BaseFloat> *,
                     StringHasher> online_ivector_feats_;

  // online_ivector_period_ helps us interpret online_ivector_feats_; it's the
  // number of frames the rows of ivector_feats are separated by.
  int32 online_ivector_period_;

  // an object of CachingOptimizingCompiler. For speed, except for "shorter
  // than chunk size" batch, we always get the ComputationRequest from
  // context_to_request_. Then we use CachingOptimizingCompiler to compiler
  // the CompuationRequest once, when it was first used.
  CachingOptimizingCompiler compiler_;

  // The current log-posteriors that we got from the last time we
  // ran the computation. The key is utterance-id. And the value is the
  // corresponding matrix which is allocated in function AcceptInput().
  // The content will be updated in function DoNnetComputation().
  // At last, when the utterance is completed, the space will be released
  // in function Clear().
  std::unordered_map<std::string, Matrix<BaseFloat>*, StringHasher> log_post_;

  // note: num_subsampled_frames_ will equal feats_.NumRows() in the normal case
  // when opts_.frame_subsampling_factor == 1.
  std::unordered_map<std::string, int32, StringHasher> num_subsampled_frames_;

  std::unordered_map<std::string, bool, StringHasher> is_computed_;
  // store each utterance id in order. We don't use a queue for here as,
  // in function "compute()" which is a blocking call, we will keep on
  // computing without taking the results out.
  std::list<std::string> utt_list_;

  // The sturcture records the information of each chunk in current batch. It
  // is used to point out how to organize the input data into batch chunk by
  // chunk and how to fetch the output data into corresponding place.
  struct BatchInfo {
    std::string utt_id;  // the utterance id. Index input map (feats_) and
                         // output map (log_post_) and so on.
    int32 first_input_frame_index;  // The first input frame index of
                                    // input feature matrix
    int32 last_input_frame_index;  // The last input frame index of
                                   // input feature matrix
    int32 first_output_subsampled_frame_index;  // The first output frame index
                                                // of output matrix
    int32 last_output_subsampled_frame_index;  // The last output frame index
                                               // of output matrix
    int32 output_offset;  // The offset index of output. It is useful in
                          // overlap with previous chunk circumstance. Transit
                          // the output index.
  };
  typedef std::list<BatchInfo> BatchInfoQueue;

  // store the information of the current batch. The key is pair
  // (tot_left_context, tot_right_context) which would equal the model
  // left/right context plus the extra left/right context. Each key corrsponds
  // to a kind of batch. When ensure_exact_final_context is true, (-1, -1) will
  // indexes those "shorter than chunk size" utterances.
  typedef std::unordered_map<std::pair<int32, int32>, BatchInfoQueue*,
                             PairHasher<int32, int32> > BatchInfoMap;
  BatchInfoMap batch_info_;
  BatchInfo last_batch_info_;

  // The key is (tot_left_context, tot_right_context), which would equal the
  // model left/right context plus the extra left/right context. The value is
  // corresponding ComputationRequest pointer. It is updated in function
  // PrepareCompuationRequest().
  typedef std::unordered_map<std::pair<int32, int32>, ComputationRequest *,
                             PairHasher<int32, int32> > ComputationRequestMap;
  ComputationRequestMap context_to_request_;

  // If an utterance length is less than opts_.frames_per_chunk, we call it
  // "shorter-than-chunk-size" utterance. This option is used to control whether
  // we deal with "shorter-than-chunk-those" utterances specially.
  // It is useful in some models, such as blstm.
  // If it is true, its "t" indexs will from
  // "-opts_.extra_left_context_initial - nnet_left_context_" to
  // "chunk_length + nnet_right_context_ + opts_.extra_right_context_final".
  // Otherwise, it will be from
  // "-opts_.extra_left_context_initial - nnet_left_context_" to
  // "opts_.frames_per_chunk + nnet_right_context_ + opts_.extra_right_context".
  bool ensure_exact_final_context_;

  int32 minibatch_size_;
};


}  // namespace nnet3
}  // namespace kaldi

#endif  // KALDI_NNET3_NNET_BATCH_COMPUTE_H_
