// nnet3/decodable-batch-looped.h

// Copyright 2020-2021  Xiaomi Corporation (Author: Zhao Yan)

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

#ifndef KALDI_NNET3_DECODABLE_BATCH_LOOPED_H_
#define KALDI_NNET3_DECODABLE_BATCH_LOOPED_H_

#if HAVE_CUDA == 1
#include <chrono>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "base/kaldi-common.h"
#include "util/kaldi-semaphore.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/online-feature-itf.h"
#include "itf/decodable-itf.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"

namespace kaldi {
namespace nnet3 {

// See also nnet-am-decodable-simple.h, which is a decodable object that's based
// on breaking up the input into fixed chunks.  The decodable object defined here is based on
// 'looped' computations, which naturally handle infinite left-context (but are
// only ideal for systems that have only recurrence in the forward direction,
// i.e. not BLSTMs... because there isn't a natural way to enforce extra right
// context for each chunk.)


struct NnetBatchLoopedComputationOptions {
  int32 extra_left_context_initial;
  int32 frame_subsampling_factor;
  int32 frames_per_chunk;
  int32 max_batch_size;
  int32 compute_interval;
  BaseFloat acoustic_scale;
  bool debug_computation;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  NnetBatchLoopedComputationOptions():
      extra_left_context_initial(0),
      frame_subsampling_factor(1),
      frames_per_chunk(20),
      max_batch_size(16),
      compute_interval(2000),
      acoustic_scale(0.1),
      debug_computation(false) { }

  void Check() const {
    KALDI_ASSERT(extra_left_context_initial >= 0 &&
                 frame_subsampling_factor > 0 && frames_per_chunk > 0 &&
                 acoustic_scale > 0.0 && max_batch_size >= 2);
  }

  void Register(OptionsItf *opts) {
    opts->Register("extra-left-context-initial", &extra_left_context_initial,
                   "Extra left context to use at the first frame of an utterance (note: "
                   "this will just consist of repeats of the first frame, and should not "
                   "usually be necessary.");
    opts->Register("frame-subsampling-factor", &frame_subsampling_factor,
                   "Required if the frame-rate of the output (e.g. in 'chain' "
                   "models) is less than the frame-rate of the original "
                   "alignment.");
    opts->Register("max-batch-size", &max_batch_size,
                   "number of max sequences for decodable.");
    opts->Register("compute-interval", &compute_interval,
                   "how many microseconds to wait after one computation.");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic log-likelihoods");
    opts->Register("frames-per-chunk", &frames_per_chunk,
                   "Number of frames in each chunk that is separately evaluated "
                   "by the neural net.  Measured before any subsampling, if the "
                   "--frame-subsampling-factor options is used (i.e. counts "
                   "input frames.  This is only advisory (may be rounded up "
                   "if needed.");
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);

    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};


/**
   When you instantiate class DecodableNnetBatchLooped, you should give it
   a const reference to this class, that has been previously initialized.
 */
class DecodableNnetBatchLoopedInfo  {
 public:
  // The constructor takes a non-const pointer to 'nnet' because it may have to
  // modify it to be able to take multiple iVectors.
  DecodableNnetBatchLoopedInfo(const NnetBatchLoopedComputationOptions &opts,
                               Nnet *nnet);

  // This constructor takes the priors from class AmNnetSimple (so it can divide by
  // them).
  DecodableNnetBatchLoopedInfo(const NnetBatchLoopedComputationOptions &opts,
                               AmNnetSimple *nnet);

  // this constructor is for use in testing.
  DecodableNnetBatchLoopedInfo(const NnetBatchLoopedComputationOptions &opts,
                               const Vector<BaseFloat> &priors,
                               Nnet *nnet);

  void Init(const NnetBatchLoopedComputationOptions &opts,
            Nnet *nnet);

  const NnetBatchLoopedComputationOptions &opts;

  Nnet &nnet;

  // the log priors (or the empty vector if the priors are not set in the model)
  CuVector<BaseFloat> log_priors;


  // frames_left_context equals the model left context plus the value of the
  // --extra-left-context-initial option.
  int32 frames_left_context;
  // frames_right_context is the same as the right-context of the model.
  int32 frames_right_context;
  // The frames_per_chunk_ equals the number of input frames we need for each
  // chunk (except for the first chunk).  This divided by
  // opts_.frame_subsampling_factor gives the number of output frames.
  int32 frames_per_chunk;

  // The output dimension of the neural network.
  int32 output_dim;

  // True if the neural net accepts iVectors.  If so, the neural net will have been modified
  // to accept the iVectors
  bool has_ivectors;
  int32 num_chunk1_ivector_frames;
  int32 num_ivector_frames;

  // The 3 computation requests that are used to create the looped
  // computation are stored in the class, as we need them to work out
  // exactly shich iVectors are needed.
  ComputationRequest request1, request2, request3;

  // The compiled, 'looped' computation.
  std::vector<NnetComputation> computation;
};

class NotifiableNnetBatchLooped {
public:
  virtual void Receive(const CuMatrixBase<BaseFloat> &output) = 0;

  virtual ~NotifiableNnetBatchLooped() {}
};

struct NnetComputeRequest {
  Matrix<BaseFloat> inputs;                    // features
  Matrix<BaseFloat> ivectors;                  // ivectors if has ivector
  NnetComputeState state;                      // state of stream for computation
  NotifiableNnetBatchLooped *notifiable;       // notifiable which to push output
  bool first_chunk;                            // first chunk or not
};


/*
  This class handles the batch neural net computation.
  It will bind with an GPU device and start one thread for computation.
  It accepts requests of computation from decoding threads and put them
  in FIFO queue. The thread for computation takes out multiple requests 
  from the queue and runs inference in batch. After that, the thread 
  for computation wakes up the decoding threads to continue, with 
  inference results.
  The input is handled sequently chunk by chunk for any stream.
  It batches multiple chunks which are from different streams every
  time, and the streams in batch may change every time. So it doesn't 
  matter if some streams have more input than other.
*/
class NnetBatchLoopedComputer {
public:
  NnetBatchLoopedComputer(const DecodableNnetBatchLoopedInfo &info);

  inline const DecodableNnetBatchLoopedInfo &GetInfo() const { return info_; }
  
  // Enqueue request of computation from decoding thread
  void Enqueue(NnetComputeRequest *request);

  ~NnetBatchLoopedComputer();

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetBatchLoopedComputer);

  static void *ThreadFunction(void *para); 

  // Start a new thread to handle the computation in batch looped mode 
  void Start();
  
  // Stop the thread which handles the computation.
  // It should stop enqueue computation requests, 
  // and handle the remain requests.
  void Stop();

  // Advance fake chunks until the NnetComputer becomes stable.
  // The NnetComputer is unstable for the first few chunks, which 
  // means the state of NnetComputer will be changed chunk by chunk,
  // e.g. the matrix represents cell of LSTM, the matrix represents
  // buffers for TDNN.
  // When the NnetComputer becomes stable, which means all of the members
  // of NnetComputer become constant, we can get and set the state for 
  // any sequence in batch correctly.
  void AdvanceChunkUntilStable(int32 batch_size, std::vector<bool> *batch_first); 
  
  // Advance one chunk in bacth
  // The sequence represented by any request in batch may be different with last 
  // chunk, so we should set state for requests before inferene, and get 
  // state for requests after inference.
  void AdvanceChunk(const std::vector<NnetComputeRequest*> &requests);

  void Compute();

  bool Continue();
  
private:
  const DecodableNnetBatchLoopedInfo &info_;

  std::vector<NnetComputerSnapshot> snapshots_;

  // Continue computation or not.
  // It don't needs lock for multi-thread synchronization,
  // because the assignment is atomic.
  bool is_working_;
  
  // Queue contains requests waiting for computation.
  // It needs a mutex to help achieve multi-thread synchronization.
  // The first chunk is different with other chunks, so needs a 
  // queue to handle requests of first chunk only.
  typedef std::chrono::system_clock::time_point TimePoint;
  typedef std::pair<NnetComputeRequest*, TimePoint> QueueElement;
  std::queue<QueueElement> queue_; 
  std::queue<QueueElement> chunk1_queue_;
  std::mutex mtx_;
  std::condition_variable condition_variable_;

  // The thread for computation
  std::thread work_thread_;

  // The vector indicates the matrices stored state are batch first or not.
  std::vector<bool> batch_first_;
};


class DecodableNnetBatchLoopedOnline : public DecodableInterface, 
                                       public NotifiableNnetBatchLooped {
public:
  // Constructor.  'input_feature' is for the feature that will be given
  // as 'input' to the neural network; 'ivector_feature' is for the iVectors
  // OnlineFeatureInterface *ivector_features);
  DecodableNnetBatchLoopedOnline(NnetBatchLoopedComputer *computer,
                                 OnlineFeatureInterface *input_features,
                                 OnlineFeatureInterface *ivector_features);

  virtual BaseFloat LogLikelihood(int32 subsampled_frame, int32 index);

  void GetOutputForFrame(int32 subsampled_frame, VectorBase<BaseFloat> *output);
  
  // note: the frame argument is on the output of the network, i.e. after any
  // subsampling, so we call it 'subsampled_frame'.
  virtual bool IsLastFrame(int32 subsampled_frame) const;

  virtual int32 NumFramesReady() const;

  virtual int32 NumIndices() const { return info_.output_dim; }

  // this is not part of the standard Decodable interface but I think is needed for
  // something.
  int32 FrameSubsamplingFactor() const {
    return info_.opts.frame_subsampling_factor;
  }

  virtual void Receive(const CuMatrixBase<BaseFloat> &output) {
    if (current_log_post_.NumRows() != output.NumRows() ||
        current_log_post_.NumCols() != output.NumCols())
      current_log_post_.Resize(output.NumRows(), output.NumCols());
    current_log_post_.CopyFromMat(output);
    semaphone_.Signal();
  }

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableNnetBatchLoopedOnline);

  inline void EnsureFrameIsComputed(int32 subsampled_frame) {
    KALDI_ASSERT(subsampled_frame >= current_log_post_subsampled_offset_ &&
                 "Frames must be accessed in order.");
    while (subsampled_frame >= current_log_post_subsampled_offset_ +
           current_log_post_.NumRows())
      AdvanceChunk();
  }

  // This function does the computation for the next chunk.  It will change
  // current_log_post_ and current_log_post_subsampled_offset_, and
  // increment num_chunks_computed_.
  void AdvanceChunk();

  // The current log-posteriors that we got from the last time we
  // ran the computation.
  Matrix<BaseFloat> current_log_post_;

  // The number of chunks we have computed so far.
  int32 num_chunks_computed_;

  // The time-offset of the current log-posteriors, equals
  // (num_chunks_computed_ - 1) *
  //    (info_.frames_per_chunk_ / info_.opts_.frame_subsampling_factor).
  int32 current_log_post_subsampled_offset_;

  OnlineFeatureInterface *input_features_;
  OnlineFeatureInterface *ivector_features_;
  
  // This object will accepts computation requests from this,
  // and push outputs to this after computation finished.
  NnetBatchLoopedComputer *computer_;
  const DecodableNnetBatchLoopedInfo &info_;
    
  NnetComputeRequest request_;
  Semaphore semaphone_;
};

class DecodableAmNnetBatchLoopedOnline : public DecodableInterface {
public:
  // Constructor.  'input_feature' is for the feature that will be given
  // as 'input' to the neural network; 'ivector_feature' is for the iVectors
  // OnlineFeatureInterface *ivector_features);
  DecodableAmNnetBatchLoopedOnline(NnetBatchLoopedComputer *computer,
                                   const TransitionModel &trans_model,
                                   OnlineFeatureInterface *input_features,
                                   OnlineFeatureInterface *ivector_features);

  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual inline int32 NumFramesReady() const {
    return decodable_nnet_.NumFramesReady();
  }

  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const { 
    return decodable_nnet_.IsLastFrame(frame); 
  }
  
private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetBatchLoopedOnline);
  DecodableNnetBatchLoopedOnline decodable_nnet_;
  const TransitionModel &trans_model_;
};

/*
  This class handles the neural net computation; it's mostly accessed
  via other wrapper classes.

  It can accept just input features, or input features plus iVectors.  */
class DecodableNnetBatchSimpleLooped : public NotifiableNnetBatchLooped {
public:
  /**
     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] computer This class submit computation requests to computer,
                          which handles the neural net computation acturallly.
     @param [in] feats  The input feature matrix.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] online_ivectors
                        If you are using iVectors estimated 'online'
                        a pointer to the iVectors, else NULL.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
  */
  DecodableNnetBatchSimpleLooped(NnetBatchLoopedComputer *computer,
                                 const MatrixBase<BaseFloat> &feats,
                                 const VectorBase<BaseFloat> *ivector = NULL,
                                 const MatrixBase<BaseFloat> *online_ivectors = NULL,
                                 int32 online_ivector_period = 1);

  // returns the number of frames of likelihoods.  The same as feats_.NumRows()
  // in the normal case (but may be less if opts_.frame_subsampling_factor !=
  // 1).
  inline int32 NumFrames() const { return num_subsampled_frames_; }
  
  inline int32 OutputDim() const { return info_.output_dim; }

  // Gets the output for a particular frame and pdf_id, with
  // 0 <= subsampled_frame < NumFrames(),
  // and 0 <= pdf_id < OutputDim().
  inline BaseFloat GetOutput(int32 subsampled_frame, int32 pdf_id) {
    KALDI_ASSERT(subsampled_frame >= current_log_post_subsampled_offset_ &&
                 "Frames must be accessed in order.");
    while (subsampled_frame >= current_log_post_subsampled_offset_ +
                            current_log_post_.NumRows())
      AdvanceChunk();
    return current_log_post_(subsampled_frame -
                             current_log_post_subsampled_offset_,
                             pdf_id);
  }
  
  virtual void Receive(const CuMatrixBase<BaseFloat> &output) {
    if (current_log_post_.NumRows() != output.NumRows() ||
        current_log_post_.NumCols() != output.NumCols())
      current_log_post_.Resize(output.NumRows(), output.NumCols());
    current_log_post_.CopyFromMat(output);
    semaphone_.Signal();
  }

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableNnetBatchSimpleLooped);

  // This function does the computation for the next chunk.  It will change
  // current_log_post_ and current_log_post_subsampled_offset_, and
  // increment num_chunks_computed_.
  void AdvanceChunk();

  // Gets the iVector for the specified frame., if we are
  // using iVectors (else does nothing).
  void GetCurrentIvector(int32 input_frame, Vector<BaseFloat> *ivector);

  // The current log-posteriors that we got from the last time we
  // ran the computation.
  Matrix<BaseFloat> current_log_post_;

  // The number of chunks we have computed so far.
  int32 num_chunks_computed_;

  // The time-offset of the current log-posteriors, equals
  // (num_chunks_computed_ - 1) *
  //    (info_.frames_per_chunk_ / info_.opts_.frame_subsampling_factor).
  int32 current_log_post_subsampled_offset_;

  const MatrixBase<BaseFloat> &feats_;
  // note: num_subsampled_frames_ will equal feats_->.NumRows() in the normal case
  // when opts_.frame_subsampling_factor == 1.
  int32 num_subsampled_frames_;

  // ivector_ is the iVector if we're using iVectors that are estimated in batch
  // mode.
  const VectorBase<BaseFloat> *ivector_;

  // online_ivector_feats_ is the iVectors if we're using online-estimated ones.
  const MatrixBase<BaseFloat> *online_ivector_feats_;
  // online_ivector_period_ helps us interpret online_ivector_feats_; it's the
  // number of frames the rows of ivector_feats are separated by.
  int32 online_ivector_period_;

  // This object will accepts computation requests from this,
  // and push outputs to this after computation finished.
  NnetBatchLoopedComputer *computer_;
  const DecodableNnetBatchLoopedInfo &info_;
  
  // This is the struct represents computation request,
  // contains feats, state of computations, and point of 
  // notifiable which will be notified when computation finished.
  NnetComputeRequest request_;
  
  // The Semaphore used to synchronize with computer_.
  Semaphore semaphone_;
};

class DecodableAmNnetBatchSimpleLooped : public DecodableInterface {
public:
  /**
     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.


     @param [in] computer This class submit computation requests to computer,
                          which handles the neural net computation acturallly.
     @param [in] trans_model  The transition model to use.  This takes care of the
                        mapping from transition-id (which is an arg to
                        LogLikelihood()) to pdf-id (which is used internally).
     @param [in] feats   A pointer to the input feature matrix; must be non-NULL.
                         We
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] online_ivectors
                        If you are using iVectors estimated 'online'
                        a pointer to the iVectors, else NULL.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
  */
  DecodableAmNnetBatchSimpleLooped(
      NnetBatchLoopedComputer *computer,
      const TransitionModel &trans_model,
      const MatrixBase<BaseFloat> &feats,
      const VectorBase<BaseFloat> *ivector = NULL,
      const MatrixBase<BaseFloat> *online_ivectors = NULL,
      int32 online_ivector_period = 1);

  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual inline int32 NumFramesReady() const {
    return decodable_nnet_.NumFrames();
  }

  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetBatchSimpleLooped);
  DecodableNnetBatchSimpleLooped decodable_nnet_;
  const TransitionModel &trans_model_;
};

class DecodableAmNnetBatchSimpleLoopedParallel : public DecodableInterface {
public:
  /**
     This decodable object is for use in multi-threaded decoding.
     It differs from DecodableAmNnetBatchSimpleLooped in respect followed:
        (1) It doesn't keep around pointers to the features and iVectors;
            instead, it creates copies of them (so the caller can
            delete the originals).
     
     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.


     @param [in] computer This class submit computation requests to computer,
                          which handles the neural net computation acturallly.
     @param [in] trans_model  The transition model to use.  This takes care of the
                        mapping from transition-id (which is an arg to
                        LogLikelihood()) to pdf-id (which is used internally).
     @param [in] feats   A pointer to the input feature matrix; must be non-NULL.
                         We
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] online_ivectors
                        If you are using iVectors estimated 'online'
                        a pointer to the iVectors, else NULL.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
  */
  DecodableAmNnetBatchSimpleLoopedParallel(
      NnetBatchLoopedComputer *computer,
      const TransitionModel &trans_model,
      const MatrixBase<BaseFloat> &feats,
      const VectorBase<BaseFloat> *ivector = NULL,
      const MatrixBase<BaseFloat> *online_ivectors = NULL,
      int32 online_ivector_period = 1);

  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual inline int32 NumFramesReady() const {
    return decodable_nnet_->NumFrames();
  }

  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

  ~DecodableAmNnetBatchSimpleLoopedParallel() { DeletePointers(); }
private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetBatchSimpleLoopedParallel);
  void DeletePointers();

  DecodableNnetBatchSimpleLooped *decodable_nnet_;
  const TransitionModel &trans_model_;

  Matrix<BaseFloat> feats_copy_;
  Vector<BaseFloat> *ivector_copy_;
  Matrix<BaseFloat> *online_ivectors_copy_;
};


} // namespace nnet3
} // namespace kaldi

#endif  // HAVE_CUDA
#endif  // KALDI_NNET3_DECODABLE_BATCH_LOOPED_H_
