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




/**
   class NnetComputationTask represents a chunk of an utterance that is
   requested to be computed.  This will be given to NnetBatchComputer, which
   will aggregate the tasks and complete them.
 */
struct NnetComputationTask {
  // The input frames, which are treated as being numbered t=0, t=1, etc.  (If
  // the lowest t value was originally nonzero in the 'natural' numbering, this
  // just means we conceptually shift the 't' values; the only real constraint
  // is that the 't' values are contiguous.
  Matrix<BaseFloat> input_frames;

  // The index of the first output frame (in the shifted numbering where the
  // first output frame is numbered zero.  This will typically be less than one,
  // because most network topologies require left context.  If this was an
  // 'interior' chunk of a recurrent topology like LSTMs, first_input_t may be
  // substantially less than zero, due to 'extra_left_context'.
  int32 first_input_t;

  // The stride of output 't' values: e.g., will be 1 for normal-frame-rate
  // models, and 3 for low-frame-rate models such as chain models.
  int32 output_t_stride;
  // The number of output 't' values (they will start from zero and be separated
  // by output_t_stride).  This will be the num-rows of 'output'.
  int32 num_output_frames;

  // 'num_initial_unused_output_frames', which will normally be zero, is the number
  // of rows of the output matrix ('output' which won't actually be needed by
  // the user, usually because they overlap with a previous chunk (this can
  // happen because the number of outputs isn't a multiple of the number of
  // chunks).
  int32 num_initial_unused_output_frames;
  // num_used_output_frames, which must be <= num_output_frames -
  // num_initial_unused_output_frames, is the number of output frames which are
  // actually going to be used by the user.  (Due to edge effects, not all are
  // necessarily used).
  int32 num_used_output_frames;

  // True if this chunk is an 'edge' (the beginning or end of an utterance) AND
  // is structurally different somehow from non-edge chunk, e.g. requires less
  // context.  This is present only so that NnetBatchComputer will know the
  // appropriate minibatch size to use.
  bool is_edge;

  // The i-vector for this chunk, if this network accepts i-vector inputs.
  Vector<BaseFloat> ivector;

  // A priority (lower is more urgent).  May be updated after this object is
  // provided to class NnetBatchComputer.
  int64 priority;

  // This semaphore will be incremented by class NnetBatchComputer when this
  // chunk is done.  After this semaphore is incremented, class
  // NnetBatchComputer will no longer hold any pointers to this class.
  Semaphore semaphore;

  // Will be set to true by the caller if they want the output of the neural net
  // to be copied to CPU (to 'output').  If false, the output will stay on
  // the GPU (if used)- in cu_output.
  bool output_to_cpu;

  // The neural net output, of dimension num_output_frames by the output-dim of
  // the neural net, will be written to 'output_cpu' if 'output_to_cpu' is true.
  // This is expected to be empty when this task is provided to class
  // NnetBatchComputer, and will be nonempty (if output_to_cpu == true) when the
  // task is completed and the semaphore is signaled.
  Matrix<BaseFloat> output_cpu;

  // The output goes here instead of 'output_to_cpu' is false.
  CuMatrix<BaseFloat> output;

  // Returns true if these two tasks can be part of the same minibatch.  There
  // are certain things that does *doesn't* check, such as the presence/absence
  // of an i-vector and output_t_stride, because they are expected to always be
  // the same in the context where you'd call this function.
  //bool IsCompatible(const NnetComputationTask &other) const {
  //return (input_frames.NumRows() == other.input_frames.NumRows() &&
  //first_input_t == other.first_input_t &&
  //num_output_frames == other.num_output_frames);
  //}
};


struct NnetBatchComputerOptions: public NnetSimpleComputationOptions {
  int32 minibatch_size;
  int32 edge_minibatch_size;
  bool ensure_exact_final_context;
  BaseFloat partial_minibatch_factor;

  NnetBatchComputerOptions(): minbatch_size(128),
                              edge_minibatch_size(32),
                              ensure_exact_final_context(false),
                              partial_minibatch_factor(0.5) {
  }

  void Register(OptionsItf *po) {
    NnetSimpleComputationOptions::Register(opts);
    po->Register("minibatch-size", &minibatch_size, "Number of chunks per "
                 "minibatch (see also edge-minibatch-size)");
    po->Register("edge-minibatch-size", &edge_minibatch_size, "Number of "
                 "chunks per minibatch: this applies to chunks at the "
                 "beginnings and ends of utterances, in cases (such as "
                 "recurrent models) when the computation would be different "
                 "from the usual one.");
    po->Register("ensure-exact-final-context", &ensure_exact_final_context,
                 "If true, for utterances shorter than --frames-per-chunk, "
                 "use exact-length, special computations.  If false, "
                 "pad with repeats of the last frame.  Would only affect "
                 "the output for backwards-recurrent models, but would "
                 "negatively impact speed in all cases.");
    po->Register("partial-minibatch-factor", &partial_minibatch_factor,
                 "Factor that controls how small partial minibatches will be "
                 "they become necessary.  We will potentially do the computation "
                 "for sizes: int(partial_minibatch_factor^n * minibatch_size "
                 ", for n = 0, 1, 2...");
  }
};


/**
   Split a single utterance into a list of separate tasks to give to class
   NnetBatchComputer.  This version is for when either you don't have i-vectors
   (ivector == NULL) or you have a single i-vector for the entire file.
   The other version (below) has more extensive documentation.
*/
void SplitUtteranceIntoTasks(
    const NnetBatchComputerOptions &opts,
    int32 nnet_left_context,
    int32 nnet_right_context,
    bool output_to_cpu,
    const Matrix<BaseFloat> &input,
    const Vector<BaseFloat> *ivector,
    std::vector<NnetComputationTask> *tasks);

/**
   Split a single utterance into a list of separate tasks to give to class
   NnetBatchComputer.  This version is for when you have 'online' i-vectors,
   i.e.  multiple i-vectors per utterance.
     @param [in] opts  Options class, e.g. used to get minibatch size.
     @param [in] nnet_left_context  This, and nnet_right_context, should be the
              result of a call like this:
        ComputeSimpleNnetContext(nnet, &nnet_left_context_, &nnet_right_context_);
     @param [in] nnet_right_context see above.
     @param [in] output_to_cpu  Will become the 'output_to_cpu' member of the
             output tasks; this controls whether the computation code should transfer
             the outputs to CPU (which is to save GPU memory).
     @param [in] online_ivectors  Matrix of ivectors, one every 'online_ivector_period' frames.
     @param [in] online_ivector_period  Affects the interpretation of 'online_ivectors'.
     @param [out]  tasks       The tasks created will be output to here.  The
                      priorities will be set to zero; setting them to a meaningful
                      value is up to the caller.
*/
void SplitUtteranceIntoTasks(
    const NnetBatchComputerOptions &opts,
    int32 nnet_left_context,
    int32 nnet_right_context,
    bool output_to_cpu,
    const Matrix<BaseFloat> &input,
    const Matrix<BaseFloat> &online_ivectors,
    int32 online_ivector_period,
    std::vector<NnetComputationTask> *tasks);


/**
   This class does neural net inference in a way that is optimized for GPU use:
   it combines chunks of multiple utterances into minibatches for more efficient
   computation.  It is thread safe, i.e. you can call it from multiple threads
   without having to worry about data races and the like.  (However, you are
   expected to call the Compute() function from only one thread).

   Note: it stores references to all arguments to the constructor, so don't
   delete them till this goes out of scope.

   @param [in] opts   The options class.  Warning: it includes an acoustic
          weight, whose default is 0.1; you may sometimes want to
          change this to 1.0.
   @param [in] nnet
          The neural net that we're going to do the computation with
   @param [in] priors
          Vector of priors-- if supplied and nonempty, we subtract
          the log of these priors from the nnet output.
*/
class NnetBatchComputer {
 public:
  /**  Constructor.

       \param [in] opts  Options struct
       \param [in] nnet  The neural net which we'll be doing the computation with
       \param [in] priors  Either zero, or a vector of prior probabilities which
                        we'll take the log of and subtract from the neural net
                        outputs (e.g. used in non-chain systems).
   */
  NnetBatchComputer(const NnetBatchComputerOptions &opts,
                    const Nnet &nnet,
                    const VectorBase<BaseFloat> &priors);


  /// Accepts a task, meaning the task will be queued.  (Note: the pointer is
  /// still owned by the caller.
  void AcceptTask(NnetComputationTask *task);

  /// Returns the number of tasks currently queued.
  int32 NumTasksQueued() const;

  /// Returns true if at least one full minibatch is ready to compute.
  bool FullMinibatchReady() const;


  /***
      Does some kind of computation, choosing the highest-priority thing to
      compute; this will be a full minibatch if FullMinibatchReady() returned
      true, and a partial one if not, and if tasks were queued (NumTasksQueued()
      > 0).  It returns true if it did some kind of computation, and false
      otherwise.  This function locks the class, but not for the entire time
      it's being called: only at the beginning and at the end.
   */
  bool Compute();

  ~NnetBatchComputer();

 private:
  // Mutex that guards this object.  It is only held for fairly quick operations
  // (not while the actual computation is being done).
  std::mutex mutex_;



  typedef unordered_map<ComputationGroupKey, ComputationGroupInfo,
                        ComputationGroupKeyHasher> MapType;
  // tasks_ contains all the queued tasks.
  // Each key contains a vector of NnetComputationTask* pointers, of the same
  // structure (i.e., IsCompatible() returns true).
  MapType tasks_;

  // Gets the priority for a group.  (A group is a list of tasks that may be
  // computed in the same minibatch).  Lower priority is more important.
  // What this function does is a kind of heuristic.
  int64 GetPriority(const std::vector<NnetComputationTask*> > &group);


  // This function either compiles and caches (in tasks_) a computation, or
  // retrieves it from tasks_ and returns it.
  std::shared_ptr<const NnetComputation> GetComputation(
      const ComputationGroupKey &key,
      int32 minibatch_size);


  // This function finds the highest-priority group of tasks, removes a
  // minibatch's worth of them from 'tasks_' and puts them into the
  // caller-supplied array 'tasks', ensures that a computation has been compiled
  // for this size of task.  It locks 'mutex_' for its duration.
  // Will return NULL if there are no tasks to compute.
  std::shared_ptr<const NnetComputation> FindHighestPriorityGroup(
      ComputationGroupKey *key,
      std::vector<NnetComputationTask*> *tasks);


  // Changes opts_.frames_per_chunk to be a multiple of
  // opts_.frame_subsampling_factor, if needed.
  void CheckAndFixConfigs();

  // this function creates and returns the computation request which is to be
  // compiled.
  static void GetComputationRequest(const NnetComputationTask &task,
                                    int32 minibatch_size,
                                    ComputationRequest *request);

  struct ComputationGroupInfo {
    std::vector<NnetComputationTask*> tasks;
    // map from minibatch-size to a pointer to the appropriate NnetComputation.
    std::map<int32, std::shared_ptr<const NnetComputation> > computation;
  }

  // This struct allows us to arrange the tasks into groups that can be
  // computed in the same minibatch.
  struct ComputationGroupKey {
    ComputationGroupKey(const NnetComputationTask &task):
        num_input_frames(task.input_matrix.NumRows()),
        first_output_t(task.first_output_t),
        num_output_frames(task.num_output_frames) {}

    bool operator == (const ComputationGroupKey &other) const {
      return num_input_frames == other.num_input_frames &&
          first_input_t == other.first_input_t &&
          num_output_frames == other.num_output_frames;
    }
    int32 num_input_frames;
    int32 first_input_t;
    int32 num_output_frames;
  };
  struct ComputationGroupKeyHasher {
    int32 operator () const (const ComputationGroupKey &key) {
      return key.num_input_frames + 18043 * key.first_input_t +
          6413 * num_output_frames;
    }
  };

  NnetBatchComputerOptions opts_;
  const Nnet &nnet_;
  CachingOptimizingCompiler compiler_;
  CuVector<BaseFloat> log_priors_;
  int32 output_dim_;



};

/**
   This class does neural net inference in a way that is optimized for GPU use:
   it combines chunks of multiple utterances into minibatches for more efficient
   computation.

   Note: it stores references to all arguments to the constructor, so don't
   delete them till this goes out of scope.

   @param [in] opts   The options class.  Warning: it includes an acoustic
          weight, whose default is 0.1; you may sometimes want to
          change this to 1.0.
   @param [in] nnet
          The neural net that we're going to do the computation with
   @param [in] priors
          Vector of priors-- if supplied and nonempty, we subtract
          the log of these priors from the nnet output.
   @param [in] online_ivector_period
           If you are using iVectors estimated 'online'
           (i.e. if online_ivectors != NULL) gives the periodicity
           (in input frames) with which the iVectors are estimated, e.g. 10.
   @param [in] ensure_exact_final_context
           If an utterance length is less than opts_.frames_per_chunk, we call
           it a "shorter-than-chunk-size" utterance.  If
           ensure_exact_final_context is true, for such utterances we will
           create a special computation just for them, that has the exact
           right size.  (This is necessary for things like BLSTMs, to get
           the correct output).  If false, we will just pad on the right
           with repeats of the last frame (which is fine for topologies
           that are not backwards recurrent).
   @param [in] minibatch_size  The number of separate chunks that we
           process in each minibatch.  (The number of frames per chunk
           comes from 'opts').
*/
class NnetBatchComputer {
 public:
  BatchNnetComputer(const NnetSimpleComputationOptions &opts,
                    const Nnet &nnet,
                    const VectorBase<BaseFloat> &priors,
                    int32 online_ivector_period = 0,
                    bool  ensure_exact_final_context = false,
                    int32 minibatch_size = 128);
  ~BatchNnetComputer();


  /**
     You call AcceptInput() for each feature file that you are going to want it
     to decode.


     TODO..
     It takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     BatchNnetComputer takes the ownership of the three pointers, and they
     will be released in function Clear().
  */
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

  // The stucture records the information of each chunk in current batch. It
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
