// nnet3/nnet-batch-compute.h

// Copyright 2012-2018  Johns Hopkins University (author: Daniel Povey)
//                2018       Hang Lyu

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
#include <condition_variable>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "decoder/lattice-faster-decoder.h"
#include "util/stl-utils.h"


namespace kaldi {
namespace nnet3 {


/**
   class NnetInferenceTask represents a chunk of an utterance that is
   requested to be computed.  This will be given to NnetBatchComputer, which
   will aggregate the tasks and complete them.
 */
struct NnetInferenceTask {
  // The copy constructor is required to exist because of std::vector's resize()
  // function, but in practice should never be used.
  NnetInferenceTask(const NnetInferenceTask &other) {
    KALDI_ERR << "NnetInferenceTask was not designed to be copied.";
  }
  NnetInferenceTask() { }


  // The input frames, which are treated as being numbered t=0, t=1, etc.  (If
  // the lowest t value was originally nonzero in the 'natural' numbering, this
  // just means we conceptually shift the 't' values; the only real constraint
  // is that the 't' values are contiguous.
  CuMatrix<BaseFloat> input;

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

  // 'num_initial_unused_output_frames', which will normally be zero, is the
  // number of rows of the output matrix ('output' or 'output_cpu') which won't
  // actually be needed by the user, usually because they overlap with a
  // previous chunk.  This can happen because the number of outputs isn't a
  // multiple of the number of chunks.
  int32 num_initial_unused_output_frames;

  // 0 < num_used_output_frames <= num_output_frames - num_initial_unused_output_frames
  // is the number of output frames which are actually going to be used by the
  // user.  (Due to edge effects, not all are necessarily used).
  int32 num_used_output_frames;

  // first_used_output_frame_index is provided for the convenience of the user
  // so that they can know how this chunk relates to the utterance which it is
  // a part of.
  // It represents an output frame index in the original utterance-- after
  // subsampling; so not a 't' value but a 't' value divided by
  // frame-subsampling-factor.  Specifically, it tells you the row index in the
  // full utterance's output which corresponds to the first 'used' frame index
  // at the output of this chunk, specifically: the row numbered
  // 'num_initial_unused_output_frames' in the 'output' or 'output_cpu' data
  // member.
  int32 first_used_output_frame_index;

  // True if this chunk is an 'edge' (the beginning or end of an utterance) AND
  // is structurally different somehow from non-edge chunk, e.g. requires less
  // context.  This is present only so that NnetBatchComputer will know the
  // appropriate minibatch size to use.
  bool is_edge;

  // True if this task represents an irregular-sized chunk.  These can happen
  // only for utterances that are shorter than the requested minibatch size, and
  // it should be quite rare.  We use a minibatch size of 1 in this case.
  bool is_irregular;

  // The i-vector for this chunk, if this network accepts i-vector inputs.
  CuVector<BaseFloat> ivector;

  // A priority (higher is more urgent); may be either sign.  May be updated
  // after this object is provided to class NnetBatchComputer.
  double priority;

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
};


struct NnetBatchComputerOptions: public NnetSimpleComputationOptions {
  int32 minibatch_size;
  int32 edge_minibatch_size;
  bool ensure_exact_final_context;
  BaseFloat partial_minibatch_factor;

  NnetBatchComputerOptions(): minibatch_size(128),
                              edge_minibatch_size(32),
                              ensure_exact_final_context(false),
                              partial_minibatch_factor(0.5) {
  }

  void Register(OptionsItf *po) {
    NnetSimpleComputationOptions::Register(po);
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
                 ", for n = 0, 1, 2....  Set it to 0.0 if you want to use "
                 "only the specified minibatch sizes.");
  }
};


/**
   Merges together the 'output_cpu' (if the 'output_to_cpu' members are true) or
   the 'output' members of 'tasks' into a single CPU matrix 'output'.  Requires that
   those outputs are nonempty (i.e. that those tasks must have been completed).

   @param [in] tasks  The vector of tasks whose outputs are to be merged.
         The tasks must have already been completed.
   @param [output  output  The spliced-together output matrix

   TODO: in the future, maybe start from GPU and use pinned matrices for the
   transfer.
 */
void MergeTaskOutput(
    const std::vector<NnetInferenceTask> &tasks,
    Matrix<BaseFloat> *output);
void MergeTaskOutput(
    const std::vector<NnetInferenceTask> &tasks,
    CuMatrix<BaseFloat> *output);

/**
   This class does neural net inference in a way that is optimized for GPU use:
   it combines chunks of multiple utterances into minibatches for more efficient
   computation.  It does the computation in one background thread that accesses
   the GPU.  It is thread safe, i.e. you can call it from multiple threads
   without having to worry about data races and the like.
*/
class NnetBatchComputer {
 public:
  /**  Constructor.  It stores references to all the arguments, so don't delete
       them till this object goes out of scop.

       \param [in] opts  Options struct
       \param [in] nnet  The neural net which we'll be doing the computation with
       \param [in] priors Either the empty vector, or a vector of prior
                        probabilities which we'll take the log of and subtract
                        from the neural net outputs (e.g. used in non-chain
                        systems).
   */
  NnetBatchComputer(const NnetBatchComputerOptions &opts,
                    const Nnet &nnet,
                    const VectorBase<BaseFloat> &priors);


  /// Accepts a task, meaning the task will be queued.  (Note: the pointer is
  /// still owned by the caller.
  /// If the max_minibatches_full >= 0, then the calling thread will block until
  /// no more than that many full minibatches are waiting to be computed.  This
  /// is a mechanism to prevent too many requests from piling up in memory.
  void AcceptTask(NnetInferenceTask *task,
                  int32 max_minibatches_full = -1);

  /// Returns the number of full minibatches waiting to be computed.
  int32 NumFullPendingMinibatches() const { return num_full_minibatches_; }


  /**
      Does some kind of computation, choosing the highest-priority thing to
      compute.  It returns true if it did some kind of computation, and false
      otherwise.  This function locks the class, but not for the entire time
      it's being called: only at the beginning and at the end.
        @param [in] allow_partial_minibatch  If false, then this will only
              do the computation if a full minibatch is ready; if true, it
              is allowed to do computation on partial (not-full) minibatches.
   */
  bool Compute(bool allow_partial_minibatch);


  /**
     Split a single utterance into a list of separate tasks which can then
     be given to this class by AcceptTask().

     @param [in] output_to_cpu  Will become the 'output_to_cpu' member of the
             output tasks; this controls whether the computation code should transfer
             the outputs to CPU (which is to save GPU memory).
     @param [in] ivector  If non-NULL, and i-vector for the whole utterance is
             expected to be supplied here (and online_ivectors should be NULL).
             This is relevant if you estimate i-vectors per speaker instead of
             online.
     @param [in] online_ivectors  Matrix of ivectors, one every 'online_ivector_period' frames.
     @param [in] online_ivector_period  Affects the interpretation of 'online_ivectors'.
     @param [out]  tasks       The tasks created will be output to here.  The
                      priorities will be set to zero; setting them to a meaningful
                      value is up to the caller.
  */
  void SplitUtteranceIntoTasks(
      bool output_to_cpu,
      const Matrix<BaseFloat> &input,
      const Vector<BaseFloat> *ivector,
      const Matrix<BaseFloat> *online_ivectors,
      int32 online_ivector_period,
      std::vector<NnetInferenceTask> *tasks);
  void SplitUtteranceIntoTasks(
      bool output_to_cpu,
      const CuMatrix<BaseFloat> &input,
      const CuVector<BaseFloat> *ivector,
      const CuMatrix<BaseFloat> *online_ivectors,
      int32 online_ivector_period,
      std::vector<NnetInferenceTask> *tasks);

  const NnetBatchComputerOptions &GetOptions() { return opts_; }

  ~NnetBatchComputer();

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetBatchComputer);

  // Information about a specific minibatch size for a group of tasks sharing a
  // specific structure (in terms of left and right context, etc.)
  struct MinibatchSizeInfo {
    // the computation for this minibatch size.
    std::shared_ptr<const NnetComputation> computation;
    int32 num_done;  // The number of minibatches computed: for diagnostics.
    int64 tot_num_tasks;  // The total number of tasks in those minibatches,
    // also for diagnostics... can be used to compute
    // how 'full', on average, these minibatches were.
    double seconds_taken;  // The total time elapsed in computation for this
                          // minibatch type.
    MinibatchSizeInfo(): computation(NULL), num_done(0),
                         tot_num_tasks(0), seconds_taken(0.0) { }
  };


  // A computation group is a group of tasks that have the same structure
  // (number of input and output frames, left and right context).
  struct ComputationGroupInfo {
    // The tasks to be completed.  This array is added-to by AcceptTask(),
    // and removed-from by GetHighestPriorityComputation(), which is called
    // from Compute().
    std::vector<NnetInferenceTask*> tasks;

    // Map from minibatch-size to information specific to this minibatch-size,
    // including the NnetComputation.  This is set up by
    // GetHighestPriorityComputation(), which is called from Compute().
    std::map<int32, MinibatchSizeInfo> minibatch_info;
  };

  // This struct allows us to arrange the tasks into groups that can be
  // computed in the same minibatch.
  struct ComputationGroupKey {
    ComputationGroupKey(const NnetInferenceTask &task):
        num_input_frames(task.input.NumRows()),
        first_input_t(task.first_input_t),
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
    int32 operator () (const ComputationGroupKey &key) const {
      return key.num_input_frames + 18043 * key.first_input_t +
          6413 * key.num_output_frames;
    }
  };


  typedef unordered_map<ComputationGroupKey, ComputationGroupInfo,
                        ComputationGroupKeyHasher> MapType;

  // Gets the priority for a group, higher means higher priority.  (A group is a
  // list of tasks that may be computed in the same minibatch).  What this
  // function does is a kind of heuristic.
  // If allow_partial_minibatch == false, it will set the priority for
  // any minibatches that are not full to negative infinity.
  inline double GetPriority(bool allow_partial_minibatch,
                            const ComputationGroupInfo &info) const;

  // Returns the minibatch size for this group of tasks, i.e. the size of a full
  // minibatch for this type of task, which is what we'd ideally like to
  // compute.  Note: the is_edge and is_irregular options should be the same
  // for for all tasks in the group.
  //   - If 'tasks' is empty or info.is_edge and info.is_irregular are both,
  //     false, then return opts_.minibatch_size
  //   - If 'tasks' is nonempty and tasks[0].is_irregular is true, then
  //     returns 1.
  //   - If 'tasks' is nonempty and tasks[0].is_irregular is false and
  //     tasks[0].is_edge is true, then returns opts_.edge_minibatch_size.
  inline int32 GetMinibatchSize(const ComputationGroupInfo &info) const;


  // This function compiles, and returns, a computation for tasks of
  // the structure present in info.tasks[0], and the specified minibatch
  // size.
  std::shared_ptr<const NnetComputation> GetComputation(
      const ComputationGroupInfo &info,
      int32 minibatch_size);


  // Returns the actual minibatch size we'll use for this computation.  In most
  // cases it will be opts_.minibatch_size (or opts_.edge_minibatch_size if
  // appropriate; but if the number of available tasks is much less than the
  // appropriate minibatch size, it may be less.  The minibatch size may be
  // greater than info.tasks.size(); in that case, the remaining 'n' values in
  // the minibatch are not used.  (It may also be less than info.tasks.size(),
  // in which case we only do some of them).
  int32 GetActualMinibatchSize(const ComputationGroupInfo &info) const;


  // This function gets the highest-priority 'num_tasks' tasks from 'info',
  // removes them from the array info->tasks, and puts them into the array
  // 'tasks' (which is assumed to be initially empty).
  // This function also updates the num_full_minibatches_ variable if
  // necessary, and takes care of notifying any related condition variables.
  void GetHighestPriorityTasks(
      int32 num_tasks,
      ComputationGroupInfo *info,
      std::vector<NnetInferenceTask*> *tasks);

  /**
      This function finds and returns the computation corresponding to the
      highest-priority group of tasks.

       @param [in] allow_partial_minibatch  If this is true, then this
             function may return a computation corresponding to a partial
             minibatch-- i.e. the minibatch size in the computation may be
             less than the minibatch size in the options class, and/or
             the number of tasks may not be as many as the minibatch size
             in the computation.
       @param [out] minibatch_size  If this function returns non-NULL, then
             this will be set to the minibatch size that the returned
             computation expects.  This may be less than tasks->size(),
             in cases where the minibatch was not 'full'.
       @param [out] tasks  The tasks which we'll be doing the computation
             for in this minibatch are put here (and removed from tasks_,
             in cases where this function returns non-NULL.
       @return  This function returns a pointer to the appropriate
             'MinibatchSizeInfo' object corresponding to the computation
             that we'll be doing for this minibatch, or NULL if there is nothing
             to compute.
  */
  MinibatchSizeInfo *GetHighestPriorityComputation(
      bool allow_partial_minibatch,
      int32 *minibatch_size,
      std::vector<NnetInferenceTask*> *tasks);

  /**
     formats the inputs to the computation and transfers them to GPU.
        @param [in]  minibatch_size  The number of parallel sequences
            we're doing this computation for.  This will be
            more than tasks.size() in some cases.
        @param [in] tasks  The tasks we're doing the computation for.
            The input comes from here.
        @param [out] input  The main feature input to the computation is
            put into here.
        @param [out] ivector  If we're using i-vectors, the i-vectors are
            put here.
  */
  void FormatInputs(int32 minibatch_size,
                    const std::vector<NnetInferenceTask*> &tasks,
                    CuMatrix<BaseFloat> *input,
                    CuMatrix<BaseFloat> *ivector);


  // Copies 'output', piece by piece, to the 'output_cpu' or 'output'
  // members of 'tasks', depending on their 'output_to_cpu' value.
  void FormatOutputs(const CuMatrix<BaseFloat> &output,
                     const std::vector<NnetInferenceTask*> &tasks);


  // Changes opts_.frames_per_chunk to be a multiple of
  // opts_.frame_subsampling_factor, if needed.
  void CheckAndFixConfigs();

  // this function creates and returns the computation request which is to be
  // compiled.
  static void GetComputationRequest(const NnetInferenceTask &task,
                                    int32 minibatch_size,
                                    ComputationRequest *request);

  // Prints some logging information about what we computed, with breakdown by
  // minibatch type.
  void PrintMinibatchStats();

  NnetBatchComputerOptions opts_;
  const Nnet &nnet_;
  CachingOptimizingCompiler compiler_;
  CuVector<BaseFloat> log_priors_;

  // Mutex that guards this object.  It is only held for fairly quick operations
  // (not while the actual computation is being done).
  std::mutex mutex_;

  // tasks_ contains all the queued tasks.
  // Each key contains a vector of NnetInferenceTask* pointers, of the same
  // structure (i.e., IsCompatible() returns true).
  MapType tasks_;

  // num_full_minibatches_ is a function of the data in tasks_ (and the
  // minibatch sizes, specified in opts_.  It is the number of full minibatches
  // of tasks that are pending, meaning: for each group of tasks, the number of
  // pending tasks divided by the minibatch-size for that group in integer
  // arithmetic.  This is kept updated for thread synchronization reasons, because
  // it is the shared variable
  int32 num_full_minibatches_;

  // a map from 'n' to a condition variable corresponding to the condition:
  // num_full_minibatches_ <= n.  Any time the number of full minibatches drops
  // below n, the corresponding condition variable is notified (if it exists).
  std::unordered_map<int32, std::condition_variable*> no_more_than_n_minibatches_full_;

  // some static information about the neural net, computed at the start.
  int32 nnet_left_context_;
  int32 nnet_right_context_;
  int32 input_dim_;
  int32 ivector_dim_;
  int32 output_dim_;
};


/**
   This class implements a simplified interface to class NnetBatchComputer,
   which is suitable for programs like 'nnet3-compute' where you want to support
   fast GPU-based inference on a sequence of utterances, and get them back
   from the object in the same order.
 */
class NnetBatchInference {
 public:

  NnetBatchInference(
      const NnetBatchComputerOptions &opts,
      const Nnet &nnet,
      const VectorBase<BaseFloat> &priors);

  /**
    The user should call this one by one for the utterances that this class
    needs to compute (interspersed with calls to GetOutput()).  This call
    will block when enough ready-to-be-computed data is present.

      @param [in] utterance_id  The string representing the utterance-id;
             it will be provided back to the user when GetOutput() is
             called.
      @param [in] input  The input features (e.g. MFCCs)
      @param [in] ivector  If non-NULL, this is expected to be the
             i-vector for this utterance (and 'online_ivectors' should
             be NULL).
      @param [in] online_ivector_period  Only relevant if
             'online_ivector' is non-NULL, this says how many
             frames of 'input' is covered by each row of
             'online_ivectors'.
  */
  void AcceptInput(const std::string &utterance_id,
                   const Matrix<BaseFloat> &input,
                   const Vector<BaseFloat> *ivector,
                   const Matrix<BaseFloat> *online_ivectors,
                   int32 online_ivector_period);

  /**
     The user should call this after the last input has been provided
     via AcceptInput().  This will force the last utterances to be
     flushed out (to be retrieved by GetOutput()), rather than waiting
     until the relevant minibatches are full.
  */
  void Finished();

  /**
      The user should call this to obtain output.  It's guaranteed to
      be in the same order as the input was provided, but it may be
      delayed.  'output' will be the output of the neural net, spliced
      together over the chunks (and with acoustic scaling applied if
      it was specified in the options; the subtraction of priors will
      depend whether you supplied a non-empty vector of priors to the
      constructor.

      This call does not block (i.e. does not wait on any semaphores) unless you
      have previously called Finished().  It returns true if it actually got any
      output; if none was ready it will return false.
  */
  bool GetOutput(std::string *utterance_id,
                 Matrix<BaseFloat> *output);

  ~NnetBatchInference();
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetBatchInference);

  // This is the computation thread, which is run in the background.  It will
  // exit once the user calls Finished() and all computation is completed.
  void Compute();
  // static wrapper for Compute().
  static void ComputeFunc(NnetBatchInference *object) { object->Compute(); }


  // This object implements the internals of what this class does.  It is
  // accessed both by the main thread (from where AcceptInput(), Finished() and
  // GetOutput() are called), and from the background thread in which Compute()
  // is called.
  NnetBatchComputer computer_;

  // This is set to true when the user calls Finished(); the computation thread
  // sees it and knows to flush
  bool is_finished_;

  // This semaphore is signaled by the main thread (the thread in which
  // AcceptInput() is called) every time a new utterance is added, and waited on
  // in the background thread in which Compute() is called.
  Semaphore tasks_ready_semaphore_;

  struct UtteranceInfo {
    std::string utterance_id;
    // The tasks into which we split this utterance.
    std::vector<NnetInferenceTask> tasks;
    // 'num_tasks_finished' is the number of tasks which are known to be
    // finished, meaning we successfully waited for those tasks' 'semaphore'
    // member.  When this reaches tasks.size(), we are ready to consolidate
    // the output into a single matrix and return it to the user.
    size_t num_tasks_finished;
  };

  // This list is only accessed directly by the main thread, by AcceptInput()
  // and GetOutput().  It is a list of utterances, with more recently added ones
  // at the back.  When utterances are given to the user by GetOutput(),
  std::list<UtteranceInfo*> utts_;

  int32 utterance_counter_;  // counter that increases on every utterance.

  // The thread running the Compute() process.
  std::thread compute_thread_;
};


/**
   Decoder object that uses multiple CPU threads for the graph search, plus a
   GPU for the neural net inference (that's done by a separate
   NnetBatchComputer object).  The interface of this object should
   accessed from only one thread, though-- presumably the main thread of the
   program.
 */
class NnetBatchDecoder {
 public:
  /**
     Constructor.
        @param [in] fst    FST that we are decoding with, will be shared between
                           all decoder threads.
        @param [in] decoder_config  Configuration object for the decoders.
        @param [in] trans_model   The transition model-- needed to construct the decoders,
                           and for determinization.
        @param [in] word_syms  A pointer to a symbol table of words, used for printing
                          the decoded words to stderr.  If NULL, the word-level output will not
                          be logged.
        @param [in] allow_partial   If true, in cases where no final-state was reached
                           on the final frame of the decoding, we still output a lattice;
                           it just may contain partial words (words that are cut off in
                           the middle).  If false, we just won't output anything for
                           those lattices.
        @param [in] num_threads  The number of decoder threads to use.  It will use
                          two more threads on top of this: the main thread, for I/O,
                          and a thread for possibly-GPU-based inference.
        @param [in] computer The NnetBatchComputer object, through which the
                           neural net will be evaluated.
   */
  NnetBatchDecoder(const fst::Fst<fst::StdArc> &fst,
                   const LatticeFasterDecoderConfig &decoder_config,
                   const TransitionModel &trans_model,
                   const fst::SymbolTable *word_syms,
                   bool allow_partial,
                   int32 num_threads,
                   NnetBatchComputer *computer);

  /**
    The user should call this one by one for the utterances that
    it needs to compute (interspersed with calls to GetOutput()).  This
    call will block when no threads are ready to start processing this
    utterance.

      @param [in] utterance_id  The string representing the utterance-id;
             it will be provided back to the user when GetOutput() is
             called.
      @param [in] input  The input features (e.g. MFCCs)
      @param [in] ivector  If non-NULL, this is expected to be the
             i-vector for this utterance (and 'online_ivectors' should
             be NULL).
      @param [in] online_ivector_period  Only relevant if
             'online_ivector' is non-NULL, this says how many
             frames of 'input' is covered by each row of
             'online_ivectors'.
  */
  void AcceptInput(const std::string &utterance_id,
                   const Matrix<BaseFloat> &input,
                   const Vector<BaseFloat> *ivector,
                   const Matrix<BaseFloat> *online_ivectors,
                   int32 online_ivector_period);

  /*
    The user should call this function each time there was a problem with an utterance
    prior to being able to call AcceptInput()-- e.g. missing i-vectors.  This will
    update the num-failed-utterances stats which are stored in this class.
   */
  void UtteranceFailed();

  /*
     The user should call this when all input has been provided, e.g.
     when AcceptInput will not be called any more.  It will block until
     all threads have terminated; after that, you can call GetOutput()
     until it returns false, which will guarantee that nothing remains
     to compute.
     It returns the number of utterances that have been successfully decoded.
   */
  int32 Finished();

  /**
      The user should call this to obtain output (This version should
      only be called if config.determinize_lattice == true (w.r.t. the
      config provided to the constructor).  The output is guaranteed to
      be in the same order as the input was provided, but it may be
      delayed, *and* some outputs may be missing, for example because
      of search failures (allow_partial will affect this).

      The acoustic scores in the output lattice will already be divided by
      the acoustic scale we decoded with.

      This call does not block (i.e. does not wait on any semaphores).  It
      returns true if it actually got any output; if none was ready it will
      return false.
         @param [out] utterance_id  If an output was ready, its utterance-id is written to here.
         @param [out] clat  If an output was ready, it compact lattice will be
                            written to here.
         @param [out] sentence  If an output was ready and a nonempty symbol table
                            was provided to the constructor of this class, contains
                            the word-sequence decoded as a string.  Otherwise will
                            be empty.
         @return  Returns true if a decoded output was ready.  (These appear asynchronously
                  as the decoding is done in background threads).
  */
  bool GetOutput(std::string *utterance_id,
                 CompactLattice *clat,
                 std::string *sentence);

  // This version of GetOutput is for where config.determinize_lattice == false
  // (w.r.t. the config provided to the constructor).  It is the same as the
  // other version except it outputs to a normal Lattice, not a CompactLattice.
  bool GetOutput(std::string *utterance_id,
                 Lattice *lat,
                 std::string *sentence);

  ~NnetBatchDecoder();

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetBatchDecoder);

  struct UtteranceInput {
    std::string utterance_id;
    const Matrix<BaseFloat> *input;
    const Vector<BaseFloat> *ivector;
    const Matrix<BaseFloat> *online_ivectors;
    int32 online_ivector_period;
  };

  // This object is created when a thread finished an utterance.  For utterances
  // where decoding failed somehow, the relevant lattice (compact_lat, if
  // opts_.determinize == true, or lat otherwise) will be empty (have no
  // states).
  struct UtteranceOutput {
    std::string utterance_id;
    bool finished;
    CompactLattice compact_lat;
    Lattice lat;
    std::string sentence;  // 'sentence' is only nonempty if a non-NULL symbol
                           // table was provided to the constructor of class
                           // NnetBatchDecoder; it's the sentence as a string (a
                           // sequence of words separated by space).  It's used
                           // for printing the sentence to stderr, which we do
                           // in the main thread to keep the order consistent.
  };

  // This is the decoding thread, several copies of which are run in the
  // background.  It will exit once the user calls Finished() and all
  // computation is completed.
  void Decode();
  // static wrapper for Compute().
  static void DecodeFunc(NnetBatchDecoder *object) { object->Decode(); }

  // This is the computation thread; it handles the neural net inference.
  void Compute();
  // static wrapper for Compute().
  static void ComputeFunc(NnetBatchDecoder *object) { object->Compute(); }


  // Sets the priorities of the tasks in a newly provided utterance.
  void SetPriorities(std::vector<NnetInferenceTask> *tasks);

  // In the single-thread case, this sets priority_offset_ to 'priority'.
  // In the multi-threaded case it causes priority_offset_ to approach
  // 'priority' at a rate that depends on the nunber of threads.
  void UpdatePriorityOffset(double priority);

  // This function does the determinization (if needed) and finds the best path through
  // the lattice to update the stats.  It is expected that when it is called, 'output' must
  // have its 'lat' member set up.
  void ProcessOutputUtterance(UtteranceOutput *output);

  const fst::Fst<fst::StdArc> &fst_;
  const LatticeFasterDecoderConfig &decoder_opts_;
  const TransitionModel &trans_model_;
  const fst::SymbolTable *word_syms_;  // May be NULL.  Owned here.
  bool allow_partial_;
  NnetBatchComputer *computer_;
  std::vector<std::thread*> decode_threads_;
  std::thread compute_thread_;  // Thread that calls computer_->Compute().


  // 'input_utterance', together with utterance_ready_semaphore_ and
  // utterance_consumed_semaphore_, use used to 'hand off' information about a
  // newly provided utterance from AcceptInput() to a decoder thread that is
  // ready to process a new utterance.
  UtteranceInput input_utterance_;
  Semaphore input_ready_semaphore_;  // Is signaled by the main thread when
                                     // AcceptInput() is called and a new
                                     // utterance is being provided (or when the
                                     // input is finished), and waited on in
                                     // decoder thread.
  Semaphore input_consumed_semaphore_;  // Is signaled in decoder thread when it
                                        // has finished consuming the input, so
                                        // the main thread can know when it
                                        // should continue (to avoid letting
                                        // 'input' go out of scope while it's
                                        // still needed).

  Semaphore tasks_ready_semaphore_; // Is signaled when new tasks are added to
                                    // the computer_ object (or when we're finished).

  bool is_finished_;  // True if the input is finished.  If this is true, a
                      // signal to input_ready_semaphore_ indicates to the
                      // decoder thread that it should terminate.

  bool tasks_finished_;  // True if we know that no more tasks will be given
                         // to the computer_ object.


  // pending_utts_ is a list of utterances that have been provided via
  // AcceptInput(), but their decoding has not yet finished.  AcceptInput() will
  // push_back to it, and GetOutput() will pop_front().  When a decoding thread
  // has finished an utterance it will set its 'finished' member to true.  There
  // is no need to synchronize or use mutexes here.
  std::list<UtteranceOutput*> pending_utts_;

  // priority_offset_ is something used in determining the priorities of nnet
  // computation tasks.  It starts off at zero and becomes more negative with
  // time, with the aim being that the priority of the first task (i.e. the
  // leftmost chunk) of a new utterance should be at about the same priority as
  // whatever chunks we are just now getting around to decoding.
  double priority_offset_;

  // Some statistics accumulated by this class, for logging and timing purposes.
  double tot_like_;  // Total likelihood (of best path) over all lattices that
                     // we output.
  int64 frame_count_;  // Frame count over all latices that we output.
  int32 num_success_;  // Number of successfully decoded files.
  int32 num_fail_;  // Number of files where decoding failed.
  int32 num_partial_;  // Number of files that were successfully decoded but
                       // reached no final-state (can only be nonzero if
                       // allow_partial_ is true).
  std::mutex stats_mutex_;  // Mutex that guards the statistics from tot_like_
                            // through num_partial_.
  Timer timer_;  // Timer used to print real-time info.
};


}  // namespace nnet3
}  // namespace kaldi

#endif  // KALDI_NNET3_NNET_BATCH_COMPUTE_H_
