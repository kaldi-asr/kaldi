// nnet3/nnet-compute.h

// Copyright   2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_COMPUTE_H_
#define KALDI_NNET3_NNET_COMPUTE_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-analyze.h"
#include "nnet3/nnet-example.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>


namespace kaldi {
namespace nnet3 {


struct NnetComputeOptions {
  bool debug;
  NnetComputeOptions(): debug(false) { }
  void Register(OptionsItf *opts) {
    opts->Register("debug", &debug, "If true, turn on "
                   "debug for the neural net computation (very verbose!) "
                   "Will be turned on regardless if --verbose >= 5");
  }

};


/**
  class NnetComputer is responsible for executing the computation described in the
  "computation" object.

  You call in sequence, the constructor, then AcceptInput() [or AcceptInputs()],
  then Run(), then GetOutput() [and if applicable, AcceptOutputDeriv], then if
  there is a backward computation, Run() [then, if applicable, GetInputDeriv()].
 */
class NnetComputer {
 public:
  /// Constructor.  nnet_to_update will be NULL if you are not doing
  /// model update or model-derivative computation.
  /// You must call computation.ComputeCudaIndexes()  before calling
  /// this function.
  ///
  /// Caution: there is another constructor that takes a pointer for
  /// 'nnet', be careful not to mix these up.
  NnetComputer(const NnetComputeOptions &options,
               const NnetComputation &computation,
               const Nnet &nnet,
               Nnet *nnet_to_update);

  /// This version of the constructor accepts a pointer to 'nnet' instead
  /// of a const reference.  The difference is that this version will,
  /// for storing statistics (the StoreStats() function of class Component),
  /// use 'nnet' instead of 'nnet_to_update' (if specified).
  NnetComputer(const NnetComputeOptions &options,
               const NnetComputation &computation,
               Nnet *nnet,
               Nnet *nnet_to_update);


  /// Copy constructor.  May not be used if memos are stored with this object
  /// (which is only a possibility if backprop will take place, and in these
  /// situations you won't normally be wanting to use the copy constructor
  /// anyway; the copy constructor is more useful for things like RNNLM lattice
  /// rescoring).
  NnetComputer(const NnetComputer &other);

  /// e.g. AcceptInput ("input", &input_mat), or for derivatives w.r.t. the
  /// output, AcceptInput("output", output_deriv_mat).  Will crash if there is
  /// no input or output node with the given name.  This function is destructive
  /// of "input" as it takes it using the Swap function of CuMatrix.  Must have
  /// the same number of rows as the corresponding input described in the
  /// ComputationRequest e.g. the indexes.size() in the corresponding
  /// IoSpecification.
  void AcceptInput(const std::string &node_name,
                   CuMatrix<BaseFloat> *input);

  /// This convenience function calls AcceptInput() in turn on all the inputs in
  /// the training example.  It needs "nnet" only in order to distinguish inputs
  /// from outputs.
  void AcceptInputs(const Nnet &nnet,
                    const std::vector<NnetIo> &io);


  /// This does either the forward or backward computation, depending
  /// when it is called (in a typical computation, the first time you call
  /// this it will do the forward computation; then you'll take the outputs
  /// and provide derivatives; and the second time you call it, it will do
  /// the backward computation.  There used to be two separate functions
  /// Forward() and Backward().
  void Run();

  // e.g. GetOutput("output").  This function can also be used to get
  // derivatives w.r.t. inputs.  It's non-const because it may only
  // be called once and it keeps track of that.
  const CuMatrixBase<BaseFloat> &GetOutput(const std::string &node_name);

  // Version of GetOutput that calls Swap(), destroying the output stored inside
  // this object.  You should probably not use this if you plan to call
  // Backward() on the same NnetComputer object, or it's a recurret
  // computation-- it may lead to a crash.
  void GetOutputDestructive(const std::string &output_name,
                            CuMatrix<BaseFloat> *output);


  ~NnetComputer();
 private:
  void Init(); // called from constructors.

  const NnetComputeOptions &options_;
  const NnetComputation &computation_;
  const Nnet &nnet_;

  int32 program_counter_;  // command index to execute next.
  // To deal with inputs and outputs that are not provided/taken by the user in
  // the same order as listed in the computation, pending_commands_ contains a
  // list of program commands that were skipped over but are in the queue to be
  // executed.
  std::vector<int32> pending_commands_;

  // A pointer to the copy of the nnet which we'll be using for stats
  // accumulation (the StoreStats() function).  May be NULL or the same
  // as nnet_ or nnet_to_update_.
  Nnet *nnet_to_store_stats_;
  // A pointer to the copy of the nnet which we'll be updating the parameters
  // of (nnet_to_update in the backprop function).  May be NULL and usually
  // will not be the same as nnet_.
  Nnet *nnet_to_update_;
  bool debug_;
  // command_attributes_ is only used if debug_=true.
  std::vector<CommandAttributes> command_attributes_;
  // submatrix_strings_ is only used if debug_=true.
  std::vector<std::string> submatrix_strings_;
  // command_strings_ is only used if debug_=true, or in case of error.
  std::vector<std::string> command_strings_;

  // The matrices used in the computation.
  std::vector<CuMatrix<BaseFloat> > matrices_;

  // Memos returned by Propagate() that must be passed to the corresponding
  // Backprop() routines, indexed by memo-index (zeroth element always
  // NULL).
  std::vector<void*> memos_;

  // This is only used when commands kCompressMatrix and kDecompressMatrix are
  // invoked.  It will be (the first time we compress a matrix) resized to be
  // the same size as 'matrices_' (i.e., indexed by matrix index).  When we
  // compress a matrix m we set compressed_matrices_[m] to a non-NULL value and
  // resize matrices_[m] to empty; and when we uncompress it, the reverse
  // happens.
  std::vector<CuCompressedMatrixBase*> compressed_matrices_;


  // executes the command in computation_.commands[program_counter_].
  void ExecuteCommand();

  // Returns the matrix index where the input (if is_output==false) or output
  // matrix index for "node_name" is stored.  This looks at the next command (at
  // program_counter_) and in pending_commands_, and sees whether we were
  // expecting any input or output for this node, and if there is a match,
  // returns it and "consumes" the command by either advancing program_counter_
  // or consuming something from pending_commands_.
  // If there is not a match (i.e. we were not expecting this type of I/O
  // at this point in the computation), it prints an error and dies.
  int32 GetIoMatrixIndex(const std::string &node_name, bool is_output);


  // This function, called from Run(), checks that there is no pending I/O
  // that we were waiting for, that would block the running of the
  // computation; it crashes if there was pending input, and ignores and
  // skips over any pending output.
  void CheckNoPendingIo();

  CuSubMatrix<BaseFloat> GetSubMatrix(int32 submatrix_index);

  void GetPointers(int32 indexes_multi_index,
                   int32 num_cols,
                   CuArray<BaseFloat*> *pointers);
  void GetPointers(int32 indexes_multi_index,
                   int32 num_cols,
                   CuArray<const BaseFloat*> *pointers);

  struct CommandDebugInfo {
    // Uncentered standard deviations of elements of all matrices that this
    // command writes.  Dimension is the same as
    // command_attributes_[c].matrices_written
    std::vector<BaseFloat> matrices_written_stddevs;
    // Uncentered standard deviations of elements of all submatrices that this
    // command writes (if they are not whole matrices).  Dimension is the same
    // as command_attributes_[c].submatrices_written
    std::vector<BaseFloat> submatrices_written_stddevs;

    // Uncentered standard deviation of parameters of the component (if any)
    // that is updated by this command.
    BaseFloat components_parameter_stddev;
  };
  // Used in debugging code
  static BaseFloat MatrixStddev(const CuMatrixBase<BaseFloat> &m);
  // Used in debugging code
  static BaseFloat ParameterStddev(const Component &c);

  // only non-const because of the way GetSubMatrix works.
  void DebugBeforeExecute(int32 command,
                          CommandDebugInfo *info);
  // only non-const because of the way GetSubMatrix works.
  void DebugAfterExecute(int32 command,
                         const CommandDebugInfo &info,
                         double command_execution_time);

  // simple helper function used in executing Propagate().
  // saves 'memo' at memo-index 'memo_index'; if memo
  // is non-NULL and memo_index is 0, it is an error.
  inline void SaveMemo(int32 memo_index, const Component &c, void *memo);

  // simple helper function used in executing Backprop().
  // Retrieves memo from 'memo_index' (or returns NULL if
  // memo_index = 0), and sets that value to NULL as
  // memos are not reusable.
  inline void *GetMemo(int32 memo_index);

  NnetComputer &operator = (const NnetComputer &other);  // Disallow.
};



} // namespace nnet3
} // namespace kaldi

#endif
