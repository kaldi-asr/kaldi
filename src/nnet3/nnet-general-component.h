// nnet3/nnet-general-component.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_GENERAL_COMPONENT_H_
#define KALDI_NNET3_NNET_GENERAL_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  This file contains declarations of components that are not "simple",
///   meaning they care about the indexes they are operating on, don't return
///   the kSimpleComponent flag in their Properties(), and may return a different
///   number of outputs than inputs.



/**
   This Component takes a larger input-dim than output-dim, where the input-dim
   must be a multiple of the output-dim, and distributes different blocks of the
   input dimension to different 'x' values.  In the normal case where the input
   is only valid at x=0, the first block of output goes to x=0, the second block
   at x=1, and so on.  It also supports a more general usage, so in general a
   value 'x' at the output will map to block 'x % n_blocks' of the dimension
   blocks of the input, and to an x value 'x / n_blocks' of the input.  For negative
   x values the % and / operations are always rounded down, not towards zero.
   */
class DistributeComponent: public Component {
 public:
  DistributeComponent(int32 input_dim, int32 output_dim) {
    Init(input_dim, output_dim);
  }
  DistributeComponent(): input_dim_(0), output_dim_(0) { }
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }

  // use the default Info() function.
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "DistributeComponent"; }
  virtual int32 Properties() const { return kLinearInInput; }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new DistributeComponent(input_dim_, output_dim_);
  }


  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  // Some functions that are specific to this class.
  void Init(int32 input_dim, int32 output_dim);
 private:
  // computes the input index corresponding to a particular output index.
  // if block != NULL, also computes which block of the input this corresponds to.
  inline void ComputeInputIndexAndBlock(const Index &output_index,
                                        Index *input_index,
                                        int32 *block) const;
  inline void ComputeInputPointers(
      const ComponentPrecomputedIndexes *indexes,
      const CuMatrixBase<BaseFloat> &in,
      int32 num_output_rows,
      std::vector<const BaseFloat*> *input_pointers) const;
  // non-const version of the above.
  inline void ComputeInputPointers(
      const ComponentPrecomputedIndexes *indexes,
      int32 num_output_rows,
      CuMatrixBase<BaseFloat> *in,
      std::vector<BaseFloat*> *input_pointers) const;
  int32 input_dim_;
  int32 output_dim_;

};

} // namespace nnet3
} // namespace kaldi


#endif
