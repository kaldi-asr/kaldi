// nnet3/nnet-general-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang    
//                2014  Vijayaditya Peddinti
//                2014  Guoguo Chen

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

#ifndef KALDI_NNET3_NNET_COMPONENT_H_
#define KALDI_NNET3_NNET_COMPONENT_H_

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

// This "nnet3" version of the p-norm component only supports the 2-norm.
class PnormComponent: public Component {
 public:
  void Init(int32 input_dim, int32 output_dim);
  explicit PnormComponent(int32 input_dim, int32 output_dim) {
    Init(input_dim, output_dim);
  }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kBackpropNeedsInput|kBackpropNeedsOutput;
  }
  PnormComponent(): input_dim_(0), output_dim_(0) { }
  virtual std::string Type() const { return "PnormComponent"; }
  virtual void InitFromString(std::string args); 
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,                        
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual Component* Copy() const { return new PnormComponent(input_dim_,
                                                              output_dim_); }
  
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

  virtual std::string Info() const;
 protected:
  int32 input_dim_;
  int32 output_dim_;
};



} // namespace nnet3
} // namespace kaldi


#endif
