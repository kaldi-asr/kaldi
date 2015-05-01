// nnet3/nnet-compile.h

// Copyright 2015    Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_COMPILE_H_
#define KALDI_NNET3_NNET_COMPILE_H_

#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {

// Options class for the process of turning a ComputationSpecification into a
// NnetComputation.
struct NnetCompileConfig {

  void Register(OptionsItf *po) {
  }
};

// The first step in compilation is to turn the ComputationSpecification
// into a ConcreteComputationGraph, where for each Cindex we have a list of
// other Cindexes that it depends on.
struct ConcreteComputationGraph {

  // Maps each Cindex to an integer (call this cindex_id) that uniquely
  // identifies it (obviously these cindex_id values are specific to the
  // computation graph).
  std::unordered_map<Cindex, int32, CindexHasher> cindex_to_index;

  // This is the reverse mapping of cindex_to_index: it maps from cindex_id to
  // Cindex.
  std::vector<Cindex> cindexes;

  // dependencies[cindex_id] gives you the list of other cindex_ids that this
  // particular Cindex directly depends on.  (In general we work this out by
  // calling GetInputIndexes on the relevant Component (with the relevant
  // Index), and then calling MapToInput on its InputDescriptor.
  std::vector<std::vector<int32> > dependencies;
  

};




} // namespace nnet2
} // namespace kaldi


#endif

/*
  Compilation model...

  list of objects to learn and have stats for (e.g. SoftmaxComponent, AffineComponent, FixedAffineComponent)

  Then a compilation method to go from either a training or test setup, to an
    NnetComputation.
  An NnetComputation is:

  A number of declarations of matrices, with sizes, each being the input or output of a
    particular layer (may be both the input of one layer and the output of
    another).  Also declarations of sub-matrices.
  A number of commands, such as:
    Initialize matrix; zero matrix; deinitialize matrix; do Propagate or Backprop;
    do forward or backward of AddMatrix or CopyMatrix.
  
  Get rid of all this complexity and just accept redundant copy??   I.e. do it all in C++?
     -- Linear list of components, each with list of inputs, if needed?  Sharing of parameters
        between components?  [ComponentInstance vs. Component]
  [?

  Given required output indexes, and maybe some background info,
  
    

  need e.g. out[1]

  out_obj=SoftmaxObject

  out(t,n,x) = SoftmaxComponent(dim=1000), input=layer10(t,n,x)
  layer10(t,n,x) = AffineComponent(input-dim=1000, output-dim=1000), input=layer9(t,n,x)
  layer9(t,n,x) = RectifiedLinearComponent(dim=1000), input=Append(layer8(t-4,n,x), layer8(t+4,n,x))
  layer8(.....) =
    ... below, input is a matrix! ...
  layer5(0,n,x) = AggregateComponent(dim=1000, sub-dim=5000), input=layer4(*, n, x)  <--- The * could be, say, [100,200,300]
              <--- layer4 would have to have indexes declared.
    ... this is many-to-one, so input would have to have a set of

  layer1[t-5],layer1[t+5]

---
  At some point we generate a program.
  Training program example:

  Declare matrices and sub-matrices (m), and "real components" (c),
    and backprop quantities n.
  
  Assume input provided as m[0], output will be given as various
    parts of the matrices, including the output itself].  (e.g. DeclareOutput(...), meaning
    we can't delete it.)
  
  
  
 */
