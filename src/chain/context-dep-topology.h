// chain/context-dep-topology.h

// Copyright      2015  Johns Hopkins University (Author: Daniel Povey)


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

#ifndef KALDI_CHAIN_CONTEXT_DEP_TOPOLOGY_H_
#define KALDI_CHAIN_CONTEXT_DEP_TOPOLOGY_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "chain/phone-topology.h"
#include "chain/phone-context.h"

namespace kaldi {
namespace chain {


/**
  The 'ContextDepTopology' object is responsible for combining the
  'PhoneTopology' model, which describes the quasi-HMM topology for each phone,
  and the 'PhoneContext' model, which describes how we create left-context
  dependent phones.  It also allocates 'graph-labels' and 'output-labels'.  It
  is analogous to 'HC' in the 'HCLG' recipe.  It's of a manageable size as an
  FST, because we limit ourselves to left context.

  A 'graph-label' is one-based, is sufficient to identify the logical CD-phone
  and the label in the topology, and can also be mapped to an 'output-label'.

  The output-label is also one-based; it is sufficient to identify the physical
  CD-phone and the label in the topology object, but won't let you identify
  the monophone (because output-labels may be shared between monophones).

  The neural-net output is indexed by the output-label minus one (to form
  a zero-based index).
*/

class ContextDepTopology {
 public:

  ContextDepTopology();

  ContextDepTopology(const PhoneTopology &topology,
                     const PhoneContext &context);

  const PhoneTopology &GetPhoneTopology() { return phone_topology_; }

  const PhoneContext &GetPhoneContext() { return phone_context_; }

  // Returns the number of output-labels (labels corresponding to the neural-net
  // output).  The actual neural-net output matrix is indexed by the label minus
  // one, which we call an output-index.
  int32 NumOutputLabels();

  // Returns the number of graph-labels.   A graph-label is what will typically
  // appear in HCLG decoding graphs; it is mappable to an output-label, but we
  // also ensure that it is mappable to a phone.
  int32 NumGraphLabels();

  // convenience function to return the number of phones.
  int32 NumPhones() { return phone_topology_.NumPhones(); }

  // maps a graph-label to an output-label.
  int32 GraphLabelToOutputLabel(int32 graph_label);

  // maps a graph label to a phone.
  int32 GraphLabelToPhone(int32 graph_label);

  // maps a graph label to a logical cd-phone [a logical cd-phone is always
  // mappable to the monophone].
  int32 GraphLabelToLogicalCdPhone(int32 graph_label);

  // maps a graph label to a physical cd-phone, as defined by the PhoneContext
  // object.
  int32 GraphLabelToPhysicalCdPhone(int32 graph_label);

  // maps a graph label to a label in the phone's topology object (needed to
  // work out phone alignments).
  int32 GraphLabelToTopologyLabel(int32 graph_label);

  // Outputs to 'output' an FST that represents this object-- it's essentially
  // the 'HC' object in the 'HCLG' recipe.  It's an unweighted transducer where
  // the input labels are phones (or epsilon) and the output labels are
  // 'graph-labels'.  Note: we will ensure that there are no epsilons on
  // the 'output side'.
  void GetAsFst(fst::VectorFst<fst::StdArc>* output) const;

  // This variant of of GetAsFst gives you 'output-labels' as the olabels, instead
  // of graph-labels.  These are indexes-into-the-nnet-output plus one.
  void GetAsFstWithOutputLabels(fst::VectorFst<fst::StdArc>* output) const;

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

 private:
  PhoneTopology phone_topology_;
  PhoneContext phone_context_;

  struct GraphLabelInfo {
    int32 logical_cd_phone;
    int32 topology_label;
    int32 output_label;
  };
};


}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CONTEXT_DEP_TOPOLOGY_H_
