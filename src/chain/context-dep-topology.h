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
  dependent phones.  It also allocates 'graph-labels' and 'output-labels'.

  A 'graph-label' is one-based, is sufficient to identify the logical CD-phone
  and the label in the topology, and can also be mapped to an 'output-label'.

  The output-label is also one-based; it is sufficient to identify the physical
  CD-phone and the label in the topology object, but won't let you identify
  the monophone (because output-labels may be shared between monophones).

  You can think of this object as roughly comparable to 'HC' in the 'HCLG'
  recipe.  It's of a manageable size as an FST, because we limit ourselves to
  left context.

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
  // one, which we call output-index.
  int32 NumOutputLabels();

  // returns the number of graph-labelss.
  int32 NumGraphLabels();

  // convenience function to return the number of phones.
  int32 NumPhones() { return phone_topology_.NumPhones(); }

  // maps a graph-label to an output-label.
  int32 GraphLabelToOutputLabel(int32 graph_label);

  // maps a graph label to a phone.
  int32 GraphLabelToPhone(int32 graph_label);

  // maps a graph label to a label in the phone's topology object (you probably
  // won't need this very often except for phone alignment).
  int32 GraphLabelToTopologyLabel(int32 graph_label);




 private:
  PhoneTopology phone_topology_;
  PhoneContext phone_context_;




  /*  First, members that relate to the base class.   */

  // repeat the typedefs (they're not inherited automatically; we could inherit
  //  but they are boilerplate so we just repeat them).
  typedef typename fst::StdArc Arc;
  typedef typename Arc::StateId StateId;  // should be int32.
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;  // should be int32.

  // The following are part of the interface from DeterministicOnDemandFst.
  virtual StateId Start() { return 0; }

  // all states are final.
  virtual Weight Final(StateId s) { return Weight::One(); }

  // Assuming 0 <= s < NumStates() and 1 <= phone <= NumPhones(),
  // this function will return true and output to Arc as follows:
  // ilabel = phone, olabel = cd-phone, weight = One(),
  // nextstate = [the next state after seeing this phone.]
  virtual bool GetArc(StateId s, Label phone, Arc *oarc) = 0;

  virtual ~PhoneContext();

  /*  Next members not relating to the base class.   */

  PhoneContext();

  // Initialization from a tree (which must be left-context only, i.e.
  // CentralPosition() == ContextWidth() - 1).  The initialization method relies
  // on enumerating all possible contexts, so it will be slow if you have a
  // ridiculously large context.
  PhoneContext(int32 num_phones, const ContextDependency &ctx_dep);

  // Phones are numbered from 1 to NumPhones().
  int32 NumPhones() { return num_phones_; }

  // Context-dependent phones are numbered from 1 to NumCdPhones().
  int32 NumCdPhones() { return num_cd_phones_; }

  // This function tells you how many phones of left-context the underlying
  // decision tree was built with: 0 for monophone, 1 for left-biphone, etc.  It
  // amounts to an assertion that if you take a given phone sequence of length
  // LeftContext(), and starting from any FST state, use that phone-sequence as
  // ilabels, you'll always end up in the same state.
  int32 LeftContext() { return left_context_; }

  // There is a concept of states in this model, whereby when it outputs
  // a phone it advances the state.  So it's an FST-like representation of
  // the decision tree.  States are numbered from 0 to NumStates() - 1.
  int32 NumStates() { return transitions_.size(); }

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is) const;

  // Outputs to 'output' an FST that's a copy of this object.  ilabels are
  // phones, olabels are cd-phones.  Note: can be implemented by taking an FST
  // 'f' with one state that's initial and final, with self-loops for each
  // phone, and then calling ComposeDeterministicOnDemand(f, *this, output).
  void GetAsFst(fst::VectorFst<fst::StdArc>* output);
 private:
  void Check();

  int32 num_phones_;
  int32 num_cd_phones_;
  int32 left_context_;

  // 'transitions_' is indexed by state, then by phone - 1 (each vector of pairs
  // is of length num_phones), and each pair is (cd-phone-index, next-state).
  // For instance (bear in mind that 0 is the initial-state that you get at the
  // begining of a phone_sequence), transitions_[0][p].first is the
  // cd-phone-index you get from seeing phone p with the left-context being the
  // beginning of a sequence (i.e. a left-context of all zeros, as far as the
  // tree is concerned); and transitions_[0][p].second is the context state you
  // go to after seeing that phone.
  std::vector<std::vector<std::pair<int32, int32> > > transitions_;

};


}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CONTEXT_DEP_TOPOLOGY_H_
