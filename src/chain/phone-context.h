// chain/phone-context.h

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


#ifndef KALDI_CHAIN_PHONE_CONTEXT_H_
#define KALDI_CHAIN_PHONE_CONTEXT_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"

namespace kaldi {
namespace chain {


/**
  The 'PhoneContext' object is responsible for mapping phones in left-context to
  cd-phones (context-dependent phones).  In the 'chain' models, we only support
  left-context, in order to make phone-level discriminative training
  sufficiently efficient.  The 'PhoneContext' model represents all the
  information we need to know about the phonetic-context decision tree (so after
  building the decision tree, we can build the PhoneContext object and then
  discard the tree).

  There two types of cd-phones: cd-phones, and physical cd-phones.  The logical
  ones can be mapped to physical.  The logical cd-phones are the ones that we
  actually put in the graph, which will enable us to work out the phone sequence
  (assuming the topology is 'alignable', which it normally will be).  Logical
  cd-phones are mappable to the (mono) phone; the physical ones are less
  detailed, and can't necessarily be mapped to the monophones.

  Note that the PhoneTopology and PhoneContext will be incorporated as data
  members in the ContextDependentTopology model, which contains information
  about topology and context, and also controls the allocation of output-ids
  (which are indexes into the neural net output, and roughly correspond to
  context-dependent states in a conventional HMM-based system).
*/

class PhoneContext: public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
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
  // ilabel = phone, olabel = logical-cd-phone, weight = One(),
  // nextstate = [the next state after seeing this phone.]
  virtual bool GetArc(StateId s, Label phone, Arc *oarc) = 0;

  // There is a concept of states in this model, whereby when it outputs a phone
  // it advances the state.  So it's an FST-like representation of the decision
  // tree.  States are numbered from 0 to NumStates() - 1.  This function is
  // actually not in the interface, but it is the same as in ExpandedFst.
  int32 NumStates() const { return transitions_.size(); }

  virtual ~PhoneContext();

  /*  Next members not relating to the base class.   */

  PhoneContext();

  // Initialization from a tree (which must be left-context only, i.e.
  // CentralPosition() == ContextWidth() - 1).  The initialization method relies
  // on enumerating all possible contexts, so it will be slow if you have a
  // ridiculously large context.

  // Note: we hope not to use this, we will use a separate version of the
  // tree-building code that tries to reduce the number of 'context states'.
  PhoneContext(int32 num_phones, const ContextDependencyInterface &ctx_dep);

  // Phones are numbered from 1 to NumPhones().
  int32 NumPhones() const { return num_phones_; }


  // Return the number of distinct labels on the topology FST for this phone:
  // the labels must be contiguously numbered from 1, so this is the same as
  // the largest topology label.
  bool GetNumLabels(int32 phone) const;

  // Logical context-dependent phones are numbered from 1 to
  // NumLogicalCdPhones().
  int32 NumLogicalCdPhones() const { return logical_to_phone_.size() - 1; }

  // Physical context-dependent phones are numbered from 1 to
  // NumPhysicalCdPhones().
  int32 NumPhysicalCdPhones() const { return num_physical_cd_phones_; }

  // This function tells you how many phones of left-context the underlying
  // decision tree was built with: 0 for monophone, 1 for left-biphone, etc.  It
  // amounts to an assertion that if you take a given phone sequence of length
  // LeftContext(), and starting from any FST state, use that phone-sequence as
  // ilabels, you'll always end up in the same state.
  int32 LeftContext() const { return left_context_; }

  // Maps a logical CD-phone to the phone index (i.e. of the monophone with
  // no context)-- you cannot map to a full context, that is not what
  // logical CD-phones mean in this code.
  int32 LogicalToPhone(int32 logical_cd_phone) const;

  // Maps a logical CD-phone to a physical CD-phone.
  int32 LogicalToPhysical(int32 logical_cd_phone) const;

  // Given a context-dependent phone index, return the set of phones it may
  // correspond to (in most cases this would be a set of just one element).
  // We'll implement this when we need it- it will require storing derived
  // variables, to make it efficient.
  //
  // void CdPhoneToPhones(int32 cd_phone, std::vector<int32> *phones);


  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is);

  // Outputs to 'output' an FST that's a copy of this object in the normal FST
  // format (as opposed to DeterministicOnDemandFst).  This is the 'C' FST
  // (the context-dependency FST) in the HCLG recipe.
  // ilabels are phones, olabels are cd-phones.  Note: can be implemented by
  // taking an FST 'f' with one state that's initial and final, with self-loops
  // for each phone, and then calling ComposeDeterministicOnDemand(f, *this,
  // output).
  void GetAsFst(fst::VectorFst<fst::StdArc>* output) const;
 private:
  void Check();
  // Sets up the cd_phone_to_phone_ array.
  void ComputeCdPhoneToPhone();

  int32 num_phones_;
  int32 num_physical_cd_phones_;
  int32 left_context_;

  // 'transitions_' is indexed by state, then by phone - 1 (each vector of pairs
  // is of length num_phones), and each pair is (cd-phone-index, next-state).
  // For instance (bear in mind that 0 is the initial-state that you get at the
  // begining of a phone_sequence), transitions_[0][p].first is the
  // logical-cd-phone you get from seeing phone p with the left-context being the
  // beginning of a sequence (i.e. a left-context of all zeros, as far as the
  // tree is concerned); and transitions_[0][p].second is the context state you
  // go to after seeing that phone.
  std::vector<std::vector<std::pair<int32, int32> > > transitions_;

  // map logical CD-phones to phones.  Indexed by logical CD-phone (zeroth
  // element not used).
  std::vector<int32> logical_to_phone_;

  // map logical CD-phones to physical CD-phones.  Indexed by logical CD-phone (zeroth
  // element not used).
  std::vector<int32> logical_to_physical_;

};


}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_PHONE_CONTEXT_H_
