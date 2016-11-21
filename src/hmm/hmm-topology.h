// hmm/hmm-topology.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_HMM_HMM_TOPOLOGY_H_
#define KALDI_HMM_HMM_TOPOLOGY_H_

#include "base/kaldi-common.h"
#include "tree/context-dep.h"
#include "util/const-integer-set.h"


namespace kaldi {


/// \addtogroup hmm_group
/// @{

/*
 // The following would be the text form for the "normal" HMM topology.
 // Note that the first state is the start state, and the final state,
 // which must have no output transitions and must be nonemitting, has
 // an exit probability of one (no other state can have nonzero exit
 // probability; you can treat the transition probability to the final
 // state as an exit probability).
 // Note also that it's valid to omit the "<PdfClass>" entry of the <State>, which
 // will mean we won't have a pdf on that state [non-emitting state].  This is equivalent
 // to setting the <PdfClass> to -1.  We do this normally just for the final state.
 // The Topology object can have multiple <TopologyEntry> blocks.
 // This is useful if there are multiple types of topology in the system.

 <Topology>
 <TopologyEntry>
 <ForPhones> 1 2 3 4 5 6 7 8 </ForPhones>
 <State> 0 <PdfClass> 0
 <Transition> 0 0.5
 <Transition> 1 0.5
 </State>
 <State> 1 <PdfClass> 1
 <Transition> 1 0.5
 <Transition> 2 0.5
 </State>
 <State> 2 <PdfClass> 2
 <Transition> 2 0.5
 <Transition> 3 0.5
 <Final> 0.5
 </State>
 <State> 3
 </State>
 </TopologyEntry>
 </Topology>
*/

// kNoPdf is used where pdf_class or pdf would be used, to indicate,
// none is there.  Mainly useful in skippable models, but also used
// for end states.
// A caveat with nonemitting states is that their out-transitions
// are not trainable, due to technical issues with the way
// we decided to accumulate the stats.  Any transitions arising from (*)
// HMM states with "kNoPdf" as the label are second-class transitions,
// They do not have "transition-states" or "transition-ids" associated
// with them.  They are used to create the FST version of the
// HMMs, where they lead to epsilon arcs.
// (*) "arising from" is a bit of a technical term here, due to the way
// (if reorder == true), we put the transition-id associated with the
// outward arcs of the state, on the input transition to the state.

/// A constant used in the HmmTopology class as the \ref pdf_class "pdf-class"
/// kNoPdf, which is used when a HMM-state is nonemitting (has no associated
/// PDF).

static const int32 kNoPdf = -1;

/// A class for storing topology information for phones.  See  \ref hmm for context.
/// This object is sometimes accessed in a file by itself, but more often
/// as a class member of the Transition class (this is for convenience to reduce
/// the number of files programs have to access).

class HmmTopology {
 public:
  /// A structure defined inside HmmTopology to represent a HMM state.
  struct HmmState {
    /// The \ref pdf_class forward-pdf-class, typically 0, 1 or 2 (the same as the HMM-state index),
    /// but may be different to enable us to hardwire sharing of state, and may be
    /// equal to \ref kNoPdf == -1 in order to specify nonemitting states (unusual).
    int32 forward_pdf_class;

    /// The \ref pdf_class self-loop pdf-class, similar to \ref pdf_class forward-pdf-class.
    /// They will either both be \ref kNoPdf, or neither be \ref kNoPdf.
    int32 self_loop_pdf_class;

    /// A list of transitions, indexed by what we call a 'transition-index'.
    /// The first member of each pair is the index of the next HmmState, and the
    /// second is the default transition probability (before training).
    std::vector<std::pair<int32, BaseFloat> > transitions;

    explicit HmmState(int32 pdf_class) {
      this->forward_pdf_class = pdf_class;
      this->self_loop_pdf_class = pdf_class;
    }
    explicit HmmState(int32 forward_pdf_class, int32 self_loop_pdf_class) {
      KALDI_ASSERT((forward_pdf_class != kNoPdf && self_loop_pdf_class != kNoPdf) ||
                   (forward_pdf_class == kNoPdf && self_loop_pdf_class == kNoPdf));
      this->forward_pdf_class = forward_pdf_class;
      this->self_loop_pdf_class = self_loop_pdf_class;
    }

    bool operator == (const HmmState &other) const {
      return (forward_pdf_class == other.forward_pdf_class &&
              self_loop_pdf_class == other.self_loop_pdf_class &&
              transitions == other.transitions);
    }

    HmmState(): forward_pdf_class(-1), self_loop_pdf_class(-1) { }
  };

  /// TopologyEntry is a typedef that represents the topology of
  /// a single (prototype) state.
  typedef std::vector<HmmState> TopologyEntry;

  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;

  // Checks that the object is valid, and throw exception otherwise.
  void Check();

  /// Returns true if this HmmTopology is really 'hmm-like', i.e. the pdf-class on
  /// the self-loops and forward transitions of all states are identical. [note: in HMMs,
  /// the densities are associated with the states.] We have extended this to
  /// support 'non-hmm-like' topologies (where those pdf-classes are different),
  /// in order to make for more compact decoding graphs in our so-called 'chain models'
  /// (AKA lattice-free MMI), where we use 1-state topologies that have different pdf-classes
  /// for the self-loop and the forward transition. Note that we always use the 'reorder=true'
  /// option so the 'forward transition' actually comes before the self-loop.
  bool IsHmm() const;

  /// Returns the topology entry (i.e. vector of HmmState) for this phone;
  /// will throw exception if phone not covered by the topology.
  const TopologyEntry &TopologyForPhone(int32 phone) const;

  /// Returns the number of \ref pdf_class "pdf-classes" for this phone;
  /// throws exception if phone not covered by this topology.
  int32 NumPdfClasses(int32 phone) const;

  /// Returns a reference to a sorted, unique list of phones covered by
  /// the topology (these phones will be positive integers, and usually
  /// contiguous and starting from one but the toolkit doesn't assume
  /// they are contiguous).
  const std::vector<int32> &GetPhones() const { return phones_; };

  /// Outputs a vector of int32, indexed by phone, that gives the
  /// number of \ref pdf_class pdf-classes for the phones; this is
  /// used by tree-building code such as BuildTree().
  void GetPhoneToNumPdfClasses(std::vector<int32> *phone2num_pdf_classes) const;

  // Returns the minimum number of frames it takes to traverse this model for
  // this phone: e.g. 3 for the normal HMM topology.
  int32 MinLength(int32 phone) const;

  HmmTopology() {}

  bool operator == (const HmmTopology &other) const {
    return phones_ == other.phones_ && phone2idx_ == other.phone2idx_
        && entries_ == other.entries_;
  }
  // Allow default assignment operator and copy constructor.
 private:
  std::vector<int32> phones_;  // list of all phones we have topology for.  Sorted, uniq.  no epsilon (zero) phone.
  std::vector<int32> phone2idx_;  // map from phones to indexes into the entries vector (or -1 for not present).
  std::vector<TopologyEntry> entries_;
};


/// @} end "addtogroup hmm_group"


} // end namespace kaldi


#endif
