// hmm/topology.h

// Copyright 2009-2011  Microsoft Corporation
//                2019  Johns Hopkins University (author: Daniel Povey)

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

#include <fst/fstlib.h>
#include "base/kaldi-common.h"


namespace kaldi {


/// \addtogroup hmm_group
/// @{

/*
  The following would be the text form for the "normal" 3-state HMM topology/
  "bakis model", with the typical reordering that we do to improve the
  compactness of the compiled FSTs.  The format is the OpenFst acceptor format.
  The fields are, for transitions,
  <from-state> <to-state> <pdf-class> <transition-cost>
 and, for final-states,
  <state> <final-cost>

  The <transition-cost> may be interpreted as negative log probabilities.
  We normally set them so as to sum to one, in order to keep the fully
  compiled (HCLG) graph fairly stochastic (meaning: sum-to-one, like an
  HMM).

  The integers on the arcs, which we call 'pdf-classes', define which
  arcs share the same "pdf" and which ones are distinct.

  Preconditions on topology:
     - pdf-classes (3rd field on arcs) must
       form a contiguous list of numbers starting from 1, although
       different arcs with the same pdf-class are allowed.  (We avoid 0
       because it is "special" in OpenFST, it is used for epsilon).
     - The start state must be state 0 and there must be no
       transitions entering it except (possibly) a self-loop (although
       a self-loop on state 0 is not advised for decoding-graph-size
       reasons)
     - The start state must not be final.


 <Topology>
 <TopologyEntry>
 <ForPhones> 1 2 3 4 5 6 7 8 </ForPhones>
 0  1  1  0.0
 1  1  1  0.693
 1  2  2  0.693
 2  2  2  0.693
 2  3  3  0.693
 3  3  3  0.693
 3  0.693
 </TopologyEntry>
 </Topology>
*/


/// A class for storing topology information for phones.  See  \ref hmm for context.
/// This object is sometimes accessed in a file by itself, but more often
/// as a class member of the Transition class (this is for convenience to reduce
/// the number of files programs have to access).

class Topology {
 public:

  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;

  // Checks that the object is valid, and throw exception otherwise.
  void Check();

  /// Returns the topology entry for this phone;
  /// will throw exception if phone not covered by the topology.
  const fst::StdFst &TopologyForPhone(int32 phone) const;

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

  // Returns the minimum number of arcs/frames it takes to traverse this model
  // for this phone: e.g. 3 for the normal HMM topology.
  int32 MinLength(int32 phone) const;

  Topology() {}

  bool operator == (const Topology &other) const;

  // was:
  //return phones_ == other.phones_ && phone2idx_ == other.phone2idx_
  //&& entries_ == other.entries_;
  // TODO: implement this; we probably need Equal() on fsts.

  // Allow default assignment operator and copy constructor.
 private:
  std::vector<int32> phones_;  // list of all phones we have topology for.  Sorted, uniq.  no epsilon (zero) phone.
  std::vector<int32> phone2idx_;  // map from phones to indexes into the entries vector (or -1 for not present).
  std::vector<fst::StdFst> entries_;
};


/// @} end "addtogroup hmm_group"


} // end namespace kaldi


#endif
