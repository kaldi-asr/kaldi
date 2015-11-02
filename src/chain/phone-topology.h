// chain/phone-topology.h

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


#ifndef KALDI_CHAIN_PHONE_TOPOLOGY_H_
#define KALDI_CHAIN_PHONE_TOPOLOGY_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"

namespace kaldi {
namespace chain {


/**
  The 'PhoneTopology' object stores the topology for each of the phones that the
  system handles.  This is the equivalent of a HMM topology, except that the
  emission probabilities are on the arcs not the states (so it's much more
  FST-like), and there are no transition probabilities (these are just folded
  into the emission probabilities).  Note that it's the fact that the 'chain'
  system is trained discriminatively from the start is what enables us to treat
  the transition probabilities this way.

  A topology is an epsilon-free finite state acceptor.  The
  'normal' topology that you get if you don't do anything special, is as
  follows:

0   1   1      # transition from state 0 to state 1 with label 1.
1   1   2      # transition from state 1 to state 1 (self-loop) with label 2.
1   0          # this says that state 1 is final.

   The FSTs have the following properties:
      - they are epsilon free
      - the start state is numbered zero.
      - the start state is not final.
      - all states are used.
      - the symbols on the labels of the FST start from 1 and are contiguous (no
        unused symbols between the smallest and largest symbol).


  Phones are given indexes from 1 to NumPhones() (no gaps are allowed here).

  A topology for a phone is an FST
 */

class PhoneTopology {
 public:
  int32 NumPhones() { returns static_cast<int32>(fsts_.size()) - 1; }

  // Returns the topology for a given phone.
  const fst::VectorFst<StdArc> &TopologyForPhone(int32 phone);

  // This constructor gives the phones the default topology.  If you want to
  // give it a different topology, then you can create the text-form of this
  // object using a script.
  PhoneTopology(int32 num_phones);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary) const;

  // returns true if all the phones' FSTs have the following properties:
  //  - the symbols on arcs out of the start-state are disjoint from the
  //    symbols on arcs out of other states.
  //  - there are no arcs ending in the start state.
  bool IsAlignable();
 private:
  void Check();

  // index zero is not used.
  std::vector<fst::VectorFst<StdArc> > fsts_;
};


}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_PHONE_TOPOLOGY_H_
