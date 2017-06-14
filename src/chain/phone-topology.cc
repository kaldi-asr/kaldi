// chain/phone-topology.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)
//                2015   Xingyu Na

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

#include "chain/phone-topology.h"

namespace kaldi {
namespace chain {


const fst::VectorFst<StdArc>& PhoneTopolgy::TopologyForPhone (int32 phone) {
  return fsts_[phone];
}

PhoneTopology::PhoneTopology (int32 num_phones) {
  fsts_.clear();
  fsts_.resize(num_phones + 1);
  for (int32 i = 1; i <= num_phones; i++) {
    fst::VectorFst<StdArc> fst;
    fst.AddState();  // state 0
    fst.SetStart(0); // set start state
    fst.AddState();  // state 1
    fst.AddArc(0, StdArc(1, 1, 0.5, 1));
    fst.AddArc(1, StdArc(2, 2, 0.5, 1));
    fst.SetFinal(1); // set final state
    fsts_[i] = fst;
  }
}

void PhoneTopology::Write(std::ostream &os, bool binary) const{
  WriteToken(os, binary, "<PhoneTopology>");
  if (!binary) os << "\n";
  int num_phones = fsts_.size() - 1;
  WriteToken(os, binary, "<NumPhones>");
  WriteBasicType(os, binary, num_phones);
  if (!binary) os << "\n";
  std::vector<fst::VectorFst<StdArc> >::iterator fiter = fsts_.begin(),
        fend = fsts_.end();
  for (++fiter; fiter != fend; ++fiter)
    WriteFstKaldi(os, binary, *fiter);
  WriteToken(os, binary, "</PhoneTopology>");
}

void PhoneTopology::Read(std::istream &is, bool binary) const{
  ExpectToken(is, binary, "<PhoneTopology>");
  int num_phones;
  ExpectToken(is, binary, "<NumPhones>");
  ReadBasicType(is, binary, &num_phones);
  fsts_.resize(num_phones + 1);
  std::vector<fst::VectorFst<StdArc> >::iterator fiter = fsts_.begin(),
        fend = fsts_.end();
  for (++fiter; fiter != fend; ++fiter)
    ReadFstKaldi(os, binary, fiter);
  ExpectToken(is, binary, "</PhoneTopology>");
}

bool PhonoTopology::IsAlignable() {
  std::vector<fst::VectorFst<StdArc> >::iterator fiter = fsts_.begin(),
        fend = fsts_.end();
  for (++fiter; fiter != fend; ++fiter) {
    // Get start state symbles
    unordered_set<int> syms;
    for (ArcIterator<Fst<Arc> >aiter(*fiter, fiter->Start()); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      syms.insert(arc.ilabel);
    }
    for (StateIterator<StdFst> siter(*fiter); !siter.Done(); siter.Next()) {
      typename Arc::StateId s = siter.Value();
      for (ArcIterator<Fst<Arc> >aiter(*fiter, s); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.nextstate == fiter->Start())
          return false;
        if (s != fiter->Start() && syms.find(arc.ilabel) != syms.end())
          return false;
      }
    }
  }
  return true;
}

}  // namespace chain
}  // namespace kaldi
