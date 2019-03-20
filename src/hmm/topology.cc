// hmm/topology.cc

// Copyright 2009-2011  Microsoft Corporation
//           2014-2019  Johns Hopkins University (author: Daniel Povey)
//           2019       Daniel Galvez
//           2019       Hossein Hadian

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

#include <vector>

#include "util/common-utils.h"
#include "hmm/topology.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"
#include "fstext/kaldi-fst-io.h"



namespace kaldi {

void Topology::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Topology>");
  if (!binary) {
    phones_.clear();
    phone2idx_.clear();
    entries_.clear();
    std::string token;
    while ( ! (is >> token).fail() ) {
      if (token == "</Topology>") {
        break; // finished parsing.
      } else if (token != "<TopologyEntry>") {
        KALDI_ERR << "Reading Topology object, expected </Topology> or "
            "<TopologyEntry>, got "<<token;
      } else {
        ExpectToken(is, binary, "<ForPhones>");
        std::vector<int32> phones;
        std::string s;
        while (1) {
          is >> s;
          if (is.fail())
            KALDI_ERR << "Reading Topology object, unexpected end of file "
                "while expecting phones.";
          if (s == "</ForPhones>") break;
          else {
            int32 phone;
            if (!ConvertStringToInteger(s, &phone))
              KALDI_ERR << "Reading Topology object, expected "
                        << "integer, got instead " << s;
            KALDI_ASSERT(phone > 0);
            phones.push_back(phone);
          }
        }

        int32 entry_index = entries_.size();
        fst::StdVectorFst fst;
        ReadFsaKaldi(is, &fst);
        entries_.push_back(fst);

        for (int32 phone : phones) {
          if (static_cast<int32>(phone2idx_.size()) <= phone)
            phone2idx_.resize(phone + 1, -1);  // -1 is invalid index.
          if (phone2idx_[phone] != -1) {
            KALDI_ERR << "Phone "
                      << phone << " appears in multiple topology entries.";
          }
          phone2idx_[phone] = entry_index;
          phones_.push_back(phone);
        }
        ExpectToken(is, binary, "</TopologyEntry>");
      }
    }
    std::sort(phones_.begin(), phones_.end());
    KALDI_ASSERT(IsSortedAndUniq(phones_));
  } else {
    ReadIntegerVector(is, binary, &phones_);
    ReadIntegerVector(is, binary, &phone2idx_);
    int32 number_topology_entries;
    ReadBasicType(is, binary, &number_topology_entries);
    for (size_t index = 0; index < number_topology_entries; ++index) {
      fst::StdVectorFst fst;
      ReadFstKaldi(is, binary, &fst);
      entries_.push_back(fst);
    }
  }
  Check();
}

// This function writes an FSA in text mode to an output stream.
template <class Arc>
static void WriteFsa(std::ostream &os, const fst::VectorFst<Arc> &fst) {
  os << '\n';
  bool acceptor = true, write_one = false;
  fst::FstPrinter<Arc> printer(fst, fst.InputSymbols(), fst.OutputSymbols(),
                               NULL, acceptor, write_one, "\t");
  printer.Print(&os, "<unknown>");
  if (os.fail())
    KALDI_ERR << "Stream failure detected writing FST to stream.";
  os << '\n';
  if (!os.good())
    KALDI_ERR << "Error writing FST to stream.";
}

void Topology::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Topology>");
  if (!binary) {
    for (int index = 0; index < entries_.size(); ++index) {
      WriteToken(os, binary, "<TopologyEntry>");
      WriteToken(os, binary, "<ForPhones>");
      for (auto phone: phones_)
        if (phone2idx_[phone] == index)
          os << phone << " ";
      os << "</ForPhones>";
      WriteFsa(os, entries_[index]);
      os << "</TopologyEntry>\n";
    }
  } else {
    WriteIntegerVector(os, binary, phones_);
    WriteIntegerVector(os, binary, phone2idx_);
    int32 number_topology_entries = entries_.size();
    WriteBasicType(os, binary, number_topology_entries);
    for (auto const& fst : entries_)
      WriteFstKaldi(os, binary, fst);
  }
  WriteToken(os, binary, "</Topology>");
}

void Topology::Check() {
  if (entries_.empty() || phones_.empty() || phone2idx_.empty())
    KALDI_ERR << "Empty object.";
  std::vector<bool> is_seen(entries_.size(), false);
  for (size_t i = 0; i < phones_.size(); i++) {
    int32 phone = phones_[i];
    if (static_cast<size_t>(phone) >= phone2idx_.size() ||
        static_cast<size_t>(phone2idx_[phone]) >= entries_.size())
      KALDI_ERR << "Phone " << phone << " has no valid index.";
    is_seen[phone2idx_[phone]] = true;
  }
  if (!std::accumulate(is_seen.begin(),
                       is_seen.end(), true, std::logical_and<bool>()))
    KALDI_ERR << "Entry with no corresponding phones.";

  for (auto const& entry: entries_) {
    int32 num_states = static_cast<int32>(entry.NumStates());
    if (num_states <= 1)
      KALDI_ERR << "Cannot only have one state (must have a "
                << "final state and a start state).";
  }

  for (auto& entry: entries_) {
    bool has_final_state = false;
    std::vector<int32> seen_pdf_classes;
    for (fst::StateIterator<fst::StdVectorFst> state_iter(entry);
         !state_iter.Done(); state_iter.Next()) {
      StateId state = state_iter.Value();
      if (entry.Final(state) != Weight::Zero())
        has_final_state = true;

      BaseFloat outward_prob_sum = exp(-entry.Final(state).Value());
      for (fst::ArcIterator<fst::StdVectorFst> aiter(entry, state);
           !aiter.Done(); aiter.Next()) {
        const fst::StdArc &arc(aiter.Value());
        if (arc.ilabel != arc.olabel)
          KALDI_ERR << "The topology must be an acceptor but ilabel != olabel.";
        if (arc.ilabel == 0)
          KALDI_ERR << "Epsilon arcs (pdf-class 0) are not allowed.";
        if (state != entry.Start() && arc.nextstate == entry.Start())
          KALDI_ERR << "Start state cannot have any inward transitions.";
        seen_pdf_classes.push_back(arc.ilabel);
        outward_prob_sum += exp(-arc.weight.Value());
      }
      if (!ApproxEqual(outward_prob_sum, 1.0))
        KALDI_WARN << "Outward transition probabilities should sum to 1.0 "
            "for each state";
    }
    if (!has_final_state) {
      KALDI_ERR << "Must have a final state.";
    }

    if (entry.Final(entry.Start()) != Weight::Zero())
      KALDI_ERR << "Start state must not be a final state.";

    if (entry.Start() != 0) {
      KALDI_ERR << "Topology::Check(), start state must be 0.";
    }

    SortAndUniq(&seen_pdf_classes);
    if (seen_pdf_classes.front() != 1 ||
        seen_pdf_classes.back() != static_cast<int32>(seen_pdf_classes.size()))
      KALDI_ERR << "pdf_classes are expected to be "
          "contiguous and start from 1.";

    fst::Connect(&entry);
    if (entry.NumStates() == 0)
      KALDI_ERR << "Some of the states in the topolgy are not reachable.";
  }
}

// Will throw if phone not covered.
const fst::StdVectorFst& Topology::TopologyForPhone(int32 phone) const {
  if (static_cast<size_t>(phone) >= phone2idx_.size()
      || phone2idx_[phone] == -1) {
    KALDI_ERR << "TopologyForPhone(), phone "<< phone <<" not covered.";
  }
  return entries_[phone2idx_[phone]];
}

int32 Topology::NumPdfClasses(int32 phone) const {
  // will throw if phone not covered.
  const fst::StdVectorFst &entry = TopologyForPhone(phone);

  std::set<int32> pdfs;
  for (fst::StateIterator<fst::StdVectorFst> siter(entry);
       !siter.Done(); siter.Next()) {
    StateId state_id = siter.Value();
    for (fst::ArcIterator<fst::StdVectorFst> aiter(entry, state_id);
         !aiter.Done(); aiter.Next()) {
      pdfs.insert(aiter.Value().ilabel);
    }
  }
  return pdfs.size();
}

void Topology::GetPhoneToNumPdfClasses(
    std::vector<int32> *phone2num_pdf_classes) const {
  KALDI_ASSERT(!phones_.empty());
  phone2num_pdf_classes->clear();
  phone2num_pdf_classes->resize(phones_.back() + 1, -1);
  for (auto phone: phones_)
    (*phone2num_pdf_classes)[phone] = NumPdfClasses(phone);
}

int32 Topology::MinLength(int32 phone) const {
  using Weight = typename fst::StdFst::Weight;
  using StateId = typename fst::StdFst::StateId;
  const fst::StdVectorFst& this_topo = TopologyForPhone(phone);
  // 1) Prepare a new FST with arc weight of 1.f and final state weight of 0.f
  // (Note that 0.f == Weight::One() in Tropical Semiring).
  // Since we are using the Std
  // We need to use a VectorFst in order to mutate members
  std::unique_ptr<fst::StdVectorFst> topo_copy(this_topo.Copy());

  std::vector<StateId> final_states;
  for (fst::StateIterator<fst::StdVectorFst> siter(*topo_copy);
       !siter.Done(); siter.Next()) {
    StateId state_id = siter.Value();

    if (topo_copy->Final(state_id) != Weight::Zero()) {
      final_states.push_back(state_id);
      topo_copy->SetFinal(state_id, Weight::One());
    }

    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(topo_copy.get(), state_id);
         !aiter.Done(); aiter.Next()) {
      Arc original_arc = aiter.Value();
      Arc distance_one_arc(original_arc.ilabel, original_arc.olabel,
                           Weight(1.0f), original_arc.nextstate);
      aiter.SetValue(distance_one_arc);
    }
  }
  KALDI_ASSERT(!final_states.empty());
  // Now run single-source nearest neightbors
  std::vector<Weight> distances;
  fst::ShortestDistance(*topo_copy, &distances);
  fst::NaturalLess<Weight> less;
  auto min_final_state_iter =
    std::min_element(final_states.begin(), final_states.end(),
                     [&distances, &less](StateId state1, StateId state2) {
                       return less(distances[state1], distances[state2]);
                     });
  Weight distance = distances[*min_final_state_iter];
  return static_cast<int32>(distance.Value());
}

bool Topology::operator==(const Topology &other) const {
  if (phones_ != other.phones_ || phone2idx_ != other.phone2idx_ ||
      entries_.size() != other.entries_.size()) {
    return false;
  } else {
    for(size_t i = 0; i < entries_.size(); ++i) {
      if (!fst::Equal(entries_[i], other.entries_[i], /*delta=*/0,
                      fst::kEqualFsts)) {
        return false;
      }
    }
    return true;
  }
}

} // End namespace kaldi
