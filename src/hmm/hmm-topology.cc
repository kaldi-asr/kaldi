// hmm/hmm-topology.cc

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

#include <vector>

#include "hmm/hmm-topology.h"
#include "util/text-utils.h"


namespace kaldi {



void HmmTopology::GetPhoneToNumPdfClasses(std::vector<int32> *phone2num_pdf_classes) const {
  KALDI_ASSERT(!phones_.empty());
  phone2num_pdf_classes->clear();
  phone2num_pdf_classes->resize(phones_.back() + 1, -1);
  for (size_t i = 0; i < phones_.size(); i++)
    (*phone2num_pdf_classes)[phones_[i]] = NumPdfClasses(phones_[i]);
}

void HmmTopology::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Topology>");
  if (!binary) {  // Text-mode read, different "human-readable" format.
    phones_.clear();
    phone2idx_.clear();
    entries_.clear();
    std::string token;
    while ( ! (is >> token).fail() ) {
      if (token == "</Topology>") { break; } // finished parsing.
      else  if (token != "<TopologyEntry>") {
        KALDI_ERR << "Reading HmmTopology object, expected </Topology> or <TopologyEntry>, got "<<token;
      } else {
        ExpectToken(is, binary, "<ForPhones>");
        std::vector<int32> phones;
        std::string s;
        while (1) {
          is >> s;
          if (is.fail()) KALDI_ERR << "Reading HmmTopology object, unexpected end of file while expecting phones.";
          if (s == "</ForPhones>") break;
          else {
            int32 phone;
            if (!ConvertStringToInteger(s, &phone))
              KALDI_ERR << "Reading HmmTopology object, expected integer, got instead "<<s;
            phones.push_back(phone);
          }
        }

        std::vector<HmmState> this_entry;
        std::string token;
        ReadToken(is, binary, &token);
        while (token != "</TopologyEntry>") {
          if (token != "<State>")
            KALDI_ERR << "Expected </TopologyEntry> or <State>, got instead "<<token;
          int32 state;
          ReadBasicType(is, binary, &state);
          if (state != static_cast<int32>(this_entry.size()))
            KALDI_ERR << "States are expected to be in order from zero, expected "
                      << this_entry.size() <<  ", got " << state;
          ReadToken(is, binary, &token);
          int32 pdf_class = kNoPdf;  // -1 by default, means no pdf.
          if (token == "<PdfClass>") {
            ReadBasicType(is, binary, &pdf_class);
            ReadToken(is, binary, &token);
          }
          this_entry.push_back(HmmState(pdf_class));
          while (token == "<Transition>") {
            int32 dst_state;
            BaseFloat trans_prob;
            ReadBasicType(is, binary, &dst_state);
            ReadBasicType(is, binary, &trans_prob);
            this_entry.back().transitions.push_back(std::make_pair(dst_state, trans_prob));  
            ReadToken(is, binary, &token);
          }
          if(token == "<Final>") // TODO: remove this clause after a while.
            KALDI_ERR << "You are trying to read old-format topology with new Kaldi.";
          if (token != "</State>")
            KALDI_ERR << "Reading HmmTopology,  unexpected token "<<token;
          ReadToken(is, binary, &token);
        }
        int32 my_index = entries_.size();
        entries_.push_back(this_entry);

        for (size_t i = 0; i < phones.size(); i++) {
          int32 phone = phones[i];
          if (static_cast<int32>(phone2idx_.size()) <= phone)
            phone2idx_.resize(phone+1, -1);  // -1 is invalid index.
          KALDI_ASSERT(phone > 0);
          if (phone2idx_[phone] != -1)
            KALDI_ERR << "Phone with index "<<(i)<<" appears in multiple topology entries.";
          phone2idx_[phone] = my_index;
          phones_.push_back(phone);
        }
      }
    }
    std::sort(phones_.begin(), phones_.end());
    KALDI_ASSERT(IsSortedAndUniq(phones_));
  } else {  // binary I/O, just read member objects directly from disk.
    ReadIntegerVector(is, binary, &phones_);
    ReadIntegerVector(is, binary, &phone2idx_);
    int32 sz;
    ReadBasicType(is, binary, &sz);
    entries_.resize(sz);
    for (int32 i = 0; i < sz; i++) {
      int32 thist_sz;
      ReadBasicType(is, binary, &thist_sz);
      entries_[i].resize(thist_sz);
      for (int32 j = 0 ; j < thist_sz; j++) {
        ReadBasicType(is, binary, &(entries_[i][j].pdf_class));
        int32 thiss_sz;
        ReadBasicType(is, binary, &thiss_sz);
        entries_[i][j].transitions.resize(thiss_sz);
        for (int32 k = 0; k < thiss_sz; k++) {
          ReadBasicType(is, binary, &(entries_[i][j].transitions[k].first));
          ReadBasicType(is, binary, &(entries_[i][j].transitions[k].second));
        }
      }
    }
    ExpectToken(is, binary, "</Topology>");
  }
  Check();  // Will throw if not ok.
}


void HmmTopology::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Topology>");
  if (!binary) {  // Text-mode write.
    os << "\n";
    for (int32 i = 0; i < static_cast<int32> (entries_.size()); i++) {
      WriteToken(os, binary, "<TopologyEntry>");
      os << "\n";
      WriteToken(os, binary, "<ForPhones>");
      os << "\n";
      for (size_t j = 0; j < phone2idx_.size(); j++) {
        if (phone2idx_[j] == i)
          os << j << " ";
      }
      os << "\n";
      WriteToken(os, binary, "</ForPhones>");
      os << "\n";
      for (size_t j = 0; j < entries_[i].size(); j++) {
        WriteToken(os, binary, "<State>");
        WriteBasicType(os, binary, static_cast<int32>(j));
        if (entries_[i][j].pdf_class != kNoPdf) {
          WriteToken(os, binary, "<PdfClass>");
          WriteBasicType(os, binary, entries_[i][j].pdf_class);
        }
        for (size_t k = 0; k < entries_[i][j].transitions.size(); k++) {
          WriteToken(os, binary, "<Transition>");
          WriteBasicType(os, binary, entries_[i][j].transitions[k].first);
          WriteBasicType(os, binary, entries_[i][j].transitions[k].second);
        }
        WriteToken(os, binary, "</State>");
        os << "\n";
      }
      WriteToken(os, binary, "</TopologyEntry>");
      os << "\n";
    }
  } else {
    WriteIntegerVector(os, binary, phones_);
    WriteIntegerVector(os, binary, phone2idx_);
    WriteBasicType(os, binary, static_cast<int32>(entries_.size()));
    for (size_t i = 0; i < entries_.size(); i++) {
      WriteBasicType(os, binary, static_cast<int32>(entries_[i].size()));
      for (size_t j = 0; j < entries_[i].size(); j++) {
        WriteBasicType(os, binary, entries_[i][j].pdf_class);
        WriteBasicType(os, binary, static_cast<int32>(entries_[i][j].transitions.size()));
        for (size_t k = 0; k < entries_[i][j].transitions.size(); k++) {
          WriteBasicType(os, binary, entries_[i][j].transitions[k].first);
          WriteBasicType(os, binary, entries_[i][j].transitions[k].second);
        }
      }
    }
  }
  WriteToken(os, binary, "</Topology>");
  if (!binary) os << "\n";
}

void HmmTopology::Check() {
  if (entries_.empty() || phones_.empty() || phone2idx_.empty())
    KALDI_ERR << "HmmTopology::Check(), empty object.";
  std::vector<bool> is_seen(entries_.size(), false);
  for (size_t i = 0; i < phones_.size(); i++) {
    int32 phone = phones_[i];
    if (static_cast<size_t>(phone) >= phone2idx_.size() ||
        static_cast<size_t>(phone2idx_[phone]) >= entries_.size())
      KALDI_ERR << "HmmTopology::Check(), phone has no valid index.";
    is_seen[phone2idx_[phone]] = true;
  }
  for (size_t i = 0; i < entries_.size(); i++) {
    if (!is_seen[i])
      KALDI_ERR << "HmmTopoloy::Check(), entry with no corresponding phones.";
    int32 num_states = static_cast<int32>(entries_[i].size());
    if (num_states <= 1)
      KALDI_ERR << "HmmTopology::Check(), cannot only have one state (i.e., must "
          "have at least one emitting state).";
    if (!entries_[i][num_states-1].transitions.empty())
      KALDI_ERR << "HmmTopology::Check(), last state must have no transitions.";
    // not sure how necessary this next stipulation is.
    if (entries_[i][num_states-1].pdf_class != kNoPdf) 
      KALDI_ERR << "HmmTopology::Check(), last state must not be emitting.";

    std::vector<bool> has_trans_in(num_states, false);
    std::vector<int32> seen_pdf_classes;

    for (int32 j = 0; j < num_states; j++) {  // j is the state-id.
      BaseFloat tot_prob = 0.0;
      if (entries_[i][j].pdf_class != kNoPdf)
        seen_pdf_classes.push_back(entries_[i][j].pdf_class);
      std::set<int32> seen_transition;
      for (int32 k = 0;
           static_cast<size_t>(k) < entries_[i][j].transitions.size();
           k++) {
        tot_prob += entries_[i][j].transitions[k].second;
        if (entries_[i][j].transitions[k].second <= 0.0)
          KALDI_ERR << "HmmTopology::Check(), negative or zero transition prob.";
        int32 dst_state = entries_[i][j].transitions[k].first;
        // The commented code in the next few lines disallows a completely
        // skippable phone, as this would cause to stop working some mechanisms
        // that are being built, which enable the creation of phone-level lattices
        // and rescoring these with a different lexicon and LM.
        if (dst_state == num_states-1 // && j != 0
            && entries_[i][j].pdf_class == kNoPdf)
          KALDI_ERR << "We do not allow any state to be "
              "nonemitting and have a transition to the final-state (this would "
              "stop the SplitToPhones function from identifying the last state "
              "of a phone.";
        if (dst_state < 0 || dst_state >= num_states)
          KALDI_ERR << "HmmTopology::Check(), invalid dest state " << (dst_state);
        if (seen_transition.count(dst_state) != 0)
          KALDI_ERR << "HmmTopology::Check(), duplicate transition found.";
        if (dst_state == k) {  // self_loop...
          KALDI_ASSERT(entries_[i][j].pdf_class != kNoPdf && "Nonemitting states cannot have self-loops.");
        }
        seen_transition.insert(dst_state);
        has_trans_in[dst_state] = true;
      }
      if (j+1 < num_states) {
        KALDI_ASSERT(tot_prob > 0.0 && "Non-final state must have transitions out."
                     "(with nonzero probability)");
        if (fabs(tot_prob - 1.0) > 0.01)
          KALDI_WARN << "Total probability for state " << j <<
              " in topology entry is " << tot_prob;
      } else
        KALDI_ASSERT(tot_prob == 0.0);
    }
    // make sure all but start state have input transitions.
    for (int32 j = 1; j < num_states; j++) 
      if (!has_trans_in[j])
        KALDI_ERR << "HmmTopology::Check, state "<<(j)<<" has no input transitions.";
    SortAndUniq(&seen_pdf_classes);
    if (seen_pdf_classes.front() != 0 ||
        seen_pdf_classes.back() != static_cast<int32>(seen_pdf_classes.size()) - 1) {
      KALDI_ERR << "HmmTopology::Check(), pdf_classes are expected to be "
          "contiguous and start from zero.";
    }
  }
}

const HmmTopology::TopologyEntry& HmmTopology::TopologyForPhone(int32 phone) const {  // Will throw if phone not covered.
  if (static_cast<size_t>(phone) >= phone2idx_.size() || phone2idx_[phone] == -1) {
    KALDI_ERR << "TopologyForPhone(), phone "<<(phone)<<" not covered.";
  }
  return entries_[phone2idx_[phone]];
}

int32 HmmTopology::NumPdfClasses(int32 phone) const {
  // will throw if phone not covered.
  const TopologyEntry &entry = TopologyForPhone(phone);
  int32 max_pdf_class = 0;
  for (size_t i = 0; i < entry.size(); i++)
    max_pdf_class = std::max(max_pdf_class, entry[i].pdf_class);
  return max_pdf_class+1;
}


} // End namespace kaldi
