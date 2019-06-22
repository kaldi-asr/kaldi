// hmm/hmm-test-utils.cc

// Copyright 2015   Johns Hopkins University (author: Daniel Povey)

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

#include "hmm/hmm-test-utils.h"

namespace kaldi {

Transitions *GenRandTransitions(ContextDependency **ctx_dep_out) {
  std::vector<int32> phones;
  phones.push_back(1);
  for (int32 i = 2; i < 20; i++)
    if (rand() % 2 == 0)
      phones.push_back(i);
  int32 N = 2 + rand() % 2, // context-size N is 2 or 3.
      P = rand() % N;  // Central-phone is random on [0, N)

  std::vector<int32> num_pdf_classes;

  ContextDependency *ctx_dep =
      GenRandContextDependencyLarge(phones, N, P,
                                    true, &num_pdf_classes);

  Topology topo = GenRandTopology(phones, num_pdf_classes);

  Transitions *trans_model = new Transitions(*ctx_dep, topo);

  if (ctx_dep_out == NULL) delete ctx_dep;
  else *ctx_dep_out = ctx_dep;
  return trans_model;
}

Topology GetDefaultTopology(const std::vector<int32> &phones_in) {
  std::vector<int32> phones(phones_in);
  std::sort(phones.begin(), phones.end());
  KALDI_ASSERT(IsSortedAndUniq(phones) && !phones.empty());

  std::ostringstream topo_string;
  topo_string <<  "<Topology>\n"
      "<TopologyEntry>\n"
      "<ForPhones> ";
  for (size_t i = 0; i < phones.size(); i++)
    topo_string << phones[i] << " ";

  topo_string <<
      "</ForPhones>\n"
      "0  1  1  0.0\n"
      "1  1  1  0.693\n"
      "1  2  2  0.693\n"
      "2  2  2  0.693\n"
      "2  3  3  0.693\n"
      "3  3  3  0.693\n"
      "3  0.693\n\n"
      "</TopologyEntry>\n"
      "</Topology>\n";

  Topology topo;
  std::istringstream iss(topo_string.str());
  topo.Read(iss, false);
  return topo;

}


Topology GenRandTopology(const std::vector<int32> &phones_in,
                            const std::vector<int32> &num_pdf_classes) {
  std::vector<int32> phones(phones_in);
  std::sort(phones.begin(), phones.end());
  KALDI_ASSERT(IsSortedAndUniq(phones) && !phones.empty());

  std::ostringstream topo_string;

   std::map<int32, std::vector<int32> > num_pdf_classes_to_phones;
  for (size_t i = 0; i < phones.size(); i++) {
    int32 p = phones[i];
    KALDI_ASSERT(static_cast<size_t>(p) < num_pdf_classes.size());
    int32 n = num_pdf_classes[p];
    KALDI_ASSERT(n > 0 && "num-pdf-classes cannot be zero.");
    num_pdf_classes_to_phones[n].push_back(p);
  }

  topo_string <<  "<Topology>\n";
  std::map<int32, std::vector<int32> >::const_iterator
      iter = num_pdf_classes_to_phones.begin(),
      end = num_pdf_classes_to_phones.end();
  for (; iter != end; ++iter) {
    topo_string << "<TopologyEntry>\n"
        "<ForPhones> ";
    int32 this_num_pdf_classes = iter->first;
    const std::vector<int32> &phones = iter->second;
    for (size_t i = 0; i < phones.size(); i++)
      topo_string << phones[i] << " ";
    topo_string << "</ForPhones>\n";
    bool ergodic = (RandInt(0, 1) == 0);
    if (ergodic) {
      // Note, this type of topology is not something we ever use in practice- it
      // has an initial nonemitting state (no PdfClass specified).  But it's
      // supported so we're testing it.
      std::vector<int32> state_to_pdf_class;
      state_to_pdf_class.push_back(-1);  // state zero, nonemitting.
      for (int32 i = 1; i <= this_num_pdf_classes; i++) {
        int32 num_states = RandInt(1, 2);
        for (int32 j = 0; j < num_states; j++)
          state_to_pdf_class.push_back(i);
      }
      state_to_pdf_class.push_back(-1);  // final non-emitting state.
      { // state zero is nonemitting.  This is not something used in any current
        // example script.
        BaseFloat prob = 1.0 / (state_to_pdf_class.size() - 2);
        for (size_t i = 1; i + 1 < state_to_pdf_class.size(); i++)
          topo_string << "0 " << i << ' ' << state_to_pdf_class[i]
                      << ' ' << -Log(prob) << '\n';
      }
      // ergodic part.
      for (size_t i = 1; i + 1 < state_to_pdf_class.size(); i++) {
        BaseFloat prob = 1.0 / (state_to_pdf_class.size() - 1);
        for (size_t j = 1; j < state_to_pdf_class.size(); j++)
          topo_string << i << ' ' << j << ' '
                      << state_to_pdf_class[i] << ' ' << -Log(prob) << '\n';
      }
      // final, nonemitting state.  No pdf-class, no transitions.
      topo_string << (state_to_pdf_class.size() - 1) << "\n\n";
    } else {
      // feedforward topology.
      int32 cur_state = 0;
      for (int32 pdf_class = 1; pdf_class <= this_num_pdf_classes; pdf_class++) {
        int32 this_num_states = RandInt(1, 2);
        for (int32 s = 0; s < this_num_states; s++) {
          topo_string << cur_state << " " << (cur_state + 1) << " " << pdf_class << "\n";
          cur_state++;
        }
      }
      // final, non-emitting state.
      topo_string << cur_state << "\n\n";
    }
    topo_string << "</TopologyEntry>\n";
  }
  topo_string << "</Topology>\n";

  Topology topo;
  std::istringstream iss(topo_string.str());
  topo.Read(iss, false);
  return topo;
}

Topology GenRandTopology() {
  std::vector<int32> phones;
  phones.push_back(1);
  for (int32 i = 2; i < 20; i++)
    if (rand() % 2 == 0)
      phones.push_back(i);
  if (RandInt(0, 1) == 0) {
    return GetDefaultTopology(phones);
  } else {
    std::vector<int32> num_pdf_classes(phones.back() + 1, -1);
    for (int32 i = 0; i < phones.size(); i++)
      num_pdf_classes[phones[i]] = RandInt(1, 5);
    return GenRandTopology(phones, num_pdf_classes);
  }
}

void GeneratePathThroughHmm(const Topology &topology,
                            int32 phone,
                            std::vector<std::pair<int32, int32> > *path) {
  path->clear();
  auto const &this_entry = topology.TopologyForPhone(phone); // an FST
  int32 cur_state = 0;  // start-state is always state zero.

  // Note: final_state == num_states - 1 is actually not something
  // that would be generally true, but it is true for the topologies we
  // use in the test code.
  int32 num_states = this_entry.NumStates(), final_state = num_states - 1;
  KALDI_ASSERT(num_states > 1);  // there has to be a final nonemitting state
  // that's different from the start state.

  while (cur_state != final_state) {
    int32 num_transitions = this_entry.NumArcs(cur_state),
        arc_index = RandInt(0, num_transitions - 1);
    fst::ArcIterator<fst::StdVectorFst> aiter(this_entry, cur_state);
    aiter.Seek(arc_index);
    auto const &arc(aiter.Value());
    KALDI_ASSERT(arc.ilabel > 0);
    std::pair<int32, int32> pr(cur_state, arc_index);
    path->push_back(pr);
    cur_state = arc.nextstate;
  }
}


void GenerateRandomAlignment(const ContextDependencyInterface &ctx_dep,
                             const Transitions &trans_model,
                             const std::vector<int32> &phone_sequence,
                             std::vector<int32> *alignment) {
  int32 context_width = ctx_dep.ContextWidth(),
      central_position = ctx_dep.CentralPosition(),
      num_phones = phone_sequence.size();

  auto all_phones = trans_model.GetPhones();
  int32 model_max_phone = *std::max_element(all_phones.begin(),
                                            all_phones.end());
  alignment->clear();
  for (int32 i = 0; i < num_phones; i++) {
    KALDI_ASSERT(phone_sequence[i] > 0
                 && phone_sequence[i] <= model_max_phone);
    std::vector<int32> context_window;
    context_window.reserve(context_width);
    for (int32 j = i - central_position;
         j < i - central_position + context_width;
         j++) {
      if (j >= 0 && j < num_phones) context_window.push_back(phone_sequence[j]);
      else context_window.push_back(0);  // zero for out-of-window phones
    }
    // 'path' is the path through this phone's HMM, represented as
    // (source-HMM-state, transition-index) pairs
    std::vector<std::pair<int32, int32> > path;
    int32 phone = phone_sequence[i];
    GeneratePathThroughHmm(trans_model.GetTopo(), phone, &path);
    for (size_t k = 0; k < path.size(); k++) {
      auto const &entry = trans_model.GetTopo().TopologyForPhone(phone);
      int32 hmm_state = path[k].first,
          arc_index = path[k].second,
          forward_pdf_id, self_loop_pdf_id;
      fst::ArcIterator<fst::StdVectorFst> aiter(entry, hmm_state);
      aiter.Seek(arc_index);
      auto const &arc(aiter.Value());
      int32 forward_pdf_class = arc.ilabel,
          self_loop_pdf_class = -1;
      for (fst::ArcIterator<fst::StdVectorFst> aiter_next(entry, arc.nextstate);
           !aiter_next.Done(); aiter_next.Next())
        if (aiter_next.Value().nextstate == arc.nextstate)
          self_loop_pdf_class = aiter_next.Value().ilabel;

      bool ans = ctx_dep.Compute(context_window, forward_pdf_class, &forward_pdf_id);
      KALDI_ASSERT(ans && "context-dependency computation failed.");
      if (self_loop_pdf_class != -1) {
        ans = ctx_dep.Compute(context_window, self_loop_pdf_class, &self_loop_pdf_id);
        KALDI_ASSERT(ans && "context-dependency computation failed.");
      } else {
        self_loop_pdf_id = -1;
      }
      int32 transition_id = trans_model.TupleToTransitionId(phone, hmm_state, arc_index,
                                                            forward_pdf_id, self_loop_pdf_id);
      alignment->push_back(transition_id);
    }
  }
}


} // End namespace kaldi
