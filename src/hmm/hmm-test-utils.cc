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

TransitionModel *GenRandTransitionModel(ContextDependency **ctx_dep_out) {
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

  HmmTopology topo = GenRandTopology(phones, num_pdf_classes);

  TransitionModel *trans_model = new TransitionModel(*ctx_dep, topo);

  if (ctx_dep_out == NULL) delete ctx_dep;
  else *ctx_dep_out = ctx_dep;
  return trans_model;
}

HmmTopology GetDefaultTopology(const std::vector<int32> &phones_in) {
  std::vector<int32> phones(phones_in);
  std::sort(phones.begin(), phones.end());
  KALDI_ASSERT(IsSortedAndUniq(phones) && !phones.empty());

  std::ostringstream topo_string;
  topo_string <<  "<Topology>\n"
      "<TopologyEntry>\n"
      "<ForPhones> ";
  for (size_t i = 0; i < phones.size(); i++)
    topo_string << phones[i] << " ";

  topo_string << "</ForPhones>\n"
      "<State> 0 <PdfClass> 0\n"
      "<Transition> 0 0.5\n"
      "<Transition> 1 0.5\n"
      "</State> \n"
      "<State> 1 <PdfClass> 1 \n"
      "<Transition> 1 0.5\n"
      "<Transition> 2 0.5\n"
      "</State>  \n"
      " <State> 2 <PdfClass> 2\n"
      " <Transition> 2 0.5\n"
      " <Transition> 3 0.5\n"
      " </State>   \n"
      " <State> 3 </State>\n"
      " </TopologyEntry>\n"
      " </Topology>\n";

  HmmTopology topo;
  std::istringstream iss(topo_string.str());
  topo.Read(iss, false);
  return topo;

}


HmmTopology GenRandTopology(const std::vector<int32> &phones_in,
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
    topo_string << "</ForPhones> ";
    bool ergodic = (RandInt(0, 1) == 0);
    if (ergodic) {
      // Note, this type of topology is not something we ever use in practice- it
      // has an initial nonemitting state (no PdfClass specified).  But it's
      // supported so we're testing it.
      std::vector<int32> state_to_pdf_class;
      state_to_pdf_class.push_back(-1);  // state zero, nonemitting.
      for (int32 i = 0; i < this_num_pdf_classes; i++) {
        int32 num_states = RandInt(1, 2);
        for (int32 j = 0; j < num_states; j++)
          state_to_pdf_class.push_back(i);
      }
      state_to_pdf_class.push_back(-1);  // final non-emitting state.
      { // state zero is nonemitting.  This is not something used in any current
        // example script.
        topo_string << "<State> 0\n";
        BaseFloat prob = 1.0 / (state_to_pdf_class.size() - 2);
        for (size_t i = 1; i + 1 < state_to_pdf_class.size(); i++) {
          topo_string << "<Transition> " << i << ' ' << prob << '\n';
        }
        topo_string << "</State>\n";
      }
      // ergodic part.
      for (size_t i = 1; i + 1 < state_to_pdf_class.size(); i++) {
        BaseFloat prob = 1.0 / (state_to_pdf_class.size() - 1);
        topo_string << "<State> " << i << " <PdfClass> "
                    << state_to_pdf_class[i] << '\n';
        for (size_t j = 1; j < state_to_pdf_class.size(); j++)
          topo_string << "<Transition> " << j << ' ' << prob << '\n';
        topo_string << "</State>\n";
      }
      // final, nonemitting state.  No pdf-class, no transitions.
      topo_string << "<State> " << (state_to_pdf_class.size() - 1) << " </State>\n";
    } else {
      // feedforward topology.
      int32 cur_state = 0;
      for (int32 pdf_class = 0; pdf_class < this_num_pdf_classes; pdf_class++) {
        int32 this_num_states = RandInt(1, 2);
        for (int32 s = 0; s < this_num_states; s++) {
          topo_string << "<State> " << cur_state << " <PdfClass> " << pdf_class
                      << "\n<Transition> " << cur_state << " 0.5\n<Transition> "
                      << (cur_state + 1) << " 0.5\n</State>\n";
          cur_state++;
        }
      }
      // final, non-emitting state.
      topo_string << "<State> " << cur_state << " </State>\n";
    }
    topo_string << "</TopologyEntry>\n";
  }
  topo_string << "</Topology>\n";

  HmmTopology topo;
  std::istringstream iss(topo_string.str());
  topo.Read(iss, false);
  return topo;
}

HmmTopology GenRandTopology() {
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

void GeneratePathThroughHmm(const HmmTopology &topology,
                            bool reorder,
                            int32 phone,
                            std::vector<std::pair<int32, int32> > *path) {
  path->clear();
  const HmmTopology::TopologyEntry &this_entry =
      topology.TopologyForPhone(phone);
  int32 cur_state = 0;  // start-state is always state zero.
  int32 num_states = this_entry.size(), final_state = num_states - 1;
  KALDI_ASSERT(num_states > 1);  // there has to be a final nonemitting state
  // that's different from the start state.
  std::vector<std::pair<int32, int32> > pending_self_loops;
  while (cur_state != final_state) {
    const HmmTopology::HmmState &cur_hmm_state = this_entry[cur_state];
    int32 num_transitions = cur_hmm_state.transitions.size(),
        transition_index = RandInt(0, num_transitions - 1);
    if (cur_hmm_state.forward_pdf_class != -1) {
      std::pair<int32, int32> pr(cur_state, transition_index);
      if (!reorder) {
        path->push_back(pr);
      } else {
        bool is_self_loop = (cur_state ==
                             cur_hmm_state.transitions[transition_index].first);
        if (is_self_loop) { // save these up, we'll put them after the forward
                            // transition.
          pending_self_loops.push_back(pr);
        } else {
          // non-self-loop: output it and then flush out any self-loops we
          // stored up.
          path->push_back(pr);
          path->insert(path->end(), pending_self_loops.begin(),
                       pending_self_loops.end());
          pending_self_loops.clear();
        }
      }
    }
    cur_state = cur_hmm_state.transitions[transition_index].first;
  }
  KALDI_ASSERT(pending_self_loops.empty());
}


void GenerateRandomAlignment(const ContextDependencyInterface &ctx_dep,
                             const TransitionModel &trans_model,
                             bool reorder,
                             const std::vector<int32> &phone_sequence,
                             std::vector<int32> *alignment) {
  int32 context_width = ctx_dep.ContextWidth(),
      central_position = ctx_dep.CentralPosition(),
      num_phones = phone_sequence.size();
  alignment->clear();
  for (int32 i = 0; i < num_phones; i++) {
    std::vector<int32> context_window;
    context_window.reserve(context_width);
    for (int32 j = i - central_position;
         j < i - central_position + context_width;
         j++) {
      if (j >= 0 && j < num_phones) context_window.push_back(phone_sequence[j]);
      else context_window.push_back(0);  // zero for out-of-window phones
    }
    // 'path' is the path through this phone's HMM, represented as
    // (emitting-HMM-state, transition-index) pairs
    std::vector<std::pair<int32, int32> > path;
    int32 phone = phone_sequence[i];
    GeneratePathThroughHmm(trans_model.GetTopo(), reorder, phone, &path);
    for (size_t k = 0; k < path.size(); k++) {
      const HmmTopology::TopologyEntry &entry =
          trans_model.GetTopo().TopologyForPhone(phone);
      int32 hmm_state = path[k].first,
          transition_index = path[k].second,
          forward_pdf_class = entry[hmm_state].forward_pdf_class,
          self_loop_pdf_class = entry[hmm_state].self_loop_pdf_class,
          forward_pdf_id, self_loop_pdf_id;
      bool ans = ctx_dep.Compute(context_window, forward_pdf_class, &forward_pdf_id);
      KALDI_ASSERT(ans && "context-dependency computation failed.");
      ans = ctx_dep.Compute(context_window, self_loop_pdf_class, &self_loop_pdf_id);
      KALDI_ASSERT(ans && "context-dependency computation failed.");
      int32 transition_state = trans_model.TupleToTransitionState(
                               phone, hmm_state, forward_pdf_id, self_loop_pdf_id),
          transition_id = trans_model.PairToTransitionId(transition_state,
                                                         transition_index);
      alignment->push_back(transition_id);
    }
  }
}


} // End namespace kaldi
