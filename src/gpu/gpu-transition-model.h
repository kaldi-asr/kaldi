#ifndef KALDI_HMM_GPU_TRANSITION_MODEL_H_
#define KALDI_HMM_GPU_TRANSITION_MODEL_H_

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "hmm/transition-model.h"

namespace kaldi{

struct GPUTransitionModel{

  struct Tuple {
    int32 phone;
    int32 hmm_state;
    int32 forward_pdf;
    int32 self_loop_pdf;
    Tuple() { }
    Tuple(int32 phone, int32 hmm_state, int32 forward_pdf, int32 self_loop_pdf):
      phone(phone), hmm_state(hmm_state), forward_pdf(forward_pdf), self_loop_pdf(self_loop_pdf) { }
    bool operator < (const Tuple &other) const {
      if (phone < other.phone) return true;
      else if (phone > other.phone) return false;
      else if (hmm_state < other.hmm_state) return true;
      else if (hmm_state > other.hmm_state) return false;
      else if (forward_pdf < other.forward_pdf) return true;
      else if (forward_pdf > other.forward_pdf) return false;
      else return (self_loop_pdf < other.self_loop_pdf);
    }
    bool operator == (const Tuple &other) const {
      return (phone == other.phone && hmm_state == other.hmm_state
              && forward_pdf == other.forward_pdf && self_loop_pdf == other.self_loop_pdf);
    }
  };

  HmmTopology topo_;

  /// Triples indexed by transition state minus one;
  /// the triples are in sorted order which allows us to do the reverse mapping from
  /// triple to transition state
  thrust::device_vector<Tuple> tuples_;

  /// Gives the first transition_id of each transition-state; indexed by
  /// the transition-state.  Array indexed 1..num-transition-states+1 (the last one
  /// is needed so we can know the num-transitions of the last transition-state.
  thrust::device_vector<int32> state2id_;

  /// For each transition-id, the corresponding transition
  /// state (indexed by transition-id).
  thrust::device_vector<int32> id2state_;

  thrust::device_vector<int32> id2pdf_id_;

  /// For each transition-id, the corresponding log-prob.  Indexed by transition-id.
  thrust::device_vector<BaseFloat> log_probs_; // Vector<BaseFloat> log_probs_;

  /// For each transition-state, the log of (1 - self-loop-prob).  Indexed by
  /// transition-state.
  thrust::device_vector<BaseFloat> non_self_loop_log_probs_;// Vector<BaseFloat> non_self_loop_log_probs_;

  /// This is actually one plus the highest-numbered pdf we ever got back from the
  /// tree (but the tree numbers pdfs contiguously from zero so this is the number
  /// of pdfs).
  int32 num_pdfs_;

  GPUTransitionModel() {}
  GPUTransitionModel(TransitionModel& t) : 
    topo_(t.GetTopo()),
    tuples_(t.tuples()),
    state2id_(t.state2id()),
    id2state_(t.id2state()),
    id2pdf_id_(t.id2pdf_id()),
    num_pdfs_(t.NumPdfs())
  {
    const size_t log_probs_dim = t.log_probs().SizeInBytes() / sizeof(BaseFloat);
    BaseFloat* log_probs_data = t.log_probs().Data();
    thrust::copy(
      log_probs_data,
      log_probs_data + log_probs_dim,
      log_probs_.begin()
    );

    const size_t non_self_loop_log_probs_dim = t.non_self_loop_log_probs().SizeInBytes() / sizeof(BaseFloat);
    BaseFloat* non_self_loop_log_probs_data = t.non_self_loop_log_probs().Data();
    thrust::copy(
      non_self_loop_log_probs_data, 
      non_self_loop_log_probs_data + non_self_loop_log_probs_dim, 
      non_self_loop_log_probs_.begin()
    );
  }
  
};

}

#endif