// This File is exact copy of 
// https://bitbucket.org/aargueta2/parallel-decoding


#ifndef GPUFST_GPUFST_GPU_FST_H
#define GPUFST_GPUFST_GPU_FST_H

#include <thrust/device_vector.h>
#include <vector>
#include <iostream>
#include <cfloat>
#include "gpufst/fst.h"
#include "gpufst/gpu-utils.h"
#include "gpufst/numberizer.h"

namespace gpufst{

struct gpu_fst {
  state_t initial;
  std::vector<int> input_offsets;
  thrust::device_vector<state_t> from_states, to_states;
  std::vector<sym_t> outputs;
  thrust::device_vector<sym_t> inputs;
  thrust::device_vector<prob_t> probs;
  thrust::device_vector<exponent> probs_e;
  thrust::device_vector<mantissa> probs_m;
  thrust::device_vector<state_t> final_states;
  thrust::device_vector<prob_t> final_probs;
  state_t num_states;
  sym_t num_inputs, num_outputs;

  gpu_fst(fst &&m)
    :
    initial(m.initial),
    num_states(m.num_states), 
    num_inputs(m.num_inputs),
    num_outputs(m.num_outputs),
    input_offsets(m.num_inputs+1),
    from_states(m.transitions.size()),
    to_states(m.transitions.size()),
    inputs(m.transitions.size()),
    outputs(m.transitions.size()),
    probs(m.transitions.size()),
    final_states(m.finals.size()),
    final_probs(m.finals.size())
  {
    int verbose = 0;
    if (verbose) std::cerr << "sorting transitions...";
    //std::sort(m.transitions.begin(), m.transitions.end(), compare_input);
    //sort_by_input(m);
    //sort_by_input_tostate_fromstate(m);
    sort_by_input_fromstate_tostate(m);
    if (verbose) std::cerr << "done.\ncomputing input offsets...";
    sym_t f_last = -1;
    for (int i=0; i<m.transitions.size(); i++) {
      state_t q, r;
      sym_t f, e;
      prob_t p;
      std::tie(q, r, f, e, p) = m.transitions[i];
      while (f > f_last) {
        f_last++;
        input_offsets[f_last] = i;
      }
      inputs[i] = f;
      outputs[i] = e;
    }
    input_offsets.back() = m.transitions.size();
    if (verbose) std::cerr << "done.\ncopying to device..." << std::endl;
    unzip_to_device<0>(m.transitions, from_states);
    unzip_to_device<1>(m.transitions, to_states);
    //unzip_to_device<3>(m.transitions, outputs);
    unzip_to_device<4>(m.transitions, probs);
    unzip_to_device<0>(m.finals, final_states);
    unzip_to_device<1>(m.finals, final_probs);
    if (verbose) std::cerr << "done.\n";
  }

  gpu_fst(fst_composed_probs &&m)
    :
    initial(m.initial),
    num_states(m.num_states), 
    num_inputs(m.num_inputs),
    num_outputs(m.num_outputs),
    input_offsets(m.num_inputs+1),
    from_states(m.transition_f.size()),
    to_states(m.transition_f.size()),
    inputs(m.transition_f.size()),
    outputs(m.transition_f.size()),
    probs(m.transition_f.size()),
    probs_e(m.transition_f.size()),
    probs_m(m.transition_f.size()),
    final_states(m.finals.size()),
    final_probs(m.finals.size())
  {
    int verbose = 0;
    if (verbose) std::cerr << "sorting transitions...";
    //std::sort(m.transitions.begin(), m.transitions.end(), compare_input);
    //sort_by_input(m);
    //sort_by_input_tostate_fromstate(m);
    sort_by_input_fromstate_tostate_em(m);
    if (verbose) std::cerr << "done.\ncomputing input offsets...";
    sym_t f_last = -1;
    for (int i=0; i<m.transition_f.size(); i++) {
      state_t q, r;
      sym_t f, e;
      prob_t p;
      exponent ee;
      mantissa m1;
      std::tie(q, r, f, e, p, ee, m1) = m.transition_f[i];
      while (f > f_last) {
        f_last++;
        input_offsets[f_last] = i;
      }
      inputs[i] = f;
      outputs[i] = e;
    }
    input_offsets.back() = m.transition_f.size();
    if (verbose) std::cerr << "done.\ncopying to device...";
    unzip_to_device<0>(m.transition_f, from_states);
    unzip_to_device<1>(m.transition_f, to_states);
    unzip_to_device<4>(m.transition_f, probs);
    unzip_to_device<5>(m.transition_f, probs_e);
    unzip_to_device<6>(m.transition_f, probs_m);
    unzip_to_device<0>(m.finals, final_states);
    unzip_to_device<1>(m.finals, final_probs);
    if (verbose) std::cerr << "done.\n";
  }
};

}

#endif
