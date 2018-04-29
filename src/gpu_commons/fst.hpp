// This FST.hpp files is modified version from 
// https://bitbucket.org/aargueta2/parallel-decoding


#ifndef FST_HPP
#define FST_HPP

#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <math.h>
#include "numberizer.hpp"

typedef float prob_t;
typedef int exponent;
typedef float mantissa;
typedef unsigned int prob_long_t;
typedef int state_t;
typedef int sym_t;

typedef std::tuple<state_t, state_t, sym_t, sym_t, prob_t> transition_t;
typedef std::tuple<state_t, state_t, sym_t, sym_t, prob_t, exponent, mantissa> transition_float;

struct fst {
  state_t initial;
  std::vector<transition_t> transitions;
  std::vector<std::tuple<state_t, prob_t>> finals;
  state_t num_states;
  sym_t num_inputs, num_outputs;

  fst() : num_states(0) { }

  void add_transition(state_t q, state_t r, sym_t f, sym_t e, prob_t p) {
    num_states = std::max(num_states, q+1);
    num_states = std::max(num_states, r+1);
    transitions.push_back(std::make_tuple(q, r, f, e, p));
  }

  void add_final(state_t q, prob_t p) {
    num_states = std::max(num_states, q+1);
    finals.push_back(std::make_tuple(q, p));
  }
};

struct fst_composed_probs{
  state_t initial;
  std::vector<transition_float> transition_f;
  std::vector<std::tuple<state_t, prob_t, exponent, mantissa>> finals;
  state_t num_states;
  sym_t num_inputs, num_outputs;

  fst_composed_probs() : num_states(0) { }

  void add_transition(state_t q, state_t r, sym_t f, sym_t e, prob_t p, exponent ee , mantissa m) {
    num_states = std::max(num_states, q+1);
    num_states = std::max(num_states, r+1);
    transition_f.push_back(std::make_tuple(q, r, f, e, p, ee, m));
  }

  void add_final(state_t q, prob_t p, exponent e, mantissa m) {
    num_states = std::max(num_states, q+1);
    finals.push_back(std::make_tuple(q, p, e, m));
  }
};

fst read_fst(const std::string &filename, const numberizer &inr, const numberizer &onr);
fst read_fst_csc(const std::string &filename, const numberizer &inr, const numberizer &onr,float neg);
fst read_fst_noLog(const std::string &filename, const numberizer &inr, const numberizer &onr);
fst read_fst_noNumberizer(const std::string &filename);
fst_composed_probs read_fst_exp_mantissa(const std::string &filename, const numberizer &inr, const numberizer &onr);

bool compare_input (const transition_t &x, const transition_t &y);
bool compare_input_fromstate_tostate(const transition_t &t1, const transition_t &t2);

bool compare_input_em(const transition_float &x, const transition_float &y);
bool compare_input_fromstate_tostate_em(const transition_float &t1, const transition_float &t2);
// Workaround for bug in nvcc
void sort_by_input(fst &m);
void sort_by_input_fromstate_tostate(fst &m);
void sort_by_input_tostate_fromstate(fst &m);

void sort_by_input_em(fst_composed_probs &m);
void sort_by_input_fromstate_tostate_em(fst_composed_probs &m);
void sort_by_input_tostate_fromstate_em(fst_composed_probs &m);
#endif
