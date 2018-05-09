#include <fstream>
#include <sstream>
#include <iostream>

#include "gpufst/fst.h"

namespace gpufst{

fst read_fst(const std::string &filename, const numberizer &inr, const numberizer &onr) {

  fst m;
  std::ifstream fst_file(filename);
  std::string line;
  bool first = true;

  m.num_inputs = inr.size();
  m.num_outputs = onr.size();

  while (getline(fst_file, line)) {
    // Count number of fields
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token)
      tokens.push_back(token);

    // Transition
    iss.str(line); iss.clear();
    state_t q, r;
    std::string fstr, estr;
    sym_t f, e;
    prob_t p;

    if (tokens.size() == 5 ) {
      iss >> q >> r >> fstr >> estr >> p;
      f = inr.word_to_num(fstr);
      e = onr.word_to_num(estr);
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_transition(q, r, f, e, log(p));

    // Final state
    } else if (tokens.size() == 2) {
      iss >> q >> p;
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_final(q, log(p));

    } else {
      throw std::runtime_error("wrong number of fields in line");
    }
  }
  return m;
}

fst read_fst_csc(const std::string &filename, const numberizer &inr, const numberizer &onr, float neg) {
  fst m;
  std::ifstream fst_file(filename);
  std::string line;
  bool first = true;

  m.num_inputs = inr.size();
  m.num_outputs = onr.size();

  while (getline(fst_file, line)) {
    // Count number of fields
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token)
      tokens.push_back(token);

    // Transition
    iss.str(line); iss.clear();
    state_t q, r;
    std::string fstr, estr;
    sym_t f, e;
    prob_t p;

    if (tokens.size() == 5) {
      // switch q and r to make it regular csr
      iss >> r >> q >> fstr >> estr >> p;
      f = inr.word_to_num(fstr);
      e = onr.word_to_num(estr);
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_transition(q, r, f, e, neg*log(p));

    // Final state
    } else if (tokens.size() == 2) {
      iss >> q >> p;
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_final(q, neg*log(p));

    } else {
      throw std::runtime_error("wrong number of fields in line");
    }
  }
  return m;
}


fst read_fst_noLog(const std::string &filename, const numberizer &inr, const numberizer &onr) {
  fst m;
  std::ifstream fst_file(filename);
  std::string line;
  bool first = true;

  m.num_inputs = inr.size();
  m.num_outputs = onr.size();

  while (getline(fst_file, line)) {
    // Count number of fields
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token)
      tokens.push_back(token);

    // Transition
    iss.str(line); iss.clear();
    state_t q, r;
    std::string fstr, estr;
    sym_t f, e;
    prob_t p;

    if (tokens.size() == 5) {
      iss >> q >> r >> fstr >> estr >> p;
      f = inr.word_to_num(fstr);
      e = onr.word_to_num(estr);
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_transition(q, r, f, e, p);

    // Final state
    } else if (tokens.size() == 2) {
      iss >> q >> p;
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_final(q, p);

    } else {
      throw std::runtime_error("wrong number of fields in line");
    }
  }
  return m;
}

fst read_fst_noNumberizer(const std::string &filename) {

  fst m;
  m.num_inputs = 0;
  m.num_outputs = 0;
  std::ifstream fst_file(filename);
  std::string line;
  bool first = true;

  while (getline(fst_file, line)) {
    // Count number of fields
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token)
      tokens.push_back(token);

    // Transition
    iss.str(line); iss.clear();
    state_t q, r;
    sym_t f, e;
    prob_t p;

//    std::cerr << "TOKEN SIZE: " << (int) tokens.size() << std::endl;
    if (tokens.size() >= 4 && tokens.size() <= 5) {
      iss >> q >> r >> f >> e;
      if(tokens.size() == 5) iss >> p;
      else p = 0.0;
      m.num_inputs = std::max(m.num_inputs, f + 1);
      m.num_outputs = std::max(m.num_outputs, e + 1);
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_transition(q, r, f, e, p);

    // Final state
    } else if (tokens.size() >= 1 && tokens.size() <= 2) {
      iss >> q;
      if(tokens.size() == 2) iss >> p;
      else p = 0.0;
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_final(q, p);

    } else {
      throw std::runtime_error("wrong number of fields in line");
    }
  }
  return m;
}

fst_composed_probs read_fst_exp_mantissa(const std::string &filename, const numberizer &inr, const numberizer &onr) {

  fst_composed_probs m;
  std::ifstream fst_file(filename);
  std::string line;
  bool first = true;

  m.num_inputs = inr.size();
  m.num_outputs = onr.size();

  while (getline(fst_file, line)) {
    // Count number of fields
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token)
      tokens.push_back(token);

    // Transition
    iss.str(line); iss.clear();
    state_t q, r;
    std::string fstr, estr;
    sym_t f, e;
    prob_t p;
    exponent ee;
    mantissa man;

    if (tokens.size() == 5) {
      iss >> q >> r >> fstr >> estr >>  p;
      man = frexp(p,&ee);
      f = inr.word_to_num(fstr);
      e = onr.word_to_num(estr);
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_transition(q, r, f, e, p, ee, man);

    // Final state
    } else if (tokens.size() == 2) {
      iss >> q >> p;
      man = frexp(p,&ee);
      if (first) {
        m.initial = q;
        first = false;
      }
      m.add_final(q, p, ee, man);

    } else {
      throw std::runtime_error("wrong number of fields in line");
    }
  }
  return m;
}


bool compare_input (const transition_t &x, const transition_t &y) {
  return std::get<2>(x) < std::get<2>(y);
}

bool compare_input_fromstate_tostate(const transition_t &t1, const transition_t &t2) {
  state_t q1, r1, q2, r2;
  sym_t f1, e1, f2, e2;
  prob_t p1, p2;
  std::tie(q1, r1, f1, e1, p1) = t1;
  std::tie(q2, r2, f2, e2, p2) = t2;
  return f1 < f2 || (f1 == f2 && q1 < q2) || (f1 == f2 && q1 == q2 && r1 < r2);
}

bool compare_input_tostate_fromstate(const transition_t &t1, const transition_t &t2) {
  state_t q1, r1, q2, r2;
  sym_t f1, e1, f2, e2;
  prob_t p1, p2;
  std::tie(q1, r1, f1, e1, p1) = t1;
  std::tie(q2, r2, f2, e2, p2) = t2;
  return f1 < f2 || (f1 == f2 && r1 < r2) || (f1 == f2 && r1 == r2 && q1 < q2);
}

bool compare_input_em(const transition_float &x, const transition_float &y) {
  return std::get<2>(x) < std::get<2>(y);
}

bool compare_input_fromstate_tostate_em(const transition_float &t1, const transition_float &t2) {
  state_t q1, r1, q2, r2;
  sym_t f1, e1, f2, e2;
  exponent ee1, ee2;
  prob_t p1,p2;
  mantissa m1, m2;
  std::tie(q1, r1, f1, e1, p1, ee1, m1) = t1;
  std::tie(q2, r2, f2, e2, p2, ee2, m2) = t2;
  return f1 < f2 || (f1 == f2 && q1 < q2) || (f1 == f2 && q1 == q2 && r1 < r2);
}

bool compare_input_tostate_fromstate_em(const transition_float &t1, const transition_float &t2) {
  state_t q1, r1, q2, r2;
  sym_t f1, e1, f2, e2;
  exponent ee1, ee2;
  prob_t p1,p2;
  mantissa m1, m2;
  std::tie(q1, r1, f1, e1, p1, ee1, m1) = t1;
  std::tie(q2, r2, f2, e2, p2, ee2, m2) = t2;
  return f1 < f2 || (f1 == f2 && r1 < r2) || (f1 == f2 && r1 == r2 && q1 < q2);
}

// Workaround for bug in nvcc

void sort_by_input(fst &m) {
  std::sort(m.transitions.begin(), m.transitions.end(), compare_input);
}

void sort_by_input_fromstate_tostate(fst &m) {
  std::sort(m.transitions.begin(), m.transitions.end(), compare_input_fromstate_tostate);
}

void sort_by_input_tostate_fromstate(fst &m) {
  std::sort(m.transitions.begin(), m.transitions.end(), compare_input_tostate_fromstate);
}
//em methods
void sort_by_input_em(fst_composed_probs &m) {
  std::sort(m.transition_f.begin(), m.transition_f.end(), compare_input_em);
}

void sort_by_input_fromstate_tostate_em(fst_composed_probs &m) {
  std::sort(m.transition_f.begin(), m.transition_f.end(), compare_input_fromstate_tostate_em);
}

void sort_by_input_tostate_fromstate_em(fst_composed_probs &m) {
  std::sort(m.transition_f.begin(), m.transition_f.end(), compare_input_tostate_fromstate_em);
}

}

