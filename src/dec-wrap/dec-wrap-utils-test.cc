
// -*- coding: utf-8 -*-
/* Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
 * MERCHANTABLITY OR NON-INFRINGEMENT.
 * See the Apache 2 License for the specific language governing permissions and
 * limitations under the License. */
#include "dec-wrap/dec-wrap-utils.h"
#include "dec-wrap/fst-equal.h"
#include <fst/fst.h>
#include <stdlib.h>

using namespace kaldi;
using namespace fst;

void test_fst_equal() {
  VectorFst<LogArc> *t = VectorFst<LogArc>::Read("T.fst");
  VectorFst<LogArc> *t2 = VectorFst<LogArc>::Read("T.fst");
  KALDI_ASSERT(Equal(*t, *t2, 0.001));
  delete t2;
  VectorFst<LogArc> *s = VectorFst<LogArc>::Read("S.fst");
  KALDI_ASSERT(!Equal(*t, *s, 0.001));
  delete t;
}


void test_posterior_sum_to_one(const std::vector<std::string> & tests) {
  typedef typename LogArc::StateId StateId;
  for (size_t k = 0; k < tests.size(); ++k) {
    VectorFst<LogArc> *t = VectorFst<LogArc>::Read(tests[k]);
    std::vector<double> alpha;
    std::vector<double> beta;
    ComputeLatticeAlphasAndBetas(*t, &alpha, &beta);
    MovePostToArcs(t, alpha, beta);

    StateId num_states = t->NumStates();
    for (StateId s = 0; s < num_states; s++) {
      double tot_prob = kLogZeroDouble;
      bool empty = true;
      for (MutableArcIterator<VectorFst<LogArc> > it(t, s); !it.Done(); it.Next()) {
        empty = false;
        const LogArc &arc = it.Value();
        double arc_prob = -ConvertToCost(arc.weight);
        tot_prob = LogAdd(tot_prob, arc_prob);
      }
      std::cerr << "Total probability for state " << s << " in " << tests[k]
        << "is " << tot_prob << std::endl;
      if(!empty && (std::abs(tot_prob) > 0.1)) {
        KALDI_ERR << "Total probability for state " << s << " in " << tests[k] 
          << " is not summing to Log(1) = 0 " << std::endl
          << "The real value is " << tot_prob << std::endl;
      }
    }

    delete t;
  }
}


void test_MovePostToArc(const std::vector<std::string> &tests) {
  for (size_t k = 0; k < tests.size(); ++k) {
    VectorFst<LogArc> *t = VectorFst<LogArc>::Read(tests[k]);
    std::vector<double> alpha;
    std::vector<double> beta;
    ComputeLatticeAlphasAndBetas(*t, &alpha, &beta);
    MovePostToArcs(t, alpha, beta);

    // save the file with posteriors 
    std::ofstream logfile;
    std::string prefix("posterior_");
    std::string post_name = prefix + tests[k];
    std::cerr << "writing posterior lattice " << post_name << std::endl;
    logfile.open(post_name.c_str());
    t->Write(logfile, fst::FstWriteOptions());
    logfile.close();

    // TODO compare the file with precomputed results(TODO)
    delete t;
  }
}


void test_ComputeLatticeAlphasAndBetas(const std::vector<std::string> &tests) {
  for (size_t k = 0; k < tests.size(); ++k) {
    VectorFst<LogArc> *t = VectorFst<LogArc>::Read(tests[k]);
    std::vector<double> alpha;
    std::vector<double> beta;
    double tot_post = ComputeLatticeAlphasAndBetas(*t, &alpha, &beta);
    std::cerr << tests[k] << "posterior probability: " << tot_post << std::endl;
    for (size_t i = 0; i < alpha.size(); ++i) {
      std::cerr << tests[k] << ": alpha[" << i << "] = " << alpha[i] 
        << " beta[" << i << "] = " << beta[i] << std::endl;
    }
    delete t;
  }
}

int main() {
  if(!system(NULL) || system("./testing_fst.sh")) {
    std::cerr << "The testing FSTs are not created" << std::endl;
    exit(1);
  }
  // generated fst from testing_fst - names must match
  std::vector<std::string> test_fst;
  test_fst.push_back("non_symetric.fst");
  test_fst.push_back("symetric.fst");
  test_fst.push_back("symetric_end.fst");
  test_fst.push_back("symetric_middle.fst");
  test_fst.push_back("negative.fst");
  test_fst.push_back("negative_end.fst");

  // launching tests
  test_ComputeLatticeAlphasAndBetas(test_fst);
  test_MovePostToArc(test_fst);

  std::vector<std::string> test_post_sum;
  test_post_sum.push_back("non_symetric.fst");
  test_post_sum.push_back("symetric.fst");
  test_post_sum.push_back("symetric_end.fst");
  // test_post_sum.push_back("symetric_middle.fst"); // failing as expected :(
  // test_post_sum.push_back("negative.fst"); // failing as expected
  test_post_sum.push_back("negative_end.fst"); // TODO passing but correct?
  test_posterior_sum_to_one(test_post_sum);
  // test_fst_equal();
  return 0;
}
