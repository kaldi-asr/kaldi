// fstext/rescale-test.cc

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

#include "fstext/rescale.h"
#include "fstext/fstext-utils.h"
#include "fstext/fst-test-utils.h"
// Just check that it compiles, for now.

namespace fst
{


template<class Arc> void TestComputeTotalWeight() {
  typedef typename Arc::Weight Weight;
  VectorFst<Arc> *fst = RandFst<Arc>();

  std::cout <<" printing FST at start\n";
  {
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  Weight max(-log(2.0));
  Weight tot = ComputeTotalWeight(*fst, max);
  std::cout << "Total weight is: " << tot.Value() << '\n';


  if (tot.Value() > max.Value()) {  // didn't max out...
    Weight tot2 = ShortestDistance(*fst);
    if (!ApproxEqual(tot, tot2, 0.05)) {
      KALDI_ERR << tot << " differs from " << tot2;
      assert(0);
    }
    std::cout << "our tot: " <<tot.Value() <<", shortest-distance tot: " << tot2.Value() << '\n';
  }

  delete fst;
}



void TestRescaleToStochastic() {
  typedef LogArc Arc;
  typedef Arc::Weight Weight;
  RandFstOptions opts;
  opts.allow_empty = false;
  VectorFst<Arc> *fst = RandFst<Arc>(opts);

  std::cout <<" printing FST at start\n";
  {
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");

  }
  float diff = 0.01;

  RescaleToStochastic(fst, diff);
  Weight tot = ShortestDistance(*fst),
      tot2 = ComputeTotalWeight(*fst, Weight(-log(2.0)));
  std::cerr <<  " tot is " << tot<<", tot2 = "<<tot2<<'\n';
  assert(ApproxEqual(tot2, Weight::One(), diff));

  delete fst;
}


} // end namespace fst


int main() {
  using namespace fst;
  for (int i = 0;i < 10;i++) {
    std::cout << "Testing with tropical\n";
    fst::TestComputeTotalWeight<StdArc>();
    std::cout << "Testing with log:\n";
    fst::TestComputeTotalWeight<LogArc>();
  }
  for (int i = 0;i < 10;i++) {
    std::cout << "i = "<<i<<'\n';
    fst::TestRescaleToStochastic();
  }
}
