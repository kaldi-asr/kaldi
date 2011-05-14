// fstext/reorder-test.cc

// Copyright 2009-2011  Microsoft Corporation

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


#include "fstext/reorder.h"
#include "fstext/fstext-utils.h"
#include "fstext/fst-test-utils.h"



namespace fst
{



template<class Arc> static void TestReorder() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = RandFst<Arc>();
  VectorFst<Arc> fst2;

  WeightArcSort(fst);
  DfsReorder(*fst, &fst2);

  std::cout <<" printing before reordering\n";
  {
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  std::cout <<" printing after reordering\n";
  {
    FstPrinter<Arc> fstprinter(fst2, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));
  delete fst;
}



} // namespace fst

int main() {
  using namespace fst;
  for (int i = 0;i < 5;i++) {
    TestReorder<fst::StdArc>();
  }
}
