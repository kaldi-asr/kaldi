// fstext/determinize-lattice-test.cc

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

#include "fstext/determinize-lattice.h"
#include "fstext/pre-determinize.h"
#include "fstext/trivial-factor-weight.h"
#include "fstext/fst-test-utils.h"



namespace fst {


void TestLatticeStringRepository() {
  typedef int32 IntType;

  LatticeStringRepository<IntType> sr;
  typedef LatticeStringRepository<IntType>::Entry Entry;

  for(int i = 0; i < 100; i++) {
    int len = rand() % 5;
    vector<IntType> str(len), str2(rand() % 4);
    const Entry *e = NULL;
    for(int i = 0; i < len; i++) {
      str[i] = rand() % 5;
      e = sr.Successor(e, str[i]);
    }
    sr.ConvertToVector(e, &str2);
    assert(str == str2);

    int len2 = rand() % 5;
    str2.resize(len2);
    const Entry *f = sr.EmptyString(); // NULL
    for(int i = 0; i < len2; i++) {
      str2[i] = rand() % 5;
      f = sr.Successor(f, str2[i]);
    }
    vector<IntType> prefix, prefix2(rand() % 10),
        prefix3;
    for(int i = 0; i < len && i < len2; i++) {
      if(str[i] == str2[i]) prefix.push_back(str[i]);
      else break;
    }
    const Entry *g = sr.CommonPrefix(e, f);
    sr.ConvertToVector(g, &prefix2);
    sr.ConvertToVector(e, &prefix3);
    sr.ReduceToCommonPrefix(f, &prefix3);
    assert(prefix == prefix2);
    assert(prefix == prefix3);
    assert(sr.IsPrefixOf(g, e));
    assert(sr.IsPrefixOf(g, f));
    if(str.size() > prefix.size())
      assert(!sr.IsPrefixOf(e, g));
  }
}


// test that determinization proceeds correctly on general
// FSTs (not guaranteed determinzable, but we use the
// max-states option to stop it getting out of control).
template<class Arc> void TestDeterminizeLattice() {
  int max_states = 100; // don't allow more det-states than this.
  for(int i = 0; i < 100; i++) {
    VectorFst<Arc> *fst = RandFst<Arc>();
    std::cout << "FST before lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> ofst;
    try {
      DeterminizeLattice<TropicalWeight, int32>(*fst, &ofst, kDelta, NULL, max_states);
      std::cout << "FST after lattice-determinizing is:\n";
      {
        FstPrinter<Arc> fstprinter(ofst, NULL, NULL, NULL, false, true);
        fstprinter.Print(&std::cout, "standard output");
      }
    } catch (...) {
      std::cout << "Failed to lattice-determinize this FST (probably not determinizable)\n";
    }
  }
}

// test that determinization proceeds correctly on acyclic FSTs
// (guaranteed determinizable in this sense).
template<class Arc> void TestDeterminizeLattice2() {
  RandFstOptions opts;
  opts.acyclic = true;
  for(int i = 0; i < 100; i++) {
    VectorFst<Arc> *fst = RandFst<Arc>(opts);
    std::cout << "FST before lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> ofst;
    DeterminizeLattice<TropicalWeight, int32>(*fst, &ofst);
    std::cout << "FST after lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(ofst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
  }
}


} // end namespace fst

int main() {
  using namespace fst;
  for (int i = 0;i < 5;i++) {  // We would need more iterations to check
    TestLatticeStringRepository();
    TestDeterminizeLattice<StdArc>();
  }
  std::cout << "Tests succeeded\n";
}
