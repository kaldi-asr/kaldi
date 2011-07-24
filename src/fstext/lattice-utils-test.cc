// fstext/lattice-utils-test.cc

// Copyright 2011  Microsoft Corporation

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

#include "fstext/lattice-utils.h"
#include "fstext/fst-test-utils.h"


namespace fst {

template<class Weight, class Int> void TestConvert(bool invert) {
  typedef ArcTpl<Weight> Arc;
  typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;
  for(int i = 0; i < 5; i++) {
    VectorFst<Arc> *fst = RandFst<Arc>();
    std::cout << "FST before converting to compact-arc is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<CompactArc> ofst;
    ConvertLatticeToCompact<Weight, Int>(*fst, &ofst, invert);

    std::cout << "FST after converting is:\n";
    {
      FstPrinter<CompactArc> fstprinter(ofst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> origfst;
    ConvertLatticeFromCompact<Weight, Int>(ofst, &origfst, invert);
    std::cout << "FST after back conversion is:\n";
    {
      FstPrinter<Arc> fstprinter(origfst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    
    assert(RandEquivalent(*fst, origfst, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));
    delete fst;
  }
}


// use TestConvertPair when the Weight can be constructed from
// a pair of floats.
template<class Weight, class Int> void TestConvertPair(bool invert) {
  typedef ArcTpl<Weight> Arc;
  typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;
  for(int i = 0; i < 5; i++) {
    VectorFst<Arc> *fst = RandPairFst<Arc>();
    std::cout << "FST before converting to compact-arc is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<CompactArc> ofst;
    ConvertLatticeToCompact<Weight, Int>(*fst, &ofst, invert);

    std::cout << "FST after converting is:\n";
    {
      FstPrinter<CompactArc> fstprinter(ofst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> origfst;
    ConvertLatticeFromCompact<Weight, Int>(ofst, &origfst, invert);
    std::cout << "FST after back conversion is:\n";
    {
      FstPrinter<Arc> fstprinter(origfst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }

    assert(RandEquivalent(*fst, origfst, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));    
    delete fst;
  }
}


} // end namespace fst

int main() {
  using namespace fst;
  typedef LatticeWeightTpl<float> LatticeWeight;
  for(int i = 0; i < 2; i++) {
    bool invert = (i % 2);
    TestConvert<TropicalWeight, int32>(invert);
    TestConvertPair<LatticeWeight, int32>(invert);
    TestConvertPair<LatticeWeight, size_t>(invert);
    TestConvertPair<LexicographicWeight<TropicalWeight, TropicalWeight>, size_t>(invert);
  }
  std::cout << "Tests succeeded\n";
}
