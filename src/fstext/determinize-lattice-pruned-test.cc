// fstext/determinize-lattice-pruned-test.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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

#include "fstext/determinize-lattice-pruned.h"
#include "fstext/lattice-utils.h"
#include "fstext/fst-test-utils.h"

namespace fst {



// test that determinization proceeds correctly on general
// FSTs (not guaranteed determinzable, but we use the
// max-states option to stop it getting out of control).
template<class Arc> void TestDeterminizeLatticePruned() {
  typedef typename Arc::Weight Weight;
  typedef int32 Int;
  typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;
  
  for(int i = 0; i < 100; i++) {
    RandFstOptions opts;
    opts.n_states = 4;
    opts.n_arcs = 10;
    opts.n_final = 2;
    opts.allow_empty = false;
    opts.weight_multiplier = 0.5; // impt for the randomly generated weights
    // to be exactly representable in float,
    // or this test fails because numerical differences can cause symmetry in 
    // weights to be broken, which causes the wrong path to be chosen as far
    // as the string part is concerned.
    
    VectorFst<Arc> *fst = RandFst<Arc>();
    if (rand() % 2 == 0)
      TopSort(fst);

    ILabelCompare<Arc> ilabel_comp;
    if (rand() % 2 == 0)
      ArcSort(fst, ilabel_comp);
    
    std::cout << "FST before lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> det_fst;
    try {
      DeterminizeLatticePrunedOptions lat_opts;
      lat_opts.max_mem = ((rand() % 2 == 0) ? 100 : 1000);
      lat_opts.max_states = ((rand() % 2 == 0) ? -1 : 20);
      lat_opts.max_arcs = ((rand() % 2 == 0) ? -1 : 30);
      bool ans = DeterminizeLatticePruned<TropicalWeight, int32>(*fst, 10.0, &det_fst, lat_opts);

      std::cout << "FST after lattice-determinizing is:\n";
      {
        FstPrinter<Arc> fstprinter(det_fst, NULL, NULL, NULL, false, true);
        fstprinter.Print(&std::cout, "standard output");
      }
      assert(det_fst.Properties(kIDeterministic, true) & kIDeterministic);
      // OK, now determinize it a different way and check equivalence.
      // [note: it's not normal determinization, it's taking the best path
      // for any input-symbol sequence....


      VectorFst<Arc> pruned_fst;
      Prune(*fst, &pruned_fst, 10.0);
      
      VectorFst<CompactArc> compact_pruned_fst, compact_pruned_det_fst;
      ConvertLattice<Weight, Int>(pruned_fst, &compact_pruned_fst, false);
      std::cout << "Compact pruned FST is:\n";
      {
        FstPrinter<CompactArc> fstprinter(compact_pruned_fst, NULL, NULL, NULL, false, true);
        fstprinter.Print(&std::cout, "standard output");
      }
      ConvertLattice<Weight, Int>(det_fst, &compact_pruned_det_fst, false);
      
      std::cout << "Compact version of determinized FST is:\n";
      {
        FstPrinter<CompactArc> fstprinter(compact_pruned_det_fst, NULL, NULL, NULL, false, true);
        fstprinter.Print(&std::cout, "standard output");
      }

      if (ans)
        assert(RandEquivalent(compact_pruned_det_fst, compact_pruned_fst, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length, max*/));
    } catch (...) {
      std::cout << "Failed to lattice-determinize this FST (probably not determinizable)\n";
    }
    delete fst;
  }
}

// test that determinization proceeds without crash on acyclic FSTs
// (guaranteed determinizable in this sense).
template<class Arc> void TestDeterminizeLatticePruned2() {
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
    DeterminizeLatticePruned<TropicalWeight, int32>(*fst, 10.0, &ofst);
    std::cout << "FST after lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(ofst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }
    delete fst;
  }
}


} // end namespace fst

int main() {
  using namespace fst;
  TestDeterminizeLatticePruned<StdArc>();
  TestDeterminizeLatticePruned2<StdArc>();
  std::cout << "Tests succeeded\n";
}
