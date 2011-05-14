// fstext/compose-trim-test.cc

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


#include "fstext/compose-trim.h"
#include "fstext/fst-test-utils.h"

namespace fst
{
  // Don't instantiate with log semiring, as RandEquivalent may fail.
  template<class Arc>  void TestComposeTrim(int sort_type, bool connect, bool left) {
    typedef typename Arc::Label Label;
    typedef typename Arc::StateId StateId;
    typedef typename Arc::Weight Weight;



    VectorFst<Arc> *fst1 = RandFst<Arc>();

    VectorFst<Arc> *fst2 = RandFst<Arc>();

    ILabelCompare<Arc> ilabel_comp;
    OLabelCompare<Arc> olabel_comp;
    if (sort_type == 0)
      ArcSort(fst1, olabel_comp);
    else if (sort_type == 1)
      ArcSort(fst2, ilabel_comp);
    else {
      ArcSort(fst1, olabel_comp);
      ArcSort(fst2, ilabel_comp);
    }

    VectorFst<Arc> *fst_mod = new VectorFst<Arc>();


    std::cout << "Connect = "<< (connect?"True\n":"False\n");

    if (left)
      ComposeTrimLeft(*fst1, *fst2, connect, fst_mod);
    else
      ComposeTrimRight(*fst1, *fst2, connect, fst_mod);

    std::cout <<" Original FST\n";
    {
      FstPrinter<Arc> fstprinter(*fst1, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }

    std::cout <<" Compose-trimmed FST\n";
    {
      FstPrinter<Arc> fstprinter(*fst_mod, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
    }

    assert(fst_mod->Properties(kAccessible, true) == kAccessible);
    if (connect) {
      assert(fst_mod->Properties(kCoAccessible, true) == kCoAccessible);
    }

    VectorFst<Arc> *fst3 = new VectorFst<Arc>();
    VectorFst<Arc> *fst3_mod = new VectorFst<Arc>();

    Compose(*fst1, *fst2, fst3);

    if (left)
      Compose(*fst_mod, *fst2, fst3_mod);
    else
      Compose(*fst1, *fst_mod, fst3_mod);

    // use fewer paths as this test is slow (would normally be 5).  Also reduce max path length.
    assert(RandEquivalent(*fst3, *fst3_mod, 2/*paths*/, 0.01/*delta*/, rand()/*seed*/, 20/*path length-- max?*/));

    delete fst1;
    delete fst_mod;
    delete fst2;
    delete fst3;
    delete fst3_mod;
  }

} // namespace fst

int main() {
  for (int i = 0;i < 1;i++) {  // should do more (was once 20), but for now testing for memory stuff.
    for (int j = 0;j < 3;j++) {
      fst::TestComposeTrim<fst::StdArc>(j, true, true);
      fst::TestComposeTrim<fst::StdArc>(j, false, true);
      fst::TestComposeTrim<fst::StdArc>(j, true, false);
      fst::TestComposeTrim<fst::StdArc>(j, false, false);
    }
  }
}


