// fstext/make-stochastic-test.cc

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

#include "fstext/make-stochastic.h"
#include "fstext/fstext-utils.h"
#include "fstext/fst-test-utils.h"
// Just check that it compiles, for now.

namespace fst
{

// Don't instantiate with Arc as the log semiring, as RandEquivalent may fail
// then.
template<class Arc, class ArcCast> void TestMakeStochastic(bool cast_is_log,  // type of ArcCast is with log semiring.
                                                          float delta) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  SymbolTable *sptr = NULL;
  VectorFst<Arc> *fst = RandFst<Arc>();

  std::cout <<" printing random FST\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  VectorFst<Arc> *fst_original = new VectorFst<Arc>(*fst);  // copy of first fst. [after trimming]

  // could do:
  //  VectorFst<ArcCast> *tmp_fst = new VectorFst<ArcCast>();
  // Cast(*fst, tmp_fst);
  // Do this in a more basic way, so the original fst gets modified
  // (otherwise the magic with ref-counting stops this from happening).

  VectorFst<ArcCast> *tmp_fst = reinterpret_cast<VectorFst<ArcCast> * > (fst);
  // do not delete tmp_fst!  It's the same memory as "fst".

#ifndef _MSC_VER
  bool is_s = IsStochasticFst(*tmp_fst, delta);
  std::cout << "Before MakeStochasticFst, IsStochastic returns " << (is_s ? "true":"false") <<'\n';
#endif
  MakeStochasticOptions opts;
  opts.delta = delta;
  vector<float> leftover_probs;
  int num_syms_added;
  MakeStochasticFst(opts, tmp_fst,
                    &leftover_probs,
                    &num_syms_added);

  // If this assert fails at some point in the future, just comment it out.
  // It may not be exactly required to be true, due to issues with making comparisons
  // at different precisions
#ifndef _MSC_VER
  assert(is_s == (num_syms_added == 0));
#endif
  std::cout<< "Num extra symbols is "<<num_syms_added;

  std::cout <<" printing FST after make-stochastic\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

#ifndef _MSC_VER
  assert(IsStochasticFst(*tmp_fst, delta));
#endif

  // delta_plus_extra is a vague measure of how much disagreement
  // in likelihood we can expect for paths through the FST, or sums
  // of paths.
  float delta_times_100 = (float)(100.0 * delta);

  if (fst->Start() != kNoStateId && cast_is_log) {  // "Connect" did not make it empty....
    // and this test is relevant..
    std::cout << "Checking weights sum to one\n";
    typename ArcCast::Weight sum = ShortestDistance(*tmp_fst, 0.00001),
        one = ArcCast::Weight::One();
    // this amount in ApproxEqual below is an approximate figure
    // for how much disagreement we can expect, not an exact formula.
    assert(ApproxEqual(sum, one, std::max(0.1F, delta_times_100)));
    // the 0.1F is because of the inaccuracy in the ShortestDistance
    // algorithm for the log semiring.
  }

  int num_syms_removed;
  ReverseMakeStochasticFst(opts, leftover_probs, fst, &num_syms_removed);

  std::cout << "replaced "<<num_syms_removed<<" symbols.\n";
  assert(num_syms_removed == num_syms_added);

  std::cout <<" printing FST after make-non-stochastic\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  // These two should be the same even though it's a different
  // semiring now.

  // NO-- not doing this test, since ShortestDistance can fail to terminate
  // in reasonable time.

  // Weight orig_sum = ShortestDistance(*fst_original, kDelta);
  // std::cout << "Shortest distance of original FST is: " << orig_sum << "\n";

  //  Weight new_sum = ShortestDistance(*fst, kDelta);
  // std::cout << "Shortest distance of new FST is: " << new_sum << "\n";

  // assert(ApproxEqual(orig_sum, new_sum, 0.1F));

  std::cout <<" Checking equivalent to original FST. If this does not terminate it may not be a fatal problem [since there is no guarantee that ShortestDistance will always terminate, in the presence of epsilon loops].  In that case you can ignore the issue\n";
  std::cout.flush();
  // giving rand() as a seed stops the random number generator from always being reset to
  // the same point each time, while maintaining determinism of the test.

  // set symbol tables to NULL before calling RandEquivalent or it
  // complains.

  assert(RandEquivalent(*(reinterpret_cast<VectorFst<ArcCast> * > (fst_original)), * (reinterpret_cast<VectorFst<ArcCast> * >(fst)), 5/*paths*/, delta_times_100/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));  // we use a big delta because the precision of MakeStochastic was just 2.  Technically this assert could fail.

  delete fst;
  delete fst_original;
  delete sptr;
}

} // end namespace fst


int main() {
  for (int i = 0;i < 3;i++) {
    float delta = (i%2 == 0 ? 1.0e-05 : 0.01);

    std::cout << "Testing with tropical sum\n";
    fst::TestMakeStochastic<fst::StdArc, fst::StdArc>(false, delta);
    std::cout << "Testing with tropical sum, log sum\n";
    fst::TestMakeStochastic<fst::StdArc, fst::LogArc>(true, delta);

    std::cout << "Testing with log sum, log sum\n";
    fst::TestMakeStochastic<fst::LogArc, fst::LogArc>(true, delta);

    std::cout << "Testing with log sum, tropical sum\n";
    fst::TestMakeStochastic<fst::LogArc, fst::StdArc>(false, delta);
  }
}


