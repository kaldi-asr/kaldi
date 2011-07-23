// fstext/fstext-utils-test.cc

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

#include "fstext/fstext-utils.h"
#include "fstext/fst-test-utils.h"
#include "fstext/make-stochastic.h"
#include "util/stl-utils.h"


namespace fst
{
template<class Arc, class I>
void TestMakeLinearAcceptor() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  int len = rand() % 10;
  vector<I> vec;
  vector<I> vec_nozeros;
  for (int i = 0; i < len; i++) {
    int j = rand() % len;
    vec.push_back(j);
    if (j != 0) vec_nozeros.push_back(j);
  }


  VectorFst<Arc> vfst;
  MakeLinearAcceptor(vec, &vfst);
  vector<I> vec2;
  vector<I> vec3;
  Weight w;
  GetLinearSymbolSequence(vfst, &vec2, &vec3, &w);
  assert(w == Weight::One());
  assert(vec_nozeros == vec2);
  assert(vec_nozeros == vec3);

  bool include_eps = (rand() % 2 == 0);
  if (!include_eps) vec = vec_nozeros;
  kaldi::SortAndUniq(&vec);

  vector<I> vec4;
  GetInputSymbols(vfst, include_eps, &vec4);
  assert(vec4 == vec);
  vector<I> vec5;
  GetInputSymbols(vfst, include_eps, &vec5);
}


template<class Arc>  void TestDeterminizeStarInLog() {
  VectorFst<Arc> *fst = RandFst<Arc>();
  VectorFst<Arc> fst_copy(fst);
  typename Arc::Label next_sym = 1 + HighestNumberedInputSymbol(*fst);
  vector<typename Arc::Label> syms;
  PreDeterminize(fst, NULL, "#", next_sym, &syms);


}

  // Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc>  void TestSafeDeterminizeWrapper() {  // also tests SafeDeterminizeMinimizeWrapper().
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();
  int n_syms = 2 + rand() % 5, n_states = 3 + rand() % 10, n_arcs = 5 + rand() % 30, n_final = 1 + rand()%3;  // Up to 2 unique symbols.
  cout << "Testing pre-determinize with "<<n_syms<<" symbols, "<<n_states<<" states and "<<n_arcs<<" arcs and "<<n_final<<" final states.\n";
  SymbolTable *sptr = new SymbolTable("my-symbol-table");

  vector<Label> all_syms;  // including epsilon.
  // Put symbols in the symbol table from 1..n_syms-1.
  for (size_t i = 0;i < (size_t)n_syms;i++) {
    std::stringstream ss;
    if (i == 0) ss << "<eps>";
    else ss<<i;
    Label cur_lab = sptr->AddSymbol(ss.str());
    assert(cur_lab == (Label)i);
    all_syms.push_back(cur_lab);
  }
  assert(all_syms[0] == 0);

  // Create states.
  vector<StateId> all_states;
  for (size_t i = 0;i < (size_t)n_states;i++) {
    StateId this_state = fst->AddState();
    if (i == 0) fst->SetStart(i);
    all_states.push_back(this_state);
  }
  // Set final states.
  for (size_t j = 0;j < (size_t)n_final;j++) {
    StateId id = all_states[rand() % n_states];
    Weight weight = (Weight)(0.33*(rand() % 5) );
    printf("calling SetFinal with %d and %f\n", id, weight.Value());
    fst->SetFinal(id, weight);
  }
  // Create arcs.
  for (size_t i = 0;i < (size_t)n_arcs;i++) {
    Arc a;
    a.nextstate = all_states[rand() % n_states];
    a.ilabel = all_syms[rand() % n_syms];
    a.olabel = all_syms[rand() % n_syms];  // same input+output vocab.
    a.weight = (Weight) (0.33*(rand() % 2));
    StateId start_state = all_states[rand() % n_states];
    fst->AddArc(start_state, a);
  }

  std::cout <<" printing before trimming\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }
  // Trim resulting FST.
  Connect(fst);

  std::cout <<" printing after trimming\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  VectorFst<Arc> *fst_copy_orig = new VectorFst<Arc>(*fst);

  VectorFst<Arc> *fst_det = new VectorFst<Arc>;

  vector<Label> extra_syms;
  if (fst->Start() != kNoStateId) {  // "Connect" did not make it empty....
    if (rand() % 2 == 0)
      SafeDeterminizeWrapper(fst_copy_orig, fst_det);
    else {
      if (rand() % 2 == 0)
        SafeDeterminizeMinimizeWrapper(fst_copy_orig, fst_det);
      else
        SafeDeterminizeMinimizeWrapperInLog(fst_copy_orig, fst_det);
    }

    // no because does shortest-dist on weights even if not pushing on them.
    // PushInLog<REWEIGHT_TO_INITIAL>(fst_det, kPushLabels);  // will always succeed.
	KALDI_LOG << "Num states [orig]: " << fst->NumStates() << "[det]" << fst_det->NumStates();
    assert(RandEquivalent(*fst, *fst_det, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));
  }
  delete fst;
  delete fst_copy_orig;
  delete fst_det;
  delete sptr;
}


  // Don't instantiate with log semiring, as RandEquivalent may fail.
void TestPushInLog() {  // also tests SafeDeterminizeMinimizeWrapper().
  typedef StdArc Arc;
  typedef  Arc::Label Label;
  typedef  Arc::StateId StateId;
  typedef  Arc::Weight Weight;

  VectorFst<Arc> *fst = RandFst<Arc>();
  VectorFst<Arc> fst2(*fst);
  PushInLog<REWEIGHT_TO_INITIAL>(&fst2, kPushLabels|kPushWeights, 0.01);  // speed it up using large delta.
  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}



template<class Arc>  void TestAcceptorMinimize() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = RandFst<Arc>();

  Project(fst, PROJECT_INPUT);
  RemoveWeights(fst);

  VectorFst<Arc> fst2(*fst);
  AcceptorMinimize(&fst2);

  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}

template<class Arc>  void TestOptimize() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = RandFst<Arc>();


  OptimizeConfig cfg;
  cfg.delta = 0.0001;
  cfg.maintain_log_stochasticity = (rand() % 2 == 1);  // either should work.
  cfg.push_labels = (rand() % 2 == 1);

  VectorFst<Arc> fst2(*fst);
  Optimize(&fst2, cfg);

  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}


static void TestOptimizeStochastic() {
  // test that Optimize preserves equivalence in tropical while
  // maintaining stochasticity in log.
  VectorFst<LogArc> *logfst = RandFst<LogArc>();

  MakeStochasticOptions opts;
  vector<float> garbage;
  MakeStochasticFst(opts, logfst, &garbage, NULL);
#if !defined(_MSC_VER)
  assert(IsStochasticFst(*logfst, kDelta*10));
#endif
  {
    std::cout << "logfst = \n";
    FstPrinter<LogArc> fstprinter(*logfst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  VectorFst<StdArc> fst;
  Cast(*logfst, &fst);
  VectorFst<StdArc> fst_copy(fst);


  OptimizeConfig cfg;
  cfg.delta = kDelta;
  cfg.maintain_log_stochasticity = true;  // must be in log for this to work.
  cfg.push_labels = (rand() % 2 == 1);
  Optimize(&fst, cfg);
  // make sure equivalent.
  assert(RandEquivalent(fst, fst_copy, 5, 0.01, rand(), 100));
  VectorFst<LogArc> logfst2;
  Cast(fst, &logfst2);

  {
    std::cout << "logfst2 = \n";
    FstPrinter<LogArc> fstprinter(logfst2, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  LogWeight min, max;
  bool ans = IsStochasticFst(logfst2, kDelta*100, &min, &max);
  if (!ans) {
    std::cout << "TestOptimizeStochastic, not stochastic, min = "<<min.Value()<<", max = "<<max.Value()<<'\n';
    assert(ApproxEqual(LogWeight::One(), max, kDelta*100));
    // it can become sub-stochastic (all sums are <= One) due to combination of paths in minimization.
  }

  delete logfst;
}


static void TestOptimizeSpecial() {
  std::cout << "TestOptimizeSpecial: \n";
  typedef StdArc Arc;
  VectorFst<Arc> *fst = RandFst<Arc>();
  {
    std::cout << "fst = \n";
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }
  OptimizeConfig cfg;
  typedef StdArc::Label Label;
  KALDI_LOG <<  "Optimize: about to determinize.";
  {  // Determinize.
    VectorFst<StdArc> det_fst;
    if (cfg.maintain_log_stochasticity) SafeDeterminizeWrapperInLog(fst, &det_fst, cfg.delta);
    else SafeDeterminizeWrapper(fst, &det_fst, cfg.delta);
    *fst = det_fst;
  }
  {
    std::cout << "fst (post-determinize) = \n";
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  if (cfg.maintain_log_stochasticity) RemoveEpsLocalSpecial(fst);
  else RemoveEpsLocal(fst);

  {
    std::cout << "fst (post-remove-eps) = \n";
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  if (cfg.push_weights || cfg.push_labels) {  // need to do some kind of pushing...
    KALDI_LOG <<  "Optimize: about to push.";
    if (cfg.push_weights &&  cfg.push_in_log) {
      VectorFst<LogArc> *log_fst = new VectorFst<LogArc>;
      Cast(*fst, log_fst);
      VectorFst<LogArc> *log_fst_pushed = new VectorFst<LogArc>;

      Push<LogArc, REWEIGHT_TO_INITIAL>
           (*log_fst, log_fst_pushed,
            (cfg.push_weights?kPushWeights:0)|(cfg.push_labels?kPushLabels:0),
            cfg.delta);

      Cast(*log_fst_pushed, fst);
      delete log_fst;
      delete log_fst_pushed;

      {
        std::cout << "fst (post-push) = \n";
        FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
        fstprinter.Print(&std::cout, "standard output");
      }

    } else {
      VectorFst<StdArc> fst_pushed;
      Push<StdArc, REWEIGHT_TO_INITIAL>
          (*fst, &fst_pushed,
           (cfg.push_weights?kPushWeights:0)|(cfg.push_labels?kPushLabels:0),
           cfg.delta);
      *fst = fst_pushed;
    }
  }
  KALDI_LOG <<  "Optimize: about to minimize.";
  MinimizeEncoded(fst, cfg.delta);

  {
    std::cout << "fst (post-minimize) = \n";
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }

  // now do DFS-order.
  KALDI_LOG << "Optimize: about to sort arcs by weight.";
  WeightArcSort(fst);
  KALDI_LOG << "Optimize: about to dfs order.";
  VectorFst<StdArc> ordered;
  DfsReorder(*fst, &ordered);
  *fst = ordered;

  {
    std::cout << "fst (post-reorder) = \n";
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
    fstprinter.Print(&std::cout, "standard output");
  }
  delete fst;
}


template<class Arc>  void TestMakeSymbolsSame() {

  VectorFst<Arc> *fst = RandFst<Arc>();
  bool foll = (rand() % 2 == 0);
  bool is_symbol = (rand() % 2 == 0);


  VectorFst<Arc> fst2(*fst);

  if (foll) {
    MakeFollowingInputSymbolsSame(is_symbol, &fst2);
    assert(FollowingInputSymbolsAreSame(is_symbol, fst2));
  } else {
    MakePrecedingInputSymbolsSame(is_symbol, &fst2);
    assert(PrecedingInputSymbolsAreSame(is_symbol, fst2));
  }


  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}


template<class Arc>
struct TestFunctor {
  typedef int32 Result;
  typedef typename Arc::Label Arg;
  Result operator () (Arg a) const {
    if (a == kNoLabel) return -1;
    else if (a == 0) return 0;
    else {
      return 1 + ((a-1) % 10);
    }
  }
};

template<class Arc>  void TestMakeSymbolsSameClass() {

  VectorFst<Arc> *fst = RandFst<Arc>();
  bool foll = (rand() % 2 == 0);
  bool is_symbol = (rand() % 2 == 0);


  VectorFst<Arc> fst2(*fst);

  TestFunctor<Arc> f;
  if (foll) {
    MakeFollowingInputSymbolsSameClass(is_symbol, &fst2, f);
    assert(FollowingInputSymbolsAreSameClass(is_symbol, fst2, f));
  } else {
    MakePrecedingInputSymbolsSameClass(is_symbol, &fst2, f);
    assert(PrecedingInputSymbolsAreSameClass(is_symbol, fst2, f));
  }

  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}


// MakeLoopFstCompare is as MakeLoopFst but implmented differently [ less efficiently
// but more clearly], so we can check for equivalence.
template<class Arc>
VectorFst<Arc>* MakeLoopFstCompare(const vector<const ExpandedFst<Arc> *> &fsts) {
  VectorFst<Arc>* ans = new VectorFst<Arc>;
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  for (Label i = 0; i <  fsts.size(); i++) {
    if (fsts[i] != NULL) {
      VectorFst<Arc> i_fst;  // accepts symbol i on output.
      i_fst.AddState(); i_fst.AddState();
      i_fst.SetStart(0); i_fst.SetFinal(1, Weight::One());
      i_fst.AddArc(0, Arc(0, i, Weight::One(), 1));
      VectorFst<Arc> other_fst(*(fsts[i]));  // copy it.
      ClearSymbols(false, true, &other_fst);  // Clear output symbols so symbols
      // are on input side.
      Concat(&i_fst, other_fst);  // now i_fst is "i_fst [concat] other_fst".
      Union(ans, i_fst);
    }
  }
  Closure(ans, CLOSURE_STAR);
  return ans;
}


template<class Arc>  void TestMakeLoopFst() {

  int num_fsts = rand() % 10;
  vector<const ExpandedFst<Arc>* > fsts(num_fsts, (const ExpandedFst<Arc>*)NULL);
  for (int i = 0; i < num_fsts; i++) {
    if (rand() % 2 == 0) {  // put an fst there.
      VectorFst<Arc> *fst = RandFst<Arc>();
      Project(fst, PROJECT_INPUT);  // make input & output labels the same.
      fsts[i] = fst;
    }
  }

  VectorFst<Arc> *fst1 = MakeLoopFst(fsts),
      *fst2 = MakeLoopFstCompare(fsts);

  assert(RandEquivalent(*fst1, *fst2, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));
  delete fst1;
  delete fst2;
  for (int i = 0; i < num_fsts; i++)
    if (fsts[i] != NULL)
      delete fsts[i];
}


void TestDeterminizeSpecialCase() {
  VectorFst<StdArc> fst;
  typedef StdArc::Weight Weight;
  typedef StdArc Arc;
  fst.AddState();
  fst.SetStart(0);
  fst.AddState();
  fst.SetFinal(1, Weight::One());
  fst.AddState();
  fst.SetFinal(2, Weight::One());
  fst.AddArc(0, Arc(0, 0, Weight::One(), 1));
  fst.AddArc(0, Arc(0, 0, Weight::One(), 2));

  VectorFst<StdArc> fst1;
  SafeDeterminizeMinimizeWrapperInLog(&fst, &fst1, kDelta);

  VectorFst<StdArc> fst2(fst);
  OptimizeConfig cfg;
  Optimize(&fst2, cfg);

  assert(RandEquivalent(fst, fst2, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));
  assert(RandEquivalent(fst, fst1, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));
}


template<class Arc>
void TestEqualAlign() {
  for (size_t i = 0; i < 4; i++) {
    RandFstOptions opts;
    opts.allow_empty = false;
    VectorFst<Arc> *fst = RandFst<Arc>();
    int length = 10 + rand() % 20;

    VectorFst<Arc> fst_path;
    if (EqualAlign(*fst, length, rand(), &fst_path)) {
      std::cout << "EqualAlign succeeded\n";
      vector<int32> isymbol_seq, osymbol_seq;
      typename Arc::Weight weight;
      GetLinearSymbolSequence(fst_path, &isymbol_seq, &osymbol_seq, &weight);
      assert(isymbol_seq.size() == length);
      Invert(&fst_path);
      VectorFst<Arc> fst_composed;
      Compose(fst_path, *fst, &fst_composed);
      assert(fst_composed.Start() != kNoStateId);  // make sure nonempty.
    } else {
      std::cout << "EqualAlign did not generate alignment\n";
    }
    delete fst;
  }
}


template<class Arc> void Print(const Fst<Arc> &fst, std::string message) {
  std::cout << message << "\n";
  FstPrinter<Arc> fstprinter(fst, NULL, NULL, NULL, false, true);
  fstprinter.Print(&std::cout, "standard output");
}


template<class Arc>
void TestRemoveUselessArcs() {
  for (size_t i = 0; i < 4; i++) {
    RandFstOptions opts;
    opts.allow_empty = false;
    VectorFst<Arc> *fst = RandFst<Arc>();
    // Print(*fst, "[testremoveuselessarcs]:fst:");
    UniformArcSelector<Arc> selector;
    RandGenOptions<UniformArcSelector<Arc> > randgen_opts(selector);
    VectorFst<Arc> fst_path;
    RandGen(*fst, &fst_path, randgen_opts);
    Project(&fst_path, PROJECT_INPUT);
    // Print(fst_path, "[testremoveuselessarcs]:fstpath:");

    VectorFst<Arc> fst_nouseless(*fst);
    RemoveUselessArcs(&fst_nouseless);
    // Print(fst_nouseless, "[testremoveuselessarcs]:fst_nouseless:");

    VectorFst<Arc> orig_composed,
        nouseless_composed;
    Compose(fst_path, *fst, &orig_composed);
    Compose(fst_path, fst_nouseless, &nouseless_composed);

    // Print(orig_composed, "[testremoveuselessarcs]:orig_composed");
    // Print(nouseless_composed, "[testremoveuselessarcs]:nouseless_composed");

    VectorFst<Arc> orig_bestpath,
        nouseless_bestpath;
    ShortestPath(orig_composed, &orig_bestpath);
    ShortestPath(nouseless_composed, &nouseless_bestpath);
    // Print(orig_bestpath, "[testremoveuselessarcs]:orig_bestpath");
    // Print(nouseless_bestpath, "[testremoveuselessarcs]:nouseless_bestpath");

    typename Arc::Weight worig, wnouseless;
    GetLinearSymbolSequence<Arc, int>(orig_bestpath, NULL, NULL, &worig);
    GetLinearSymbolSequence<Arc, int>(nouseless_bestpath, NULL, NULL, &wnouseless);
    assert(ApproxEqual(worig, wnouseless, kDelta));

    // assert(RandEquivalent(orig_bestpath, nouseless_bestpath, 5/*paths*/, 0.01/*delta*/, rand()/*seed*/, 100/*path length-- max?*/));
    delete fst;
  }
}




} // end namespace fst


int main() {
  for (int i = 0; i < 5; i++) {
    fst::TestMakeLinearAcceptor<fst::StdArc, int>();  // this also tests GetLinearSymbolSequence, GetInputSymbols and GetOutputSymbols.
    fst::TestMakeLinearAcceptor<fst::StdArc, int32>();
    fst::TestMakeLinearAcceptor<fst::StdArc, uint32>();
    fst::TestSafeDeterminizeWrapper<fst::StdArc>();
    fst::TestAcceptorMinimize<fst::StdArc>();
    fst::TestOptimize<fst::StdArc>();
    fst::TestOptimizeStochastic();  // make sure Optimize() preserves stochasticity.
    fst::TestOptimizeSpecial();  // a special test to debug something weird with this function.
    // fst::TestPushInLog(): mostly succeeds but sometimes doesn't
    //  terminate, which is actually valid given the nature of the
    //  test...
    fst::TestMakeSymbolsSame<fst::StdArc>();
    fst::TestMakeSymbolsSame<fst::LogArc>();
    fst::TestMakeSymbolsSameClass<fst::StdArc>();
    fst::TestMakeSymbolsSameClass<fst::LogArc>();
    fst::TestMakeLoopFst<fst::StdArc>();
    fst::TestMakeLoopFst<fst::LogArc>();
    fst::TestDeterminizeSpecialCase();
    fst::TestEqualAlign<fst::StdArc>();
    fst::TestEqualAlign<fst::LogArc>();
    fst::TestRemoveUselessArcs<fst::StdArc>();
  }
}


