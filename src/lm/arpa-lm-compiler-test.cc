// lm/arpa-lm-compiler-test.cc

// Copyright 2009-2011 Gilles Boulianne
// Copyright 2016 Smart Action LLC (kkm)

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

#include <iostream>
#include <string>
#include <sstream>

#include "base/kaldi-error.h"
#include "base/kaldi-math.h"
#include "lm/arpa-lm-compiler.h"
#include "util/kaldi-io.h"

namespace kaldi {

// Predefine some symbol values, because any integer is as good than any other.
enum {
  kEps = 0,
  kDisambig,
  kBos,kEos,
};

// Random path coverage test parameters.
static const int kMaxSentenceLenght = 1000;
static const int kRandomPaths = 20;

// Creates an FST that generates any sequence of symbols taken from given
// symbol table. The FST is then associated with the symbol table.
static fst::StdVectorFst* CreateGenFst(const fst::SymbolTable* pst) {
  fst::StdVectorFst* genFst = new fst::StdVectorFst;
  genFst->SetInputSymbols(pst);
  genFst->SetOutputSymbols(pst);

  fst::StdArc::StateId initId  = genFst->AddState();
  fst::StdArc::StateId midId   = genFst->AddState();
  fst::StdArc::StateId finalId = genFst->AddState();
  genFst->SetStart(initId);
  genFst->SetFinal(finalId, fst::StdArc::Weight::One());
  genFst->AddArc(initId, fst::StdArc(kBos, kBos, 0, midId));
  genFst->AddArc(midId,  fst::StdArc(kEos, kEos, 0, finalId));
  // Add a loop for each symbol except epsilon, BOS and EOS.
  fst::SymbolTableIterator si(*pst);
  for (si.Reset(); !si.Done(); si.Next()) {
    if (si.Value() == kBos || si.Value() == kEos ||
        si.Value() == kEps || si.Value() == kDisambig)
      continue;
    genFst->AddArc(midId, fst::StdArc(si.Value(), si.Value(),
                                      fst::StdArc::Weight::One(), midId));
  }
  return genFst;
}

// Randomly generates num_paths paths with uniform distribution.
static fst::StdVectorFst* CreateRandPathFst(
    int num_paths, const fst::StdVectorFst* genFst) {
  typedef fst::UniformArcSelector<fst::StdArc> UniformSelector;

  const int num_trials = 50;
  UniformSelector uniform_sel;
  fst::RandGenOptions<UniformSelector> opts(uniform_sel,
                                            kMaxSentenceLenght,
                                            num_paths);
  for (int i = 0; i < num_trials; i++) {
    fst::StdVectorFst* tmpFst = new fst::StdVectorFst;
    RandGen(*genFst, tmpFst, opts);
    if (tmpFst->Properties(fst::kCoAccessible, true)) {
      return tmpFst;
    }
    // not good, try another
    delete tmpFst;
  }
  // couldn't generate it within allowed trials
  std::cerr << " Warning: couldn't generate complete paths within "
            << num_trials << " trials and " << kMaxSentenceLenght
            << " max length\n";
  return NULL;
}

// Tests if all paths generated from genFst are included in testFst.
static bool VerifyCoverage(const fst::StdVectorFst* testFst) {
  // Create an FST that generates any sequence of symbols taken from
  // the grammar output.
  fst::StdVectorFst* genFst = CreateGenFst(testFst->OutputSymbols());

  // Generate kRandomPaths random paths with uniform distribution.
  fst::StdVectorFst* pathFst = CreateRandPathFst(kRandomPaths, genFst);
  if (!pathFst) {
    delete genFst;
    return false;
  }

  // Compose paths with language model fst.
  fst::StdVectorFst outFst;
  Compose(*pathFst, *testFst, &outFst);

  // Composition result must have ntests arcs out of initial state
  int narcs = outFst.NumArcs(outFst.Start());
  bool success = narcs == kRandomPaths;

  delete genFst;
  delete pathFst;

  return success;
}

// Compile given ARPA file.
ArpaLmCompiler* Compile(const string &infile) {
  ArpaParseOptions options;
  fst::SymbolTable symbols;
  // Use spaces on special symbols, so we never read them by mistake.
  symbols.AddSymbol(" <eps>", kEps);
  symbols.AddSymbol(" #0", kDisambig);
  options.bos_symbol = symbols.AddSymbol("<s>", kBos);
  options.eos_symbol = symbols.AddSymbol("</s>", kEos);
  options.oov_handling = ArpaParseOptions::kAddToSymbols;

  // Tests in this form cannot be run with epsilon substitution, unless every
  // random path is also fitted with a #0-transducing self-loop.
  ArpaLmCompiler* lm_compiler =
      new ArpaLmCompiler(options,
                         0,  // No epslilon substitution.
                         &symbols);
  ReadKaldiObject(infile, lm_compiler);
  return lm_compiler;
}

// Compiles infile and runs num_paths random coverage tests on the compiled FST.
bool CoverageTest(const string &infile) {
  ArpaLmCompiler* lm_compiler = Compile(infile);

  // See if path generated in this FST are covered by the LM FST.
  bool success = VerifyCoverage(&lm_compiler->Fst());
  delete lm_compiler;

  std::cout << "CoverageTests on " << infile << ": "
            << kRandomPaths << " random paths - "
            << (success ? "PASSED" : "FAILED") << '\n';

  return success;
}

bool ScoringTest(const string &infile, const string& sentence, float expected) {
  ArpaLmCompiler* lm_compiler = Compile(infile);
  const fst::SymbolTable* symbols = lm_compiler->Fst().InputSymbols();

  // Create a sentence FST for scoring.
  fst::StdVectorFst sentFst;
  fst::StdArc::StateId state = sentFst.AddState();
  sentFst.SetStart(state);

  std::stringstream ss(sentence);
  string word;
  while (ss >> word) {
    int64 word_sym = symbols->Find(word);
    KALDI_ASSERT(word_sym != -1);
    fst::StdArc::StateId next_state  = sentFst.AddState();
    sentFst.AddArc(state, fst::StdArc(0, word_sym, 0, next_state));
    state = next_state;
  }
  sentFst.SetFinal(state, 0);
  sentFst.SetOutputSymbols(symbols);

  // Do the composition and extract final weight.
  fst::StdVectorFst composed;
  fst::Compose(sentFst, lm_compiler->Fst(), &composed);

  std::vector<fst::StdArc::Weight> shortest;
  fst::ShortestDistance(composed, &shortest, true);
  float actual = shortest[composed.Start()].Value();

  std::cout << "Scored " << sentence << " in " << infile
            << ": Expected=" << expected << " actual=" << actual << "\n";
  bool success = ApproxEqual(expected, actual);

  delete lm_compiler;

  return success;
}

}  // namespace kaldi


int main(int argc, char *argv[]) {
  bool ok = true;
  ok &= kaldi::CoverageTest("test_data/missing_backoffs.arpa");
  ok &= kaldi::CoverageTest("test_data/unused_backoffs.arpa");
  ok &= kaldi::CoverageTest("test_data/input.arpa");

  ok &= kaldi::ScoringTest("test_data/input.arpa", "<s> b b b a </s>", 59.2649);
  ok &= kaldi::ScoringTest("test_data/input.arpa", "<s> a b </s>", 4.36082);

  return ok ? 0 : 1;
}
