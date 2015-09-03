// lm/lm-lib-test.cc
//
// Copyright 2009-2011 Gilles Boulianne.
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

/// @addtogroup LanguageModel
/// @{

/**
 * @file lm-lib-test.cc
 * @brief Unit tests for language model code.
 */

#include <iostream>
#include <string>
#include <sstream>
#include "lm/kaldi-lm.h"

namespace kaldi {

// hard-coded symbols (for now)

#define startOfSentence "<s>"
#define endOfSentence   "</s>"
#define epsilon         "<eps>"
#define MAX_SENTENCE_LENGTH 1000

/// @brief Recursively prints all complete paths starting at s and their score.
static LangModelFst::LmWeight PrintCompletePath(fst::SymbolTable *pst,
                                                fst::StdVectorFst *pfst,
                                                fst::StdArc::StateId s,
                                                LangModelFst::LmWeight score) {
  fst::ArcIterator<fst::StdVectorFst> ai(*pfst, s);
  for (ai.Reset(); !ai.Done(); ai.Next()) {
    std::cout << pst->Find(ai.Value().ilabel) << " ";
    fst::StdArc::Weight w = score;             // initialize with current score
    // reset weight to 0 if we are going through the initial state again
    if (s == pfst->Start()) {
      w = fst::StdArc::Weight::One();
    }
    std::cout << " \tcurrent score " << w;
    w = fst::Times(w, ai.Value().weight);     // add in value from current arc
    std::cout << " added arc " << ai.Value().weight;
    fst::StdArc::Weight fw = pfst->Final(ai.Value().nextstate);
    if (fw != fst::StdArc::Weight::Zero()) {
      w = fst::Times(w, fw);   // add in destination state weight if final
      std::cout << " added state weight " << w << '\n';
    }
    std::cout << '\n';
    score = PrintCompletePath(pst, pfst, ai.Value().nextstate, w);
  }
  // test this after recursive call in case there are arcs out of a final state
  if (pfst->Final(s) == fst::StdArc::Weight::One()) {
    // we hit final state, stop there
    // std::cout << " total score: " << score << '\n';
  }
  return score;
}

/// @brief Recursively prints all complete paths starting from initial state.
static LangModelFst::LmWeight PrintCompletePaths(fst::SymbolTable *pst,
                                                 fst::StdVectorFst *pfst) {
  KALDI_ASSERT(pst);
  KALDI_ASSERT(pfst);
  KALDI_ASSERT(pfst->Start() >=0);
  return PrintCompletePath(pst, pfst, pfst->Start(),
                           fst::StdArc::Weight::One());
}

/// @brief Creates an FST that generates any sequence of symbols
/// taken from given symbol table.
/// This FST is then associated with given symbol table.
static fst::StdVectorFst* CreateGenFst(fst::SymbolTable *pst) {
  fst::StdArc::StateId initId, midId, finalId;
  fst::StdVectorFst *genFst = new fst::StdVectorFst;
  pst->AddSymbol(epsilon);                         // added if not there
  int64 boslab = pst->AddSymbol(startOfSentence);  // added if not there
  int64 eoslab = pst->AddSymbol(endOfSentence);    // added if not there
  genFst->SetInputSymbols(pst);
  genFst->SetOutputSymbols(pst);

  initId  = genFst->AddState();
  midId   = genFst->AddState();
  finalId = genFst->AddState();
  genFst->SetStart(initId);                        // initial state
  genFst->SetFinal(finalId, fst::StdArc::Weight::One());  // final state
  genFst->AddArc(initId, fst::StdArc(boslab, boslab, 0, midId));
  genFst->AddArc(midId,  fst::StdArc(eoslab, eoslab, 0, finalId));
  // add a loop for each symbol except epsilon, begin and end of sentence
  fst::SymbolTableIterator si(*pst);
  for (si.Reset(); !si.Done(); si.Next()) {
    if (si.Value() == boslab ||
        si.Value() == eoslab ||
        si.Value() == 0) continue;
    genFst->AddArc(midId, fst::StdArc(si.Value(), si.Value(), 0, midId));
  }
  return genFst;
}

/// @brief Randomly generates ntests paths with uniform distribution.
static fst::StdVectorFst* CreateRandPathFst(int n, fst::StdVectorFst *genFst) {
  typedef fst::UniformArcSelector<fst::StdArc> UniformSelector;

  int nTrials = 50;
  UniformSelector uniform_sel;
  fst::RandGenOptions<UniformSelector > opts(uniform_sel,
                                             MAX_SENTENCE_LENGTH, n);

  for (int i = 0; i < nTrials; i++) {
    fst::StdVectorFst *tmpFst = new fst::StdVectorFst;
    RandGen(*genFst, tmpFst, opts);
    if (tmpFst->Properties(fst::kCoAccessible, true)) {
      // std::cout << "Got valid random path after " << i << " tries" << '\n';
      return tmpFst;
    }
    // not good, try another
    delete tmpFst;
  }
  // couldn't generate it within allowed trials
  std::cerr << " Warning: couldn't generate complete paths within " << nTrials;
  std::cerr << " trials and " << MAX_SENTENCE_LENGTH << " max length" << '\n';
  return NULL;
}

/// @brief Tests if all paths generated from genFst are included in testFst.
static bool coverageTests(fst::StdVectorFst *genFst,
                          fst::StdVectorFst *testFst,
                          int ntests) {
  bool success = true;
#ifdef KALDI_PARANOID
  KALDI_ASSERT(genFst != NULL);
  KALDI_ASSERT(testFst != NULL);
#endif

  std::cout << "Generating " << ntests << " tests";
  std::cout.flush();

  // randomly generate ntests paths with uniform distribution
  fst::StdVectorFst *pathFst = CreateRandPathFst(ntests, genFst);
  if (!pathFst) return false;

  // compose paths with language model fst
  fst::StdVectorFst *outFst = new fst::StdVectorFst;
  // std::cout << "Path FST " << '\n';
  // printFirstCompletePath(pst, pathFst, pathFst->Start());

  Compose(*pathFst, *testFst, outFst);

  // Composition result must have ntests arcs out of initial state
  int narcs = outFst->NumArcs(outFst->Start());
  std::cout << ", composition has " << narcs << " arcs out of start state" << '\n';
  if (narcs !=  ntests) success = false;

  // std::cout << "Out  FST " << '\n';
  // printFirstCompletePath(pst, outFst, outFst->Start());

  delete pathFst;
  delete outFst;

  return success;
}

/// @brief Tests read and write methods.
bool TestLmTableReadWrite(int nTests,
                          const string &infile,
                          const string &outfile) {
  bool success = true;
  // reading test: create a language model FST from input file
  std::cout << "LangModelFst test: read file " << infile << '\n';
  LangModelFst lm;
  if (!lm.Read(infile, kArpaLm)) return false;

  // first create an FST that generates
  // any sequence of symbols taken from symbol table
  fst::StdVectorFst *genFst = CreateGenFst(lm.GetFst()->MutableInputSymbols());

  // see if path generated in this FST are covered by the LM FST
  std::cout << "For any sequence of symbols found in symbol table:" << '\n';
  if (coverageTests(genFst, lm.GetFst(), nTests)) {
    std::cout << "PASSED";
  } else {
    std::cout << "FAILED";
    success = false;
  }
  std::cout <<'\n';

  // writing test: write out FST, read it back in a new lm
  // reading doesn't provide symbol tables automatically ?
  std::cout << "LangModelFst test: write to " << outfile;
  std::cout << " and read it back" << '\n';
  // std::cout << "lm input symbol table:" << '\n';
  // lm.GetFst()->InputSymbols()->WriteText(std::cout);
  // std::cout << "lm output symbol table:" << '\n';
  // lm.GetFst()->OutputSymbols()->WriteText(std::cout);
  lm.Write(outfile);

  std::cout << "LangModelFst test: read from " << outfile << '\n';
  LangModelFst lm2;
  if (!lm2.Read(outfile, kFst)) return false;
  // std::cout << "lm2 output symbol table:" << '\n';
  // lm2.GetFst()->InputSymbols()->WriteText(std::cout);
  // std::cout << "lm2 output symbol table:" << '\n';
  // lm2.GetFst()->OutputSymbols()->WriteText(std::cout);

  // generate random sequences from the original LM
  // and see if they are covered by the FST that was just read
  std::cout << "For any complete path in original LM:" << '\n';
  if (coverageTests(lm.GetFst(), lm2.GetFst(), nTests)) {
    std::cout << "PASSED";
  } else {
    std::cout << "FAILED";
    success = false;
  }
  std::cout <<'\n';
  delete genFst;

  return success;
}

/// @brief Tests correctness of path weights.
bool TestLmTableEvalScore(const string &inpfile,
                          const string &intext,
                          const string &refScoreFile) {
  bool success = true;

  // read in reference score
  std::ifstream strm(refScoreFile.c_str(), std::ifstream::in);
  LangModelFst::LmWeight refScore;
  strm >> refScore;
  std::cout << "Reference score is " << refScore << '\n';

  std::cout << "LangModelFst test: score text strings with LM " << intext << '\n';
  // use original log base for testing
  LangModelFst lm;
  if (!lm.Read(inpfile, kArpaLm, NULL, false)) return false;

  std::cout << "LangModelFst test: read text strings " << intext << '\n';
  // here specify symbol table to be used so composition works
  LangModelFst txtString;
  if (!txtString.Read(intext, kTextString,
                      lm.GetFst()->MutableInputSymbols())) {
    return false;
  }

  // PrintCompletePaths(txtString.GetFst()->InputSymbols(), txtString.GetFst());
  // std::cout << "Fst string input symbol table:" << '\n';
  // txtString.GetFst()->OutputSymbols()->WriteText(std::cout);
  // std::cout << "Fst string output symbol table:" << '\n';
  // txtString.GetFst()->OutputSymbols()->WriteText(std::cout);

  // compose paths with language model fst
  fst::StdVectorFst composedFst;
  fst::ComposeFstOptions < fst::StdArc,
    fst::Matcher<fst::StdFst >,
    fst::MatchComposeFilter< fst::Matcher<fst::StdFst > > > copts;
  copts.gc_limit = 0;  // Cache only the last state for fastest copy.
  composedFst = fst::ComposeFst<fst::StdArc>(*txtString.GetFst(),
                                             *lm.GetFst(),
                                             copts);
  composedFst.Write("composed.fst");

  // find best path score
  fst::StdVectorFst *bestFst = new fst::StdVectorFst;
  fst::ShortestPath(composedFst, bestFst, 1);

  std::cout << "Best path has " << bestFst->NumStates() << " states" << '\n';
  LangModelFst::LmWeight testScore = PrintCompletePaths(
      bestFst->MutableInputSymbols(),
      bestFst);
  std::cout << "Complete path score is " << testScore << '\n';

  if (testScore.Value() <= refScore.Value()) {
    std::cout << "PASSED";
  } else {
    std::cout << "FAILED";
    success = false;
  }
  std::cout <<'\n';

  delete bestFst;

  unlink("composed.fst");

  return success;
}

}  // end namespace kaldi

int main(int argc, char *argv[]) {
  int ntests;
  bool success = true;
  std::string infile = "input.arpa";
  std::string outfile = "output.fst";

  // Note that for these tests to work, language models must be acceptors
  // (i.e. have same symbol table for input and output) since we
  // compose them with one another

  ntests = 20;
  std::cout << "Testing small arpa file with missing backoffs" << '\n';
  infile = "missing_backoffs.arpa";
  success &= kaldi::TestLmTableReadWrite(ntests, infile, outfile);

  std::cout << "Testing small arpa file with unused backoffs" << '\n';
  infile = "unused_backoffs.arpa";
  success &= kaldi::TestLmTableReadWrite(ntests, infile, outfile);

  std::cout << "Testing normal small arpa file" << '\n';
  infile = "input.arpa";
  success &= kaldi::TestLmTableReadWrite(ntests, infile, outfile);

  ntests = 2;
  // note that we use latest value of 'infile' as the tested language model
  for (int i = 1; i <= ntests; i++) {
    std::ostringstream intext("");
    std::ostringstream refscore("");
    // these inputN.txt sentences have been scored
    // by an external LM tool with results in inputN.score
    intext << "input" << i << ".txt";
    refscore << "input" << i << ".score";
    success &= kaldi::TestLmTableEvalScore(infile,
                                           intext.str(),
                                           refscore.str());
  }

  unlink("output.fst");

  exit(success ? 0 : 1);
}
/// @}

