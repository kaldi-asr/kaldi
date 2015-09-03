// lm/kaldi-lm.cc
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
/**
 * @file kaldi-lm.cc
 * @brief Language model FST implementation.
 *
 * See kaldi-lm.h for more details.
 *
 */

#include "lm/kaldi-lm.h"
#include <stdexcept>

namespace kaldi {
// add the string contained in inpline to the current transducer
// starting at initial state
LangModelFst::StateId LangModelFst::ReadTxtLine(const string &inpline) {
  KALDI_ASSERT(pfst_);
  KALDI_ASSERT(pfst_->InputSymbols());
  KALDI_ASSERT(pfst_->OutputSymbols());

  StateId dst = pfst_->Start(), src = pfst_->Start();
  // this will split on white spaces only
  string curwrd;  // Have a buffer string
  std::stringstream ss(inpline);  // Insert the string into a stream
  while (ss >> curwrd) {
    // add labels to symbol tables
    int64 ilab = pfst_->MutableInputSymbols()->AddSymbol(curwrd);
    int64 olab = pfst_->MutableOutputSymbols()->AddSymbol(curwrd);
    dst = pfst_->AddState();
    pfst_->AddArc(src, fst::StdArc(ilab, olab, 0, dst));
    // cerr << "  adding word " << curwrd << " from state " << src;
    // cerr << " to state " << dst <<endl;
    src = dst;
  }
  return dst;
}

// create a path in the FST for each line of the input stream
// fst must already be provided with symbol tables
void LangModelFst::ReadTxtString(std::istream &strm) {
  string inpline;
  StateId src, final = pfst_->AddState();

  while (getline(strm, inpline) && !strm.eof()) {
    // cerr << "ReadTxtString: read line " << inpline << endl;
    src = ReadTxtLine(inpline);
    // add arc from last state produced to final state
    pfst_->AddArc(src, fst::StdArc(0, 0, 0, final));
  }
  pfst_->SetFinal(final, fst::StdArc::Weight::One());
}

// allocate an FST and provide symbol tables if not provided through pst
// we allocate the FST here to parallel OpenFst Read()
// although this is questionable
fst::StdVectorFst* LangModelFst::ReadStream(
                                            std::istream &strm,
                                            const string &sourcename,
                                            GrammarType gtype,
                                            fst::SymbolTable *pst,
                                            bool useNaturalLog,
                                            const string startSent,
                                            const string endSent) {
  if (gtype == kArpaLm || gtype == kTextString) {
    // always allocate local symbol table so we know we always have to delete it
    fst::SymbolTable *psyms = new fst::SymbolTable("lmInputSymbols");

    // initialize FST and reserve initial state
    // (we can retrieve it through Start())
    pfst_ = new fst::StdVectorFst;
    pfst_->SetStart(pfst_->AddState());

    // these will be added if not already there
    pst = pst ? pst : psyms;
    pst->AddSymbol("<eps>");
    pst->AddSymbol(startSent);
    pst->AddSymbol(endSent);

    // this creates reference-counted copies managed by fst
    pfst_->SetInputSymbols(pst);
    pfst_->SetOutputSymbols(pst);

    // so local objects are not needed anymore
    delete psyms;

    if (gtype == kArpaLm) {
      LmTable lmt;
      lmt.ReadFstFromLmFile(strm, pfst_, useNaturalLog, startSent, endSent);
    } else if (gtype== kTextString) {
      ReadTxtString(strm);
    }

  } else if (gtype == kFst) {
    // this is going to be reference-counted
    pfst_ = fst::StdVectorFst::Read(sourcename);

  } else {
    KALDI_ERR << "LangModelFst: unsupported grammar type";
  }
  return pfst_;
}

}  // end namespace kaldi


