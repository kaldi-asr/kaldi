// fstext/kaldi-fst-io.cc

// Copyright 2009-2011  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)
//                2013  Guoguo Chen

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

#include "fstext/kaldi-fst-io.h"
#include "base/kaldi-error.h"
#include "base/kaldi-math.h"
#include "util/kaldi-io.h"

namespace fst {

VectorFst<StdArc> *ReadFstKaldi(std::string rxfilename) {
  if (rxfilename == "") rxfilename = "-"; // interpret "" as stdin,
  // for compatibility with OpenFst conventions.
  kaldi::Input ki(rxfilename);
  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), rxfilename))
    KALDI_ERR << "Reading FST: error reading FST header from "
              << kaldi::PrintableRxfilename(rxfilename);
  FstReadOptions ropts("<unspecified>", &hdr);
  VectorFst<StdArc> *fst = VectorFst<StdArc>::Read(ki.Stream(), ropts);
  if (!fst)
    KALDI_ERR << "Could not read fst from "
              << kaldi::PrintableRxfilename(rxfilename);
  return fst;
}

void ReadFstKaldi(std::string rxfilename, fst::StdVectorFst *ofst) {
  fst::StdVectorFst *fst = ReadFstKaldi(rxfilename);
  *ofst = *fst;
  delete fst;
}

void WriteFstKaldi(const VectorFst<StdArc> &fst,
                   std::string wxfilename) {
  if (wxfilename == "") wxfilename = "-"; // interpret "" as stdout,
  // for compatibility with OpenFst conventions.
  bool write_binary = true, write_header = false;
  kaldi::Output ko(wxfilename, write_binary, write_header);
  FstWriteOptions wopts(kaldi::PrintableWxfilename(wxfilename));
  fst.Write(ko.Stream(), wopts);
}

} // end namespace fst
