// latbin/lattice-copy.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Copy lattices (e.g. useful for changing to text mode or changing\n"
        "format to standard from compact lattice.)\n"
        "Usage: lattice-copy [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-copy --write-compact=false ark:1.lats ark,t:text.lats\n"
        "See also: lattice-to-fst, and the script egs/wsj/s5/utils/convert_slf.pl\n";
    
    ParseOptions po(usage);
    bool write_compact = true;
    po.Register("write-compact", &write_compact, "If true, write in normal (compact) form.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);

    int32 n_done = 0;
    
    if (write_compact) {
      SequentialCompactLatticeReader lattice_reader(lats_rspecifier);
      CompactLatticeWriter lattice_writer(lats_wspecifier);
      for (; !lattice_reader.Done(); lattice_reader.Next(), n_done++)
        lattice_writer.Write(lattice_reader.Key(), lattice_reader.Value());
    } else {
      SequentialLatticeReader lattice_reader(lats_rspecifier);
      LatticeWriter lattice_writer(lats_wspecifier);
      for (; !lattice_reader.Done(); lattice_reader.Next(), n_done++)
        lattice_writer.Write(lattice_reader.Key(), lattice_reader.Value());
    }
    KALDI_LOG << "Done copying " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
