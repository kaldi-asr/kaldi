// latbin/lattice-word-align.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "lat/word-align-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "(note: from the s5 scripts onward, this is deprecated, see lattice-align-words)\n"
        "Create word-aligned lattices (in which the arcs correspond with\n"
        "word boundaries)\n"
        "Usage: lattice-word-align [options] <model> <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-word-align --silence-phones=1:2 --wbegin-phones=2:6:10:14 \\\n"
        "   --wend-phones=3:7:11:15 --winternal-phones=4:8:12:16 --wbegin-and-end-phones=5:9:13:17 \\\n"
        "   --silence-label=2 --partial-word-label=16342 \\\n"
        "   final.mdl ark:1.lats ark:aligned.lats\n";
      
    ParseOptions po(usage);
    bool output_error_lats = true;
    bool test = false;

    po.Register("output-error-lats", &output_error_lats, "If true, output aligned lattices "
                "even if there was an error (e.g. caused by forced-out lattice)");
    po.Register("test", &test, "If true, activate checks designed to test the code.");
    
    WordBoundaryInfoOpts opts;
    opts.Register(&po);
    

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        model_rxfilename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    TransitionModel tmodel;
    ReadKaldiObject(model_rxfilename, &tmodel);
    
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 
    WordBoundaryInfo info(opts);
    
    int32 n_ok = 0, n_err_write = 0, n_err_nowrite = 0; // Note: we may have some output even in
    // error cases.

    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      const CompactLattice &lat = compact_lattice_reader.Value();

      CompactLattice aligned_lat;
      bool ans = WordAlignLattice(lat, tmodel, info, -1, &aligned_lat);

      if (test && ans)
        TestWordAlignedLattice(lat, tmodel, info, aligned_lat);

      if (!ans) {
        if (!output_error_lats) {
          KALDI_WARN << "Lattice for " << key << " did not correctly word align;"
              " not outputting it since --output-error-lats=false.";
          n_err_nowrite++;
        } else {
          if (aligned_lat.Start() == fst::kNoStateId) {
            KALDI_WARN << "Lattice for " << key << " did not correctly word align;"
                " empty result, producing no output.";
            n_err_nowrite++;
          } else {
            KALDI_WARN << "Lattice for " << key << " did not correctly word align;"
                " outputting it anyway since --output-error-lats=true.";
            n_err_write++;
            compact_lattice_writer.Write(key, aligned_lat);
          }
        }
      } else {
        if (aligned_lat.Start() == fst::kNoStateId) {
          n_err_nowrite++;
          KALDI_WARN << "Empty aligned lattice for " << key;
        } else {
          n_ok++;
          KALDI_LOG << "Aligned lattice for " << key;
          compact_lattice_writer.Write(key, aligned_lat);
        }
      }
    }
    int32 n_done = n_ok + n_err_write + n_err_nowrite;
    KALDI_LOG << "Done " << n_done << " lattices: " << n_ok << " OK, "
              << n_err_write << " in error but written anyway, "
              << n_err_nowrite << " in error and not written.";
    return (n_ok != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
