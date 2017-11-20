// latbin/lattice-align-phones.cc

// Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey)

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
#include "lat/kaldi-lattice.h"
#include "lat/phone-align-lattice.h"
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::StdArc;
    using kaldi::int32;

    const char *usage =
        "Convert lattices so that the arcs in the CompactLattice format correspond with\n"
        "phones.  The output symbols are still words, unless you specify --replace-output-symbols=true\n"
        "Usage: lattice-align-phones [options] <model> <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-align-phones final.mdl ark:1.lats ark:phone_aligned.lats\n"
        "See also: lattice-to-phone-lattice, lattice-align-words, lattice-align-words-lexicon\n"
        "Note: if you just want the phone alignment from a lattice, the easiest path is\n"
        " lattice-1best | nbest-to-linear [keeping only alignment] | ali-to-phones\n"
        "If you want the words and phones jointly (i.e. pronunciations of words, with word\n"
        "alignment), try\n"
        " lattice-1best | nbest-to-prons\n";
    
    ParseOptions po(usage);
    bool write_compact = true;
    bool output_if_error = true;

    po.Register("write-compact", &write_compact, "If true, write in normal (compact) form.");
    po.Register("output-error-lats", &output_if_error, "Output lattices that aligned "
                "with errors (e.g. due to force-out");
    
    PhoneAlignLatticeOptions opts;
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
    
    SequentialCompactLatticeReader clat_reader;
    CompactLatticeWriter clat_writer;
    SequentialLatticeReader lat_reader;
    LatticeWriter lat_writer;

    if (write_compact) {
      clat_reader.Open(lats_rspecifier);
      clat_writer.Open(lats_wspecifier);
    } else {
      lat_reader.Open(lats_rspecifier);
      lat_writer.Open(lats_wspecifier);
    }

    int32 num_done = 0, num_err = 0;
    
    for (; write_compact ? !clat_reader.Done() : !lat_reader.Done(); 
           write_compact ? clat_reader.Next() : lat_reader.Next()) {
      std::string key = write_compact ? clat_reader.Key() : lat_reader.Key();
      
      // Compute a map from each (t, tid) to (sum_of_acoustic_scores, count)
      unordered_map<std::pair<int32,int32>, std::pair<BaseFloat, int32>,
                                          PairHasher<int32> > acoustic_scores;

      CompactLattice aligned_clat;
      bool ok;
      if (write_compact) {
        const CompactLattice &clat = clat_reader.Value();

        ok = PhoneAlignLattice(clat, tmodel, opts, &aligned_clat);
      } else {
        const Lattice &lat = lat_reader.Value();
        ComputeAcousticScoresMap(lat, &acoustic_scores);

        CompactLattice clat;
        fst::ConvertLattice(lat, &clat);

        ok = PhoneAlignLattice(clat, tmodel, opts, &aligned_clat);
      }
      
      if (!ok) {
        num_err++;
        if (!output_if_error)
          KALDI_WARN << "Lattice for " << key << " did align correctly";
        else {
          if (aligned_clat.Start() != fst::kNoStateId) {
            KALDI_LOG << "Outputting partial lattice for " << key;
            TopSortCompactLatticeIfNeeded(&aligned_clat);

            if (write_compact) {
              clat_writer.Write(key, aligned_clat);
            } else {
              Lattice out_lat;
              fst::ConvertLattice(aligned_clat, &out_lat);

              // Replace each arc (t, tid) with the averaged acoustic score from
              // the computed map
              ReplaceAcousticScoresFromMap(acoustic_scores, &out_lat);
              lat_writer.Write(key, out_lat);
            }
          }
        }
      } else {
        if (aligned_clat.Start() == fst::kNoStateId) {
          num_err++;
          KALDI_WARN << "Lattice was empty for key " << key;
        } else {
          num_done++;
          KALDI_VLOG(2) << "Aligned lattice for " << key;
          TopSortCompactLatticeIfNeeded(&aligned_clat);

          if (write_compact) {
            clat_writer.Write(key, aligned_clat);
          } else {
            Lattice out_lat;
            fst::ConvertLattice(aligned_clat, &out_lat);

            // Replace each arc (t, tid) with the averaged acoustic score from
            // the computed map
            ReplaceAcousticScoresFromMap(acoustic_scores, &out_lat);
            lat_writer.Write(key, out_lat);
          }
        }
      }
    }
    KALDI_LOG << "Successfully aligned " << num_done << " lattices; "
              << num_err << " had errors.";
    return (num_done > num_err ? 0 : 1); // We changed the error condition slightly here,
    // if there are errors in the word-boundary phones we can get situations
    // where most lattices give an error.
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
