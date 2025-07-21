// latbin/lattice-to-smbr-post.cc

// Copyright 2009-2012  Chao Weng
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
#include "lat/lattice-functions.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;
    using namespace kaldi;
  
    const char *usage =
        "Do forward-backward and collect frame level posteriors for\n"
        "the state-level minimum Bayes Risk criterion (SMBR), which\n"
        "is like MPE with the criterion at a context-dependent state level.\n"
        "The output may be fed into gmm-acc-stats2 or similar to train the\n"
        "models discriminatively.  The posteriors may be positive or negative.\n"
        "Usage: lattice-to-smbr-post [options] <model> <num-posteriors-rspecifier>\n"
        " <lats-rspecifier> <posteriors-wspecifier> \n"
        "e.g.: lattice-to-smbr-post --acoustic-scale=0.1 1.mdl ark:num.post\n"
        " ark:1.lats ark:1.post\n";

    kaldi::BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;
    bool one_silence_class = false;
    std::string silence_phones_str;
    kaldi::ParseOptions po(usage);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    po.Register("one-silence-class", &one_silence_class, "If true, newer "
                 "behavior which will tend to reduce insertions.");
    po.Register("silence-phones", &silence_phones_str, "Colon-separated "
                "list of integer id's of silence phones, e.g. 46:47");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
 
    std::vector<int32> silence_phones;
    if (!kaldi::SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    kaldi::SortAndUniq(&silence_phones);
    if (silence_phones.empty())
      KALDI_WARN << "No silence phones specified, make sure this is what you intended.";

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";

    std::string model_filename = po.GetArg(1), 
        alignments_rspecifier = po.GetArg(2),  
        lats_rspecifier = po.GetArg(3),
        posteriors_wspecifier = po.GetArg(4);

    SequentialLatticeReader lattice_reader(lats_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);
    if (alignments_rspecifier.find("ali-to-post") != std::string::npos) {
      KALDI_WARN << "Warning, this program has been changed to read alignments "
                 << "not posteriors.  Remove ali-to-post from your scripts.";
    }
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);
    
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
    }

    int32 num_done = 0, num_err = 0;
    double total_lat_frame_acc = 0.0, lat_frame_acc;
    double total_time = 0, lat_time;


    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      kaldi::Lattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();
      if (acoustic_scale != 1.0 || lm_scale != 1.0)
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &lat);
      
      kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&lat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      }
      
      if (!alignments_reader.HasKey(key)) {
        KALDI_WARN << "No alignment for utterance " << key;
        num_err++;
      } else {
        const std::vector<int32> &alignment = alignments_reader.Value(key);
        Posterior post;
        lat_frame_acc = LatticeForwardBackwardMpeVariants(
            trans_model, silence_phones, lat, alignment,
            "smbr", one_silence_class, &post);
        total_lat_frame_acc += lat_frame_acc;
        lat_time = post.size();
        total_time += lat_time;
        KALDI_VLOG(2) << "Processed lattice for utterance: " << key << "; found "
                      << lat.NumStates() << " states and " << fst::NumArcs(lat)
                      << " arcs. Average frame accuracies = " << (lat_frame_acc/lat_time)
                      << " over " << lat_time << " frames.";
        posterior_writer.Write(key, post);
        num_done++; 
      }
    }

    KALDI_LOG << "Overall average frame-accuracy is "
              << (total_lat_frame_acc/total_time) << " over " << total_time
              << " frames.";
    KALDI_LOG << "Done " << num_done << " lattices.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
