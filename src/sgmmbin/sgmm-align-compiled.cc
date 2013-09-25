// sgmmbin/sgmm-align-compiled.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/training-graph-compiler.h"
#include "sgmm/decodable-am-sgmm.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given [SGMM-based] models.\n"
        "Usage: sgmm-align-compiled [options] model-in graphs-rspecifier "
        "feature-rspecifier alignments-wspecifier\n"
        "e.g.: sgmm-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali\n";

    ParseOptions po(usage);
    bool binary = true;
    BaseFloat beam = 200.0;
    BaseFloat retry_beam = 0.0;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    BaseFloat log_prune = 5.0;
    
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    SgmmGselectConfig sgmm_opts;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("beam", &beam, "Decoding beam");
    po.Register("retry-beam", &retry_beam, "Decoding beam for second try "
                "at alignment");
    po.Register("log-prune", &log_prune, "Pruning beam used to reduce number "
                "of exp() evaluations.");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic "
                "likelihoods");
    po.Register("transition-scale", &transition_scale, "Scaling factor for "
                "some transition probabilities [see also self-loop-scale].");
    po.Register("self-loop-scale", &self_loop_scale, "Scaling factor for "
                "self-loop versus non-self-loop probability mass [controls "
                "most transition probabilities.]");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices "
                "(rspecifier)");
    sgmm_opts.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    if (retry_beam != 0 && retry_beam <= beam)
      KALDI_WARN << "Beams do not make sense: beam " << beam
                 << ", retry-beam " << retry_beam;

    FasterDecoderOptions decode_opts;
    decode_opts.beam = beam;  // Don't set the other options.

    std::string model_in_filename = po.GetArg(1);
    std::string fst_rspecifier = po.GetArg(2);
    std::string feature_rspecifier = po.GetArg(3);
    std::string alignment_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmSgmm am_sgmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);

    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int num_success = 0, num_no_feat = 0, num_other_error = 0;
    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string utt = fst_reader.Key();
      if (!feature_reader.HasKey(utt)) num_no_feat++;
      else {
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        // stops copy-on-write of the fst by deleting the fst inside the reader,
        // since we're about to mutate the fst by adding transition probs.
        fst_reader.FreeCurrent();

        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_other_error++;
          continue;
        }

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.v_s = spkvecs_reader.Value(utt);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
            num_other_error++;
            continue;
          }
        }  // else spk_vars is "empty"

        bool have_gselect  = !gselect_rspecifier.empty()
            && gselect_reader.HasKey(utt)
            && gselect_reader.Value(utt).size() == features.NumRows();
        if (!gselect_rspecifier.empty() && !have_gselect)
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
        std::vector<std::vector<int32> > empty_gselect;
        const std::vector<std::vector<int32> > *gselect =
            (have_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

        if (decode_fst.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty decoding graph for " << utt;
          num_other_error++;
          continue;
        }
        
        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &decode_fst);
        }

        FasterDecoder decoder(decode_fst, decode_opts);

        DecodableAmSgmmScaled sgmm_decodable(sgmm_opts, am_sgmm, spk_vars, trans_model,
                                             features, *gselect, log_prune, acoustic_scale);

        decoder.Decode(&sgmm_decodable);
        
        VectorFst<LatticeArc> decoded;  // linear FST.
        bool ans = decoder.ReachedFinal() // consider only final states.
            && decoder.GetBestPath(&decoded);  
        if (!ans && retry_beam != 0.0) {
          KALDI_WARN << "Retrying utterance " << utt << " with beam "
                     << retry_beam;
          decode_opts.beam = retry_beam;
          decoder.SetOptions(decode_opts);
          decoder.Decode(&sgmm_decodable);
          ans = decoder.ReachedFinal() // consider only final states.
              && decoder.GetBestPath(&decoded);
          decode_opts.beam = beam;
          decoder.SetOptions(decode_opts);
        }
        if (ans) {
          std::vector<int32> alignment;
          std::vector<int32> words;
          LatticeWeight weight;
          frame_count += features.NumRows();

          GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
          BaseFloat like = (-weight.Value1() -weight.Value2()) / acoustic_scale;
          tot_like += like;
          alignment_writer.Write(utt, alignment);
          num_success ++;
          if (num_success % 50  == 0) {
            KALDI_LOG << "Processed " << num_success << " utterances, "
                      << "log-like per frame for " << utt << " is "
                      << (like / features.NumRows()) << " over "
                      << features.NumRows() << " frames.";
          }
        } else {
          KALDI_WARN << "Did not successfully decode file " << utt << ", len = "
                     << (features.NumRows());
          num_other_error++;
        }
      }
    }

    KALDI_LOG << "Done " << num_success << ", could not find features for "
              << num_no_feat << ", other errors on " << num_other_error;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


