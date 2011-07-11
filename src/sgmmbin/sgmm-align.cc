// sgmmbin/sgmm-align.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/training-graph-compiler.h"
#include "decoder/decodable-am-sgmm.h"



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given [SGMM-based] models.\n"
        "Usage: sgmm-align [options] tree-in model-in lexicon-fst-in "
        "feature-rspecifier transcriptions-rspecifier alignments-wspecifier\n"
        "e.g.: sgmm-align 1.tree 1.mdl lex.fst scp:train.scp ark:train.tra ark:1.ali\n";
    ParseOptions po(usage);
    bool binary = false;
    BaseFloat beam = 200.0;
    BaseFloat retry_beam = 0.0;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat log_prune = 5.0;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    TrainingGraphCompilerOptions gopts;
    SgmmGselectConfig sgmm_opts;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("beam", &beam, "Decoding beam");
    po.Register("retry-beam", &retry_beam, "Decoding beam for second try at alignment");
    po.Register("log-prune", &log_prune, "Pruning beam used to reduce number of exp() evaluations.");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices (rspecifier)");
    gopts.Register(&po);
    sgmm_opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    FasterDecoderOptions decode_opts;
    decode_opts.beam = beam;  // Don't set the other options.

    std::string tree_in_filename = po.GetArg(1);
    std::string model_in_filename = po.GetArg(2);
    std::string lex_in_filename = po.GetArg(3);
    std::string feature_rspecifier = po.GetArg(4);
    std::string transcript_rspecifier = po.GetArg(5);
    std::string alignment_wspecifier = po.GetArg(6);

    ContextDependency ctx_dep;  // the tree.
    {
      bool binary;
      Input is(tree_in_filename, &binary);
      ctx_dep.Read(is.Stream(), binary);
    }

    TransitionModel trans_model;
    AmSgmm am_sgmm;
    {
      bool binary;
      Input is(model_in_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_sgmm.Read(is.Stream(), binary);
    }

    VectorFst<StdArc> *lex_fst = NULL;  // ownership will be taken by gc.
    {
      std::ifstream is(lex_in_filename.c_str());
      if (!is.good()) KALDI_EXIT << "Could not open lexicon FST " << (std::string)lex_in_filename;
      lex_fst =
          VectorFst<StdArc>::Read(is, fst::FstReadOptions((std::string)lex_in_filename));
      if (lex_fst == NULL)
        exit(1);
    }

    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, gopts);

    lex_fst = NULL;  // we gave ownership to gc.

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader transcript_reader(transcript_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader;
    if (!gselect_rspecifier.empty() && !gselect_reader.Open(gselect_rspecifier))
      KALDI_ERR << "Unable to open stream for gaussian-selection indices";
    RandomAccessBaseFloatVectorReader spkvecs_reader;
    if (!spkvecs_rspecifier.empty())
      if (!spkvecs_reader.Open(spkvecs_rspecifier))
        KALDI_ERR << "Cannot read speaker vectors.";

    RandomAccessTokenReader utt2spk_reader;
    if (!utt2spk_rspecifier.empty())
      if (!utt2spk_reader.Open(utt2spk_rspecifier))
        KALDI_ERR << "Could not open utt2spk map: " << utt2spk_rspecifier;

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int num_success = 0, num_no_transcript = 0, num_other_error = 0;
    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!transcript_reader.HasKey(utt)) num_no_transcript++;
      else {
        const Matrix<BaseFloat> &features = feature_reader.Value();
        const std::vector<int32> &transcript = transcript_reader.Value(utt);

        string utt_or_spk;
        if (utt2spk_rspecifier.empty())  utt_or_spk = utt;
        else {
          if (!utt2spk_reader.HasKey(utt)) {
            KALDI_WARN << "Utterance " << utt << " not present in utt2spk map; "
                       << "skipping this utterance.";
            num_other_error++;
            continue;
          } else {
            utt_or_spk = utt2spk_reader.Value(utt);
          }
        }

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt_or_spk)) {
            spk_vars.v_s = spkvecs_reader.Value(utt_or_spk);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt_or_spk;
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

        VectorFst<StdArc> decode_fst;
        if (!gc.CompileGraph(transcript, &decode_fst)) {
          KALDI_WARN << "Problem creating decoding graph for utterance " <<
              utt <<" [serious error]";
          num_other_error++;
          continue;
        }

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_other_error++;
          continue;
        }
        if (decode_fst.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty decoding graph for " << utt;
          num_other_error++;
          continue;
        }

        FasterDecoder decoder(decode_fst, decode_opts);
        DecodableAmSgmmScaled sgmm_decodable(sgmm_opts, am_sgmm, spk_vars, trans_model,
                                             features, *gselect, log_prune, acoustic_scale);
        decoder.Decode(&sgmm_decodable);

        VectorFst<StdArc> decoded;  // linear FST.
        bool ans = decoder.GetOutput(true,  // consider only final states.
                                     &decoded);
        if (!ans && retry_beam != 0.0) {
          KALDI_WARN << "Retrying utterance " << utt << " with beam " << retry_beam;
          decode_opts.beam = retry_beam;
          decoder.SetOptions(decode_opts);
          decoder.Decode(&sgmm_decodable);
          ans = decoder.GetOutput(true,  // consider only final states.
                                  &decoded);
          decode_opts.beam = beam;
          decoder.SetOptions(decode_opts);
        }
        if (ans) {
          std::vector<int32> alignment;
          std::vector<int32> words;
          StdArc::Weight weight;
          frame_count += features.NumRows();

          GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
          BaseFloat like = -weight.Value() / acoustic_scale;
          tot_like += like;
          assert(words == transcript);
          alignment_writer.Write(utt, alignment);
          num_success ++;
          KALDI_LOG << "Log-like per frame for this file is "
                    << (like / features.NumRows()) << " over "
                    << features.NumRows() << " frames.";
        } else {
          KALDI_WARN << "Did not successfully decode file " << utt << ", len = "
                     << (features.NumRows());
          num_other_error++;
        }
      }
    }
    KALDI_LOG << "Done " << num_success << ", could not find transcripts for "
              << num_no_transcript << ", other errors on " << num_other_error;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


