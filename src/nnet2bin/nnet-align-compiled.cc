// nnet2bin/nnet-align-compiled.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/training-graph-compiler.h"
#include "nnet2/decodable-am-nnet.h"
#include "lat/kaldi-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given neural-net-based model\n"
        "Usage:   nnet-align-compiled [options] model-in graphs-rspecifier feature-rspecifier alignments-wspecifier\n"
        "e.g.: \n"
        " nnet-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali\n"
        "or:\n"
        " compile-train-graphs tree 1.mdl lex.fst ark:train.tra b, ark:- | \\\n"
        "   nnet-align-compiled 1.mdl ark:- scp:train.scp t, ark:1.ali\n";

    ParseOptions po(usage);
    std::string use_gpu = "yes";
    BaseFloat beam = 200.0;
    BaseFloat retry_beam = 0.0;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;

    po.Register("beam", &beam, "Decoding beam");
    po.Register("retry-beam", &retry_beam,
                "Decoding beam for second try at alignment");
    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale of self-loop versus non-self-loop "
                "log probs [relative to acoustics]");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional, only has effect if compiled with CUDA");     
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }
    if (retry_beam != 0 && retry_beam <= beam)
      KALDI_WARN << "Beams do not make sense: beam " << beam
                 << ", retry-beam " << retry_beam;
    
    FasterDecoderOptions decode_opts;
    decode_opts.beam = beam;  // Don't set the other options.

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    
    std::string model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        alignment_wspecifier = po.GetArg(4),
        scores_wspecifier = po.GetOptArg(5);


    int num_success = 0, num_no_feat = 0, num_other_error = 0;
    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    {
      TransitionModel trans_model;
      AmNnet am_nnet;
      {
        bool binary;
        Input ki(model_in_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_nnet.Read(ki.Stream(), binary);
      }

      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
      RandomAccessBaseFloatCuMatrixReader feature_reader(feature_rspecifier);
      Int32VectorWriter alignment_writer(alignment_wspecifier);
      BaseFloatWriter scores_writer(scores_wspecifier);

      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string key = fst_reader.Key();
        if (!feature_reader.HasKey(key)) {
          num_no_feat++;
          KALDI_WARN << "No features for utterance " << key;
        } else {
          const CuMatrix<BaseFloat> &features = feature_reader.Value(key);
          VectorFst<StdArc> decode_fst(fst_reader.Value());
          fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
          // by deleting the fst inside the reader, since we're about to mutate
          // the fst by adding transition probs.

          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << key;
            num_other_error++;
            continue;
          }
          if (decode_fst.Start() == fst::kNoStateId) {
            KALDI_WARN << "Empty decoding graph for " << key;
            num_other_error++;
            continue;
          }

          {  // Add transition-probs to the FST.
            std::vector<int32> disambig_syms;  // empty.
            AddTransitionProbs(trans_model, disambig_syms,
                               transition_scale, self_loop_scale,
                               &decode_fst);
          }

          // SimpleDecoder decoder(decode_fst, beam);
          FasterDecoder decoder(decode_fst, decode_opts);
          // makes it a bit faster: 37 sec -> 26 sec on 1000 RM utterances @ beam 200.

          CuVector<BaseFloat> empty_spk_info; // TODO: add support for speaker vectors.
          bool pad_input = true;
          DecodableAmNnet nnet_decodable(trans_model, am_nnet, features, empty_spk_info,
                                         pad_input, acoustic_scale);
          decoder.Decode(&nnet_decodable);

          VectorFst<LatticeArc> decoded;  // linear FST.
          bool ans = decoder.ReachedFinal() // consider only final states.
              && decoder.GetBestPath(&decoded);  
          if (!ans && retry_beam != 0.0) {
            KALDI_WARN << "Retrying utterance " << key << " with beam " << retry_beam;
            decode_opts.beam = retry_beam;
            decoder.SetOptions(decode_opts);
            decoder.Decode(&nnet_decodable);
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
            BaseFloat like = -(weight.Value1()+weight.Value2()) / acoustic_scale;
            tot_like += like;
            if (scores_writer.IsOpen())
              scores_writer.Write(key, -(weight.Value1()+weight.Value2()));
            alignment_writer.Write(key, alignment);
            num_success ++;
            KALDI_VLOG(2) << "Log-like per frame for utterance " << key 
                          << " is " << (like / features.NumRows()) << " over "
                          << features.NumRows() << " frames.";
            if (num_success % 50  == 0) {
              KALDI_LOG << "Processed " << num_success << " utterances, "
                        << "log-like per frame for " << key << " is "
                        << (like / features.NumRows()) << " over "
                        << features.NumRows() << " frames.";
            }
          } else {
            KALDI_WARN << "Did not successfully decode file " << key << ", len = "
                       << (features.NumRows());
            num_other_error++;
          }
        }
      }
      KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
                << " over " << frame_count<< " frames.";
      KALDI_LOG << "Done " << num_success << ", could not find features for "
                << num_no_feat << ", other errors on " << num_other_error;
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


