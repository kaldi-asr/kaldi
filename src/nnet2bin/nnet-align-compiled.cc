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
#include "decoder/decoder-wrappers.h"
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
        "Usage:   nnet-align-compiled [options] <model-in> <graphs-rspecifier> "
        "<feature-rspecifier> <alignments-wspecifier>\n"
        "e.g.: \n"
        " nnet-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali\n"
        "or:\n"
        " compile-train-graphs tree 1.mdl lex.fst 'ark:sym2int.pl -f 2- words.txt text|' \\\n"
        "   ark:- | nnet-align-compiled 1.mdl ark:- scp:train.scp t, ark:1.ali\n";

    ParseOptions po(usage);
    AlignConfig align_config;
    std::string use_gpu = "yes";
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    std::string per_frame_acwt_wspecifier;

    align_config.Register(&po);
    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale of self-loop versus non-self-loop "
                "log probs [relative to acoustics]");
    po.Register("write-per-frame-acoustic-loglikes", &per_frame_acwt_wspecifier,
                "Wspecifier for table of vectors containing the acoustic log-likelihoods "
                "per frame for each utterance. E.g. ark:foo/per_frame_logprobs.1.ark");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        alignment_wspecifier = po.GetArg(4),
        scores_wspecifier = po.GetOptArg(5);

    int num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
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
      BaseFloatVectorWriter per_frame_acwt_writer(per_frame_acwt_wspecifier);

      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "No features for utterance " << utt;
          num_err++;
          continue;
        }
        const CuMatrix<BaseFloat> &features = feature_reader.Value(utt);
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &decode_fst);
        }

        bool pad_input = true;
        DecodableAmNnet nnet_decodable(trans_model, am_nnet, features,
                                       pad_input, acoustic_scale);

        AlignUtteranceWrapper(align_config, utt,
                              acoustic_scale, &decode_fst, &nnet_decodable,
                              &alignment_writer, &scores_writer,
                              &num_done, &num_err, &num_retry,
                              &tot_like, &frame_count, &per_frame_acwt_writer);
      }
      KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
                << " over " << frame_count<< " frames.";
      KALDI_LOG << "Retried " << num_retry << " out of "
                << (num_done + num_err) << " utterances.";
      KALDI_LOG << "Done " << num_done << ", errors on " << num_err;
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
