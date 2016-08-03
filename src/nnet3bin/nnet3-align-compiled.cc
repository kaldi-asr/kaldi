// nnet2bin/nnet-align-compiled.cc

// Copyright 2009-2012     Microsoft Corporation
//                         Johns Hopkins University (author: Daniel Povey)
//                2015     Vijayaditya Peddinti
//                2015-16  Vimal Manohar

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
#include "nnet3/nnet-am-decodable-simple.h"
#include "lat/kaldi-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given nnet3 neural net model\n"
        "Usage:   nnet3-align-compiled [options] <nnet-in> <graphs-rspecifier> "
        "<features-rspecifier> <alignments-wspecifier>\n"
        "e.g.: \n"
        " nnet3-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali\n"
        "or:\n"
        " compile-train-graphs tree 1.mdl lex.fst ark:train.tra b, ark:- | \\\n"
        "   nnet3-align-compiled 1.mdl ark:- scp:train.scp t, ark:1.ali\n";

    ParseOptions po(usage);
    AlignConfig align_config;
    NnetSimpleComputationOptions decodable_opts;
    std::string use_gpu = "yes";
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;

    std::string ivector_rspecifier,
        online_ivector_rspecifier,
        utt2spk_rspecifier;
    int32 online_ivector_period = 0;
    align_config.Register(&po);
    decodable_opts.Register(&po);
    
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale of self-loop versus non-self-loop "
                "log probs [relative to acoustics]");
    po.Register("ivectors", &ivector_rspecifier, "Rspecifier for "
                "iVectors as vectors (i.e. not estimated online); per utterance "
                "by default, or per speaker if you provide the --utt2spk option.");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier for "
                "iVectors estimated online, as matrices.  If you supply this,"
                " you must set the --online-ivector-period option.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of frames "
                "between iVectors in matrices supplied to the --online-ivectors "
                "option");
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
      AmNnetSimple am_nnet;
      {
        bool binary;
        Input ki(model_in_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_nnet.Read(ki.Stream(), binary);
      }

      RandomAccessBaseFloatMatrixReader online_ivector_reader(
          online_ivector_rspecifier);
      RandomAccessBaseFloatVectorReaderMapped ivector_reader(
          ivector_rspecifier, utt2spk_rspecifier);


      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      Int32VectorWriter alignment_writer(alignment_wspecifier);
      BaseFloatWriter scores_writer(scores_wspecifier);


      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "No features for utterance " << utt;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        const Matrix<BaseFloat> *online_ivectors = NULL;
        const Vector<BaseFloat> *ivector = NULL;
        if (!ivector_rspecifier.empty()) {
          if (!ivector_reader.HasKey(utt)) {
            KALDI_WARN << "No iVector available for utterance " << utt;
            num_err++;
            continue;
          } else {
            ivector = &ivector_reader.Value(utt);
          }
        }
        if (!online_ivector_rspecifier.empty()) {
          if (!online_ivector_reader.HasKey(utt)) {
            KALDI_WARN << "No online iVector available for utterance " << utt;
            num_err++;
            continue;
          } else {
            online_ivectors = &online_ivector_reader.Value(utt);
          }
        }

        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &decode_fst);
        }

        DecodableAmNnetSimple nnet_decodable(
            decodable_opts, trans_model, am_nnet,
            features, ivector, online_ivectors,
            online_ivector_period);

        AlignUtteranceWrapper(align_config, utt,
                              decodable_opts.acoustic_scale,
                              &decode_fst, &nnet_decodable,
                              &alignment_writer, &scores_writer,
                              &num_done, &num_err, &num_retry,
                              &tot_like, &frame_count);
      }
      KALDI_LOG << "Overall log-likelihood per frame is "
                << (tot_like/frame_count)
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


