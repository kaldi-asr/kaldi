// gmmbin/gmm-align.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2014 Johns Hopkins University (Author: Daniel Povey)

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
#include "fstext/fstext-utils.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/training-graph-compiler.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given [GMM-based] models.\n"
        "Usage:   gmm-align [options] tree-in model-in lexicon-fst-in feature-rspecifier "
        "transcriptions-rspecifier alignments-wspecifier\n"
        "e.g.: \n"
        " gmm-align tree 1.mdl lex.fst scp:train.scp "
        "'ark:sym2int.pl -f 2- words.txt text|' ark:1.ali\n";
    ParseOptions po(usage);
    AlignConfig align_config;
    BaseFloat acoustic_scale = 1.0;
    std::string disambig_rxfilename;
    TrainingGraphCompilerOptions gopts;

    align_config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");

    gopts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_in_filename = po.GetArg(1);
    std::string model_in_filename = po.GetArg(2);
    std::string lex_in_filename = po.GetArg(3);
    std::string feature_rspecifier = po.GetArg(4);
    std::string transcript_rspecifier = po.GetArg(5);
    std::string alignment_wspecifier = po.GetArg(6);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_in_filename, &ctx_dep);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    // ownership will be taken by gc.
    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_in_filename);

    std::vector<int32> disambig_syms;    
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;
    
    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms,
                             gopts);

    lex_fst = NULL;  // we gave ownership to gc.
    
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader transcript_reader(transcript_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);
    
    int32 num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!transcript_reader.HasKey(utt)) {
        KALDI_WARN << "No transcript found for utterance " << utt;
        num_err++;
        continue;
      }

      const Matrix<BaseFloat> &features = feature_reader.Value();
      const std::vector<int32> &transcript = transcript_reader.Value(utt);

      VectorFst<StdArc> decode_fst;
      if (!gc.CompileGraphFromText(transcript, &decode_fst)) {
        KALDI_WARN << "Problem creating decoding graph for utterance "
                   << utt <<" [serious error]";
        num_err++;
        continue;
      }
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length features for utterance: " << utt;
        num_err++;
        continue;
      }

      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);

      
      AlignUtteranceWrapper(align_config, utt,
                            acoustic_scale, &decode_fst, &gmm_decodable,
                            &alignment_writer, NULL,
                            &num_done, &num_err, &num_retry,
                            &tot_like, &frame_count);
    }
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count<< " frames.";
    KALDI_LOG << "Retried " << num_retry << " out of "
              << (num_done + num_err) << " utterances.";
    KALDI_LOG << "Done " << num_done << ", errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


