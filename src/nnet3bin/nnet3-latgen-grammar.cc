// nnet3bin/nnet3-latgen-grammar.cc

// Copyright      2018   Johns Hopkins University (author: Daniel Povey)

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "decoder/grammar-fst.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  // note: making this program work with GPUs is as simple as initializing the
  // device, but it probably won't make a huge difference in speed for typical
  // setups.
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using nnet3 neural net model, and GrammarFst-based graph\n"
        "see kaldi-asr.org/doc/grammar.html for more context.\n"
        "\n"
        "Usage: nnet3-latgen-grammar [options] <nnet-in> <grammar-fst-in> <features-rspecifier>"
        " <lattice-wspecifier> [ <words-wspecifier> [<alignments-wspecifier>] ]\n";

    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    LatticeFasterDecoderConfig config;
    NnetSimpleComputationOptions decodable_opts;

    std::string word_syms_filename;
    std::string ivector_rspecifier,
        online_ivector_rspecifier,
        utt2spk_rspecifier;
    int32 online_ivector_period = 0;
    config.Register(&po);
    decodable_opts.Register(&po);
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");
    po.Register("ivectors", &ivector_rspecifier, "Rspecifier for "
                "iVectors as vectors (i.e. not estimated online); per utterance "
                "by default, or per speaker if you provide the --utt2spk option.");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for "
                "utt2spk option used to get ivectors per speaker");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier for "
                "iVectors estimated online, as matrices.  If you supply this,"
                " you must set the --online-ivector-period option.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of frames "
                "between iVectors in matrices supplied to the --online-ivectors "
                "option");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        grammar_fst_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      CollapseModel(CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped ivector_reader(
        ivector_rspecifier, utt2spk_rspecifier);

    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    // this compiler object allows caching of computations across
    // different utterances.
    CachingOptimizingCompiler compiler(am_nnet.GetNnet(),
                                       decodable_opts.optimize_config);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    fst::GrammarFst fst;
    ReadKaldiObject(grammar_fst_rxfilename, &fst);
    timer.Reset();

    {
      LatticeFasterDecoderTpl<fst::GrammarFst> decoder(fst, config);

      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        const Matrix<BaseFloat> &features (feature_reader.Value());
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          continue;
        }
        const Matrix<BaseFloat> *online_ivectors = NULL;
        const Vector<BaseFloat> *ivector = NULL;
        if (!ivector_rspecifier.empty()) {
          if (!ivector_reader.HasKey(utt)) {
            KALDI_WARN << "No iVector available for utterance " << utt;
            num_fail++;
            continue;
          } else {
            ivector = &ivector_reader.Value(utt);
          }
        }
        if (!online_ivector_rspecifier.empty()) {
          if (!online_ivector_reader.HasKey(utt)) {
            KALDI_WARN << "No online iVector available for utterance " << utt;
            num_fail++;
            continue;
          } else {
            online_ivectors = &online_ivector_reader.Value(utt);
          }
        }

        DecodableAmNnetSimple nnet_decodable(
            decodable_opts, trans_model, am_nnet,
            features, ivector, online_ivectors,
            online_ivector_period, &compiler);

        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, nnet_decodable, trans_model, word_syms, utt,
                decodable_opts.acoustic_scale, determinize, allow_partial,
                &alignment_writer, &words_writer, &compact_lattice_writer,
                &lattice_writer,
                &like)) {
          tot_like += like;
          frame_count += nnet_decodable.NumFramesReady();
          num_success++;
        } else num_fail++;
      }
    }

    kaldi::int64 input_frame_count =
        frame_count * decodable_opts.frame_subsampling_factor;

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed * 100.0 / input_frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over "
              << frame_count << " frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
