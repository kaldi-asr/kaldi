// nnet3bin/nnet3-latgen-faster-compose.cc

// Copyright      2020   Brno University of Technology (author: Karel Vesely)
//           2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2014   Guoguo Chen

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
#include "base/timer.h"

#include <fst/compose.h>
#include <fst/rmepsilon.h>
#include <memory>


int main(int argc, char *argv[]) {
  // note: making this program work with GPUs is as simple as initializing the
  // device, but it probably won't make a huge difference in speed for typical
  // setups.  You should use nnet3-latgen-faster-batch if you want to use a GPU.
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using nnet3 neural net model, with on-the-fly composition HCLG o B.\n"
        "B is utterance-specific boosting graph, typically a single-state FST with\n"
        "all words from words.txt on self loop arcs (then composition is not prohibitevly slow).\n"
        "Some word-arcs will have score discounts as costs, to boost them in HMM beam-search.\n"
        "Or, by not including words in B, we can remove them from HCLG network.\n"
        "Usage: nnet3-latgen-faster-compose [options] <nnet-in> <fst-in> <boost-fsts-rspecifier> <features-rspecifier>"
        " <lattice-wspecifier> [ <words-wspecifier> [<alignments-wspecifier>] ]\n"
        "See also: nnet3-latgen-faster-parallel, nnet3-latgen-faster-batch\n";

    ParseOptions po(usage);

    Timer timer, timer_compose;
    double elapsed_compose = 0.0;

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

    std::string model_in_filename = po.GetArg(1),
        hclg_fst_rxfilename = po.GetArg(2),
        boosting_fst_rspecifier = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        lattice_wspecifier = po.GetArg(5),
        words_wspecifier = po.GetOptArg(6),
        alignment_wspecifier = po.GetOptArg(7);

    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
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

    std::unique_ptr<fst::SymbolTable> word_syms = nullptr;
    if (word_syms_filename != "") {
      word_syms.reset(fst::SymbolTable::ReadText(word_syms_filename));
      if (!word_syms)
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;
    }

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    // this compiler object allows caching of computations across
    // different utterances.
    CachingOptimizingCompiler compiler(am_nnet.GetNnet(),
                                       decodable_opts.optimize_config);

    KALDI_ASSERT(ClassifyRspecifier(hclg_fst_rxfilename, NULL, NULL) == kNoRspecifier);
    {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

      RandomAccessTableReader<fst::VectorFstHolder> boosting_fst_reader(boosting_fst_rspecifier);

      // 'hclg_fst' is a single FST.
      VectorFst<StdArc> hclg_fst;
      {
        auto hclg_fst_tmp = std::unique_ptr<Fst<StdArc>>(fst::ReadFstKaldiGeneric(hclg_fst_rxfilename));
        hclg_fst = VectorFst<StdArc>(*hclg_fst_tmp); // Fst -> VectorFst, as it has to be MutableFst...
        // 'hclg_fst_tmp' is deleted by 'going out of scope' ...
      }

      // make sure hclg is sorted on olabel
      if (hclg_fst.Properties(fst::kOLabelSorted, true) == 0) {
        fst::OLabelCompare<StdArc> olabel_comp;
        fst::ArcSort(&hclg_fst, olabel_comp);
      }

      timer.Reset();

      //// MAIN LOOP ////
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

        // get the boosting graph,
        VectorFst<StdArc> boosting_fst;
        if (!boosting_fst_reader.HasKey(utt)) {
          KALDI_WARN << "No boosting fst for utterance " << utt;
          num_fail++;
          continue;
        } else {
          boosting_fst = boosting_fst_reader.Value(utt); // copy,
        }

        timer_compose.Reset();

        // RmEpsilon saved 30% of composition runtime...
        // - Note: we are loading 2-state graphs with eps back-link to the initial state.
        if (boosting_fst.Properties(fst::kIEpsilons, true) != 0) {
          fst::RmEpsilon(&boosting_fst);
        }

        // make sure boosting graph is sorted on ilabel,
        if (boosting_fst.Properties(fst::kILabelSorted, true) == 0) {
          fst::ILabelCompare<StdArc> ilabel_comp;
          fst::ArcSort(&boosting_fst, ilabel_comp);
        }

        // run composition,
        VectorFst<StdArc> decode_fst;
        fst::Compose(hclg_fst, boosting_fst, &decode_fst);

        // check that composed graph is non-empty,
        if (decode_fst.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty 'decode_fst' HCLG for utterance "
                     << utt << " (bad boosting graph?)";
          num_fail++;
          continue;
        }

        elapsed_compose += timer_compose.Elapsed();

        DecodableAmNnetSimple nnet_decodable(
            decodable_opts, trans_model, am_nnet,
            features, ivector, online_ivectors,
            online_ivector_period, &compiler);

        // Note: decode_fst is VectorFst, not ConstFst.
        //
        //       OpenFst docs say that more specific iterators
        //       are faster than generic iterators. And in HCLG
        //       is usually loaded for decoding as ConstFst.
        //
        //       auto decode_fst_ = ConstFst<StdArc>(decode_fst);
        //
        //       In this way, I tried to cast VectorFst to ConstFst,
        //       but this made the decoding 20% slower.
        //
        LatticeFasterDecoder decoder(decode_fst, config);

        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, nnet_decodable, trans_model, word_syms.get(), utt,
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
    KALDI_LOG << "Composition time "<< elapsed_compose
              << "s (" << (elapsed_compose * 100.0 / elapsed) << "%)";
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over "
              << frame_count << " frames.";

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
