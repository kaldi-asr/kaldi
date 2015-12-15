// gmmbin/gmm-latgen-faster.cc

// Copyright 2009-2012  Hainan Xu


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "tree/build-tree-virtual.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/lattice-faster-decoder.h"
#include "gmm/decodable-am-diag-gmm-multi.h"
#include "util/timer.h"
#include "feat/feature-functions.h"  // feature reversal
#include <vector>

using std::vector;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using GMM-based model.\n"
        "Usage: gmm-latgen-faster-multi [options] model-in tree-file"
        " mapping_file (fst-in|fsts-rspecifier) features-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;
    int num_trees = 0;
    BaseFloat exp_weight = 0.1;
    
    std::string word_syms_filename;
    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");
    po.Register("num-trees", &num_trees,  // there shouldn't be any problem
                "Number of trees.");
    po.Register("exp-weight", &exp_weight,  // there shouldn't be any problem
                "Weight constant for the exp-weighted average computation");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 6 || po.NumArgs() > 8) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        tree_file = po.GetOptArg(2),
        mapping_file = po.GetArg(3),
        fst_in_str = po.GetArg(4),
        feature_rspecifier = po.GetArg(5),
        lattice_wspecifier = po.GetArg(6),
        words_wspecifier = po.GetOptArg(7),
        alignment_wspecifier = po.GetOptArg(8);
    
    vector<TransitionModel> trans_models(num_trees);

    vector<AmDiagGmm> am_gmms(num_trees);

    for (size_t i = 0; i < num_trees; i++)
    {
      char temp[4];
      sprintf(temp, "-%d", (int)i);
      std::string file_affix(temp);
      bool binary;
      Input ki(model_in_filename + file_affix, &binary);
      trans_models[i].Read(ki.Stream(), binary);
      am_gmms[i].Read(ki.Stream(), binary);
    }

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_file, &ctx_dep);

    unordered_map<int32, vector<int32> > mapping;
    {
      bool binary;
      Input input(mapping_file, &binary);
      ReadMultiTreeMapping(mapping, input.Stream(), binary);
    }

    TransitionModel trans_model(ctx_dep, mapping, trans_models);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_done = 0, num_err = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      VectorFst<StdArc> *decode_fst = fst::ReadFstKaldi(fst_in_str);
      
      {
        LatticeFasterDecoder decoder(*decode_fst, config);
    
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          Matrix<BaseFloat> features (feature_reader.Value());
          feature_reader.FreeCurrent();
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_err++;
            continue;
          }

          DecodableAmDiagGmmScaledMulti gmm_decodable(am_gmms, mapping,
                        trans_model, features, 
                        acoustic_scale, exp_weight);

          double like;
          if (DecodeUtteranceLatticeFaster(
                decoder, gmm_decodable, trans_model, word_syms, utt,
                acoustic_scale, determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &like)) {
            tot_like += like;
            frame_count += features.NumRows();
            num_done++;
          } else num_err++;
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);          
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no features available.";
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        LatticeFasterDecoder decoder(fst_reader.Value(), config);

        DecodableAmDiagGmmScaledMulti gmm_decodable(am_gmms, mapping,
                       trans_model, features, acoustic_scale,
                       exp_weight);
        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, gmm_decodable, trans_model, word_syms, utt,
                acoustic_scale, determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &like)) {
          tot_like += like;
          frame_count += features.NumRows();
          num_done++;
        } else num_err++;
      }
    }
      
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_done << " utterances, failed for "
              << num_err;
    KALDI_LOG << "Overall log-likelihood per frame is " 
              << (tot_like/frame_count)
              << " over " << frame_count << " frames.";

    if (word_syms) delete word_syms;
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
