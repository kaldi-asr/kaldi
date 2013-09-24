// gmmbin/gmm-decode-faster-regtree-fmllr.cc

// Copyright 2009-2012  Microsoft Corporation;  Saarland University;
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

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/regression-tree.h"
#include "transform/regtree-fmllr-diag-gmm.h"
#include "transform/fmllr-diag-gmm.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "transform/decodable-am-diag-gmm-regtree.h"
#include "util/timer.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using kaldi::BaseFloat;
using std::string;
using std::vector;
using kaldi::LatticeWeight;
using kaldi::LatticeArc;

struct DecodeInfo {
 public:
  DecodeInfo(const kaldi::AmDiagGmm &am,
             const kaldi::TransitionModel &tm, kaldi::FasterDecoder *decoder,
             BaseFloat scale, bool allow_partial,
             const kaldi::Int32VectorWriter &wwriter,
             const kaldi::Int32VectorWriter &awriter, fst::SymbolTable *wsyms)
      : acoustic_model(am), trans_model(tm), decoder(decoder),
        acoustic_scale(scale), allow_partial(allow_partial), words_writer(wwriter),
        alignment_writer(awriter), word_syms(wsyms) {}

  const kaldi::AmDiagGmm &acoustic_model;
  const kaldi::TransitionModel &trans_model;
  kaldi::FasterDecoder *decoder;
  BaseFloat acoustic_scale;
  bool allow_partial;
  const kaldi::Int32VectorWriter &words_writer;
  const kaldi::Int32VectorWriter &alignment_writer;
  fst::SymbolTable *word_syms;

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodeInfo);
};

bool DecodeUtterance(kaldi::FasterDecoder *decoder,
                     kaldi::DecodableInterface *decodable,
                     DecodeInfo *info,
                     const string &uttid,
                     int32 num_frames,
                     BaseFloat *total_like) {
  decoder->Decode(decodable);
  KALDI_LOG << "Length of file is " << num_frames;

  VectorFst<LatticeArc> decoded;  // linear FST.
  if ( (info->allow_partial || decoder->ReachedFinal())
       && decoder->GetBestPath(&decoded) ) {
    if (!decoder->ReachedFinal())
      KALDI_WARN << "Decoder did not reach end-state, outputting partial "
          "traceback.";
    
    vector<kaldi::int32> alignment, words;
    LatticeWeight weight;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

    info->words_writer.Write(uttid, words);
    if (info->alignment_writer.IsOpen())
      info->alignment_writer.Write(uttid, alignment);
    if (info->word_syms != NULL) {
      std::ostringstream ss;
      ss << uttid << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        string s = info->word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        ss << s << ' ';
      }
      ss << '\n';
      KALDI_LOG << ss.str();
    }

    BaseFloat like = -weight.Value1() -weight.Value2();
    KALDI_LOG << "Log-like per frame = " << (like/num_frames);
    (*total_like) += like;
    return true;
  } else {
    KALDI_WARN << "Did not successfully decode utterance, length = "
               << num_frames;
    return false;
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage = "Decode features using GMM-based model.\n"
              "Usage: gmm-decode-faster-regtree-fmllr [options] model-in fst-in "
              "regtree-in features-rspecifier transforms-rspecifier "
              "words-wspecifier [alignments-wspecifier]\n";
    ParseOptions po(usage);
    bool binary = true;
    bool allow_partial = true;
    BaseFloat acoustic_scale = 0.1;
    
    std::string word_syms_filename, utt2spk_rspecifier;
    FasterDecoderOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("utt2spk", &utt2spk_rspecifier, "rspecifier for utterance to "
                "speaker map");
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("acoustic-scale", &acoustic_scale,
        "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
        "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "Produce output even when final state was not reached");
    po.Read(argc, argv);

    if (po.NumArgs() < 6 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_filename = po.GetArg(2),
        regtree_filename = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        xforms_rspecifier = po.GetArg(5),
        words_wspecifier = po.GetArg(6),
        alignment_wspecifier = po.GetOptArg(7);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    VectorFst<StdArc> *decode_fst = fst::ReadFstKaldi(fst_in_filename);

    RegressionTree regtree;
    {
      bool binary_read;
      Input in(regtree_filename, &binary_read);
      regtree.Read(in.Stream(), binary_read, am_gmm);
    }

    RandomAccessRegtreeFmllrDiagGmmReaderMapped fmllr_reader(xforms_rspecifier,
                                                             utt2spk_rspecifier);

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      word_syms = fst::SymbolTable::ReadText(word_syms_filename);
      if (!word_syms) {
        KALDI_ERR << "Could not read symbol table from file "
            << word_syms_filename;
      }
    }

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    FasterDecoder decoder(*decode_fst, decoder_opts);

    Timer timer;

    DecodeInfo decode_info(am_gmm, trans_model, &decoder, acoustic_scale,
                           allow_partial, words_writer, alignment_writer,
                           word_syms);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    for (; !feature_reader.Done(); feature_reader.Next()) {
      string utt = feature_reader.Key();

      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }

      if (!fmllr_reader.HasKey(utt)) {  // Decode without FMLLR if none found
        KALDI_WARN << "No FMLLR transform for key " << utt <<
            ", decoding without fMLLR.";
        kaldi::DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model,
                                                      features,
                                                      acoustic_scale);
        if (DecodeUtterance(&decoder, &gmm_decodable, &decode_info,
                            utt, features.NumRows(), &tot_like)) {
          frame_count += gmm_decodable.NumFrames();
          num_success++;
        } else {
          num_fail++;
        }
        continue;
      }

      // If found, load the transforms for the current utterance.
      RegtreeFmllrDiagGmm fmllr(fmllr_reader.Value(utt));
      if (fmllr.NumRegClasses() == 1) {
        Matrix<BaseFloat> xformed_features(features);
        Matrix<BaseFloat> fmllr_matrix;
        fmllr.GetXformMatrix(0, &fmllr_matrix);
        for (int32 i = 0; i < xformed_features.NumRows(); i++) {
          SubVector<BaseFloat> row(xformed_features, i);
          ApplyAffineTransform(fmllr_matrix, &row);
        }
        kaldi::DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model,
                                                      xformed_features,
                                                      acoustic_scale);

        if (DecodeUtterance(&decoder, &gmm_decodable, &decode_info,
                            utt, xformed_features.NumRows(), &tot_like)) {
          frame_count += gmm_decodable.NumFrames();
          num_success++;
        } else {
          num_fail++;
        }
      } else {
        kaldi::DecodableAmDiagGmmRegtreeFmllr gmm_decodable(am_gmm, trans_model,
                                                            features, fmllr,
                                                            regtree,
                                                            acoustic_scale);
        if (DecodeUtterance(&decoder, &gmm_decodable, &decode_info,
                            utt, features.NumRows(), &tot_like)) {
          frame_count += gmm_decodable.NumFrames();
          num_success++;
        } else {
          num_fail++;
        }
      }
    }  // end looping over all utterances

    KALDI_LOG << "Average log-likelihood per frame is " << (tot_like
                                                            / frame_count) << " over " << frame_count << " frames.";

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] " << elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed * 100.0 / frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    if (word_syms) delete word_syms;
    delete decode_fst;
    if (num_success != 0)
      return 0;
    else
      return 1;
  }
  catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


