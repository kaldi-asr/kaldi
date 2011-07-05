// gmmbin/gmm-decode-faster-regtree-mllr.cc

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

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/regression-tree.h"
#include "transform/regtree-mllr-diag-gmm.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-am-diag-gmm.h"
#include "util/timer.h"

using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using kaldi::BaseFloat;
using std::string;
using std::vector;

struct DecodeInfo {
 public:
  DecodeInfo(const kaldi::AmDiagGmm &am,
             const kaldi::TransitionModel &tm, kaldi::FasterDecoder *decoder,
             BaseFloat scale, bool reverse,
             const kaldi::Int32VectorWriter &wwriter,
             const kaldi::Int32VectorWriter &awriter, fst::SymbolTable *wsyms)
      : acoustic_model(am), trans_model(tm), decoder(decoder),
        acoustic_scale(scale), time_reversed(reverse), words_writer(wwriter),
        alignment_writer(awriter), word_syms(wsyms) {}

  const kaldi::AmDiagGmm &acoustic_model;
  const kaldi::TransitionModel &trans_model;
  kaldi::FasterDecoder *decoder;
  BaseFloat acoustic_scale;
  bool time_reversed;
  const kaldi::Int32VectorWriter &words_writer;
  const kaldi::Int32VectorWriter &alignment_writer;
  fst::SymbolTable *word_syms;

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodeInfo);
};

void ReverseFeatures(kaldi::Matrix<BaseFloat> *feats) {
  kaldi::Vector<BaseFloat> tmp(feats->NumCols());
  for (int32 i = 0; i < feats->NumRows() / 2; i++) {
    size_t j = feats->NumRows() - i - 1;  // mirror-image of i.
    tmp.CopyRowFromMat(*feats, i);
    feats->Row(i).CopyRowFromMat(*feats, j);
    feats->Row(j).CopyFromVec(tmp);
  }
}


bool DecodeUtterance(kaldi::FasterDecoder *decoder,
                     kaldi::DecodableInterface *decodable,
                     DecodeInfo *info,
                     const string &uttid,
                     int32 num_frames,
                     BaseFloat *total_like) {
  decoder->Decode(decodable);
  KALDI_LOG << "Length of file is " << num_frames << '\n';

  VectorFst<StdArc> decoded;  // linear FST.
  bool saw_endstate = decoder->GetOutput(true /*only final states*/, &decoded);
  if (saw_endstate || decoder->GetOutput(false, &decoded)) {
    if (!saw_endstate) {
      KALDI_WARN << "Decoder did not reach end-state, outputting partial "
          "traceback.";
    }
    vector<kaldi::int32> alignment, words;
    StdArc::Weight weight;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

    if (info->time_reversed) {
      kaldi::ReverseVector(&alignment);
      kaldi::ReverseVector(&words);
    }

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

    BaseFloat like = -weight.Value();
    KALDI_LOG << "Log-like per frame = " << (like/num_frames)
        << "\n";
    (*total_like) += like;
    return true;
  } else {
    KALDI_WARN << "Did not successfully decode utterance, length = "
        << num_frames << "\n";
    return false;
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage = "Decode features using GMM-based model.\n"
              "Usage: gmm-decode-faster-regtree-mllr [options] model-in fst-in "
              "regtree-in features-rspecifier transforms-rspecifier "
              "words-wspecifier [alignments-wspecifier]\n";
    ParseOptions po(usage);
    bool binary = false;
    bool time_reversed = false;
    BaseFloat acoustic_scale = 0.1;

    std::string word_syms_filename, utt2spk_rspecifier;
    FasterDecoderOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("utt2spk", &utt2spk_rspecifier, "rspecifier for utterance to "
                "speaker map");
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("time-reversed", &time_reversed,
        "If true, decode backwards in time [requires reversed graph.]\n");
    po.Register("acoustic-scale", &acoustic_scale,
        "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
        "Symbol table for words [for debug output]");
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
      Input is(model_in_filename, &binary_read);
      trans_model.Read(is.Stream(), binary_read);
      am_gmm.Read(is.Stream(), binary_read);
    }

    VectorFst<StdArc> *decode_fst = NULL;
    {
      std::ifstream is(fst_in_filename.c_str(), std::ifstream::binary);
      if (!is.good())
        KALDI_EXIT << "Could not open decoding-graph FST " << fst_in_filename;
      decode_fst = VectorFst<StdArc>::Read(is, fst::FstReadOptions(fst_in_filename));
      if (decode_fst == NULL)  // fst code will warn.
        exit(1);
    }

    RegressionTree regtree;
    {
      bool binary_read;
      Input in(regtree_filename, &binary_read);
      regtree.Read(in.Stream(), binary_read, am_gmm);
    }

    RandomAccessRegtreeMllrDiagGmmReader mllr_reader(xforms_rspecifier);

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer;
    if (alignment_wspecifier != "")
      if (!alignment_writer.Open(alignment_wspecifier))
        KALDI_ERR << "Failed to open alignments output.";

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      word_syms = fst::SymbolTable::ReadText(word_syms_filename);
      if (!word_syms) {
        KALDI_EXIT << "Could not read symbol table from file "
            << word_syms_filename;
      }
    }

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    FasterDecoder decoder(*decode_fst, decoder_opts);

    Timer timer;

    DecodeInfo decode_info(am_gmm, trans_model, &decoder, acoustic_scale,
                           time_reversed, words_writer, alignment_writer,
                           word_syms);


    RandomAccessTokenReader utt2spk_reader;
    if (utt2spk_rspecifier != "")  // per-speaker adaptation
      if (!utt2spk_reader.Open(utt2spk_rspecifier))
        KALDI_ERR << "Could not open the utt2spk map: "
                  << utt2spk_rspecifier;


    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    for (; !feature_reader.Done(); feature_reader.Next()) {
      string utt = feature_reader.Key();
      string utt_or_spk;
      if (utt2spk_rspecifier == "") {
        utt_or_spk = utt;
      } else {
        if (!utt2spk_reader.HasKey(utt)) {
          KALDI_WARN << "Utterance " << utt << " not present in utt2spk map; "
                     << "skipping this utterance.";
          num_fail++;
          continue;
        } else {
          utt_or_spk = utt2spk_reader.Value(utt);
        }
      }

      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }
      if (time_reversed) ReverseFeatures(&features);

      if (!mllr_reader.HasKey(utt_or_spk)) {  // Decode without MLLR if none found
        KALDI_WARN << "No MLLR transform for key " << utt_or_spk <<
            ", decoding without MLLR.";
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
      const RegtreeMllrDiagGmm& mllr = mllr_reader.Value(utt_or_spk);
      kaldi::DecodableAmDiagGmmRegtreeMllr gmm_decodable(am_gmm, trans_model,
                                                         features, mllr,
                                                         regtree,
                                                         acoustic_scale);
      if (DecodeUtterance(&decoder, &gmm_decodable, &decode_info,
                          utt, features.NumRows(), &tot_like)) {
        frame_count += gmm_decodable.NumFrames();
        num_success++;
      } else {
        num_fail++;
      }
    }  // end looping over all utterances

    std::cerr << "Average log-likelihood per frame is " << (tot_like
        / frame_count) << " over " << frame_count << " frames.\n";

    double elapsed = timer.Elapsed();
    std::cerr << "Time taken [excluding initialization] " << elapsed
        << "s: real-time factor assuming 100 frames/sec is " << (elapsed
        * 100.0 / frame_count) << '\n';
    std::cerr << "Succeeded for " << num_success << " utterances, failed for "
        << num_fail << '\n';

    delete decode_fst;
    if (num_success != 0)
      return 0;
    else
      return 1;
  }
  catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


