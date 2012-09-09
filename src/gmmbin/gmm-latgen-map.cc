// gmmbin/gmm-decode-faster-regtree-fmllr.cc

// Copyright 2012  Cisco Systems (author: Nega Agrawal)

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
#include "transform/fmllr-diag-gmm.h"
#include "fstext/fstext-lib.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/decodable-am-diag-gmm.h"
#include "util/timer.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc
#include "gmm/map-diag-gmm-accs.h"

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
             const kaldi::TransitionModel &tm,
             kaldi::LatticeFasterDecoder *decoder,
             BaseFloat scale, bool allow_partial,
             const kaldi::Int32VectorWriter &wwriter,
             const kaldi::Int32VectorWriter &awriter,
             const kaldi::CompactLatticeWriter &cl_writer,
             const kaldi::LatticeWriter &lwriter,
             bool write_lattices, bool det,
             fst::SymbolTable *wsyms)
      : acoustic_model(am), trans_model(tm), decoder(decoder),
        acoustic_scale(scale), allow_partial(allow_partial),
        words_writer(wwriter), alignment_writer(awriter),
        compact_lattice_writer(cl_writer), lattice_writer(lwriter),
        write_lattices(write_lattices), determinize(det), word_syms(wsyms) {}

  const kaldi::AmDiagGmm &acoustic_model;
  const kaldi::TransitionModel &trans_model;
  kaldi::LatticeFasterDecoder *decoder;
  BaseFloat acoustic_scale;
  bool allow_partial;
  const kaldi::Int32VectorWriter &words_writer;
  const kaldi::Int32VectorWriter &alignment_writer;
  const kaldi::CompactLatticeWriter &compact_lattice_writer;
  const kaldi::LatticeWriter &lattice_writer;
  bool write_lattices;
  bool determinize;
  fst::SymbolTable *word_syms;

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodeInfo);
};

bool DecodeUtterance(kaldi::LatticeFasterDecoder *decoder,
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

 
    //##############
    if (info->write_lattices) {
      if (info->determinize) {
        kaldi::CompactLattice fst;
        if (!decoder->GetLattice(&fst))
          KALDI_ERR << "Unexpected problem getting lattice for utterance "
                    << uttid;

        if (info->acoustic_scale != 0.0) // We'll write the lattice 
          // without acoustic scaling
          fst::ScaleLattice(fst::AcousticLatticeScale(
              1.0 / info->acoustic_scale), &fst);

        info->compact_lattice_writer.Write(uttid, fst);

      } else {

        kaldi::Lattice fst;
        if (!decoder->GetRawLattice(&fst)) 
          KALDI_ERR << "Unexpected problem getting lattice for utterance "
                    << uttid;
        fst::Connect(&fst); // Will get rid of this later... 
        // shouldn't have any
        // disconnected states there, but we seem to.
        if (info->acoustic_scale != 0.0) // We'll write the lattice 
          // without acoustic scaling
          fst::ScaleLattice(fst::AcousticLatticeScale(
              1.0 / info->acoustic_scale), &fst);
        info->lattice_writer.Write(uttid, fst);

      }
    }

    //################
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
        "This version reads a separate model per (utterance or speaker)\n"
        "typically from an archive piped in from gmm-est-map.\n"
        "Usage: gmm-latgen-map [options] model-in "
        "map-rspecifier fsts-rspecifier features-rspecifier "
        "words-wspecifier [alignments-wspecifier lattice-wspecifier]\n";
        
    ParseOptions po(usage);
    bool binary = true;
    bool allow_partial = true;
    BaseFloat acoustic_scale = 0.1;
        
    std::string word_syms_filename, utt2spk_rspecifier;
    LatticeFasterDecoderConfig decoder_opts;
    decoder_opts.Register(&po);
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

    if (po.NumArgs() < 5 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        map_rspecifier = po.GetArg(2),
        fst_in_filename = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        words_wspecifier = po.GetArg(5),
        alignment_wspecifier = po.GetOptArg(6),
        lattice_wspecifier = po.GetOptArg(7);

    RandomAccessTableReader<fst::VectorFstHolder> fst_reader(fst_in_filename);

    TransitionModel trans_model;
    {
      bool binary_read;
      Input is(model_in_filename, &binary_read);
      trans_model.Read(is.Stream(), binary_read);
    }
    RandomAccessMapAmDiagGmmReader map_reader(map_rspecifier);


    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);


    bool determinize = decoder_opts.determinize_lattice;
    if (!determinize)
      KALDI_WARN << "determinize is set to FASLE ...";
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;

    bool write_lattices = false;
    if (lattice_wspecifier != "") {
      write_lattices = true;
      if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
             : lattice_writer.Open(lattice_wspecifier)))
        KALDI_ERR << "Could not open table for writing lattices: "
                  << lattice_wspecifier;
    }
        


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
    Timer timer;

    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    for (; !feature_reader.Done(); feature_reader.Next()) {
      string utt = feature_reader.Key();

      if (!fst_reader.HasKey(utt)) {
        KALDI_WARN << "Utterance " << utt << " has no corresponding FST"
                   << "skipping this utterance.";
        num_fail++;
        continue;
      }

      string utt_or_spk;
      if (utt2spk_rspecifier == "") utt_or_spk = utt;
      else {
        if (!utt2spk_reader.HasKey(utt)) {
          KALDI_WARN << "Utterance " << utt 
                     << " not present in utt2spk map; "
                     << "skipping this utterance.";
          num_fail++;
          continue;
        } else utt_or_spk = utt2spk_reader.Value(utt);
      }

      if (!map_reader.HasKey(utt_or_spk)) {
        KALDI_WARN << "Utterance " << utt_or_spk 
                   << " has no corresponding MAP model skipping this utterance.";
        num_fail++;
        continue;
      }
      AmDiagGmm am_gmm;
      am_gmm.CopyFromAmDiagGmm(map_reader.Value(utt_or_spk));

      LatticeFasterDecoder decoder(fst_reader.Value(utt), decoder_opts);
      DecodeInfo decode_info(am_gmm, trans_model, &decoder, acoustic_scale,
                             allow_partial, words_writer, alignment_writer, 
                             compact_lattice_writer, lattice_writer, write_lattices, 
                             determinize, word_syms);


      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }

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
    }  // end looping over all utterances

    KALDI_LOG << "Average log-likelihood per frame is " 
              << (tot_like / frame_count) << " over " << frame_count << " frames.";

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] " << elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed * 100.0 / frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    if (word_syms) delete word_syms;
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


