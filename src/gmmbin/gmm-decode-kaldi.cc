// gmmbin/gmm-decode-kaldi.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/kaldi-decoder-left.h"
// you can either use left or right: without or with reorder option
#include "decoder/decodable-am-diag-gmm.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc
#include "util/timer.h"

using namespace kaldi;

fst::ConstFst<fst::StdArc> *ReadNetwork(std::string filename) {
  // read decoding network FST
  Input ki(filename); // use ki.Stream() instead of is.
  if (!ki.Stream().good()) KALDI_EXIT << "Could not open decoding-graph FST "
                                      << filename;

  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), "<unknown>")) {
    KALDI_ERR << "Reading FST: error reading FST header.";
  }
  if (hdr.ArcType() != fst::StdArc::Type()) {
    KALDI_ERR << "FST with arc type " << hdr.ArcType() << " not supported.\n";
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::ConstFst<fst::StdArc> *decode_fst = NULL;
  
  if (hdr.FstType() == "vector") {
    fst::VectorFst<fst::StdArc> *read_fst = NULL;
    read_fst = fst::VectorFst<fst::StdArc>::Read(ki.Stream(), ropts);
    if (read_fst == NULL) { // fst code will warn.
      KALDI_ERR << "Error reading FST (after reading header).";
    }
    KALDI_WARN << "We suggest to use ConstFST instead of VectorFST.";
    decode_fst = new fst::ConstFst<fst::StdArc>(*read_fst);
    // copy to ConstFst.  If memory exhausted, should copy as ConstFst to disk
    // the conversion is very time consuming due to computation of FstProperties
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(ki.Stream(), ropts);
        // fst::FstReadOptions(filename));
  } else {
    KALDI_ERR << "Reading FST: unsupported FST type: " << hdr.FstType();
  }
  if (decode_fst == NULL) { // fst code will warn.
    KALDI_ERR << "Error reading FST (after reading header).";
    return NULL;
  } else {
    return decode_fst;
  }
}


int main(int argc, char *argv[]) {
  try {
#ifdef _MSC_VER
    if (0) { new fst::ConstFst<fst::StdArc>(* static_cast<fst::VectorFst<fst::StdArc>*> (NULL)); }
#endif
    const char *usage =
        "Decode features using GMM-based model.\n"
        "Usage:   gmm-decode-kaldi [options] model-in fst-in features-rspecifier words-wspecifier [lattice-wspecifier]\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    std::string word_syms_filename;
    KaldiDecoderOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words");

    po.Read(argc, argv);

    if ((po.NumArgs() < 4) || (po.NumArgs() > 5)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        words_wspecifier = po.GetArg(4),
        lattice_wspecifier = po.GetArg(5);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input is(model_in_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_gmm.Read(is.Stream(), binary);
    }

    Int32VectorWriter words_writer(words_wspecifier);

    bool write_lattices = false;
    LatticeWriter lattice_writer;
    if (lattice_wspecifier != "") {
      if (lattice_writer.Open(lattice_wspecifier)) {
        write_lattices = true;
      } else {
        KALDI_EXIT << "Could not open table for writing lattices: "
                   << lattice_wspecifier;
      }
    }

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_EXIT << "Could not read symbol table from file "
                   << word_syms_filename;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    // It's important that we initialize decode_fst after feature_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    fst::ConstFst<fst::StdArc> *decode_fst = ReadNetwork(fst_in_filename);

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    KaldiDecoder<DecodableAmDiagGmmScaled, fst::ConstFst<fst::StdArc> > decoder(decoder_opts);
    // templating on ConstFst gives a small improvement in speed

    Timer timer;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &features = feature_reader.Value();

      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        continue;
      }

      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);
      fst::VectorFst<LatticeArc> *decoded = decoder.Decode(*decode_fst,
                                                            &gmm_decodable);

      if (decoded == NULL) {
        KALDI_WARN << "Could not decode file " << key;
      } else {
      
//        std::cout << "best path:\n";
//        fst::FstPrinter<fst::StdArc> fstprinter(*decoded, NULL, NULL, NULL, false, true);
//        fstprinter.Print(&std::cout, "standard output");

        if (write_lattices) {
          //if (acoustic_scale != 0.0) // We'll write the lattice without acoustic scaling
          //  fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), *decoded);
          //fst::VectorFst<CompactLatticeArc> decoded1;
          //ConvertLattice(decoded, &decoded1, true);
          lattice_writer.Write(key, *decoded);
        }
      
        std::vector<kaldi::int32> words;
        LatticeWeight weight(0.0, 0.0);
        GetLinearSymbolSequence(*decoded, static_cast<std::vector<kaldi::int32>*>(NULL),
                                &words, &weight);

        frame_count += features.NumRows();

        words_writer.Write(key, words);

        if (word_syms != NULL) {
          std::cerr << key << ' ';
          for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
        BaseFloat like = -weight.Value1() -weight.Value2();
        tot_like += like;
        KALDI_LOG << "Log-like per frame for utterance " << key << " is "
                  << (like / features.NumRows()) << " over "
                  << features.NumRows() << " frames.";
        delete decoded;
      }
    }
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);

    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";
    
    if (word_syms) delete word_syms;
    delete decode_fst;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


