// nnet3bin/nnet3-latgen-faster-parallel.cc

// Copyright 2012-2016   Johns Hopkins University (author: Daniel Povey)
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


#include "base/timer.h"
#include "base/kaldi-common.h"
#include "decoder/decoder-wrappers.h"
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-batch-compute.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"

namespace kaldi {

void HandleOutput(bool determinize,
                  const fst::SymbolTable *word_syms,
                  nnet3::NnetBatchDecoder *decoder,
                  CompactLatticeWriter *clat_writer,
                  LatticeWriter *lat_writer) {
  // Write out any lattices that are ready.
  std::string output_utterance_id, sentence;
  if (determinize) {
    CompactLattice clat;
    while (decoder->GetOutput(&output_utterance_id, &clat, &sentence)) {
      if (word_syms != NULL)
        std::cerr << output_utterance_id << ' ' << sentence << '\n';
      clat_writer->Write(output_utterance_id, clat);
    }
  } else {
    Lattice lat;
    while (decoder->GetOutput(&output_utterance_id, &lat, &sentence)) {
      if (word_syms != NULL)
        std::cerr << output_utterance_id << ' ' << sentence << '\n';
      lat_writer->Write(output_utterance_id, lat);
    }
  }
}

}  // namespace kaldi

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
        "Generate lattices using nnet3 neural net model.  This version is optimized\n"
        "for GPU-based inference.\n"
        "Usage: nnet3-latgen-faster-parallel [options] <nnet-in> <fst-in> <features-rspecifier>"
        " <lattice-wspecifier>\n";
    ParseOptions po(usage);

    bool allow_partial = false;
    LatticeFasterDecoderConfig decoder_opts;
    NnetBatchComputerOptions compute_opts;
    std::string use_gpu = "yes";

    std::string word_syms_filename;
    std::string ivector_rspecifier,
        online_ivector_rspecifier,
        utt2spk_rspecifier;
    int32 online_ivector_period = 0, num_threads = 1;
    decoder_opts.Register(&po);
    compute_opts.Register(&po);
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");
    po.Register("ivectors", &ivector_rspecifier, "Rspecifier for "
                "iVectors as vectors (i.e. not estimated online); per utterance "
                "by default, or per speaker if you provide the --utt2spk option.");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier for "
                "iVectors estimated online, as matrices.  If you supply this,"
                " you must set the --online-ivector-period option.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of frames "
                "between iVectors in matrices supplied to the --online-ivectors "
                "option");
    po.Register("num-threads", &num_threads, "Number of decoder (i.e. "
                "graph-search) threads.  The number of model-evaluation threads "
                "is always 1; this is optimized for use with the GPU.");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().AllowMultithreading();
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string model_in_rxfilename = po.GetArg(1),
        fst_in_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(model_in_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      CollapseModel(CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    bool determinize = decoder_opts.determinize_lattice;
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

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;


    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_rxfilename);

    int32 num_success;
    {
      NnetBatchComputer computer(compute_opts, am_nnet.GetNnet(),
                                 am_nnet.Priors());
      NnetBatchDecoder decoder(*decode_fst, decoder_opts,
                               trans_model, word_syms, allow_partial,
                               num_threads, &computer);

      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        const Matrix<BaseFloat> &features (feature_reader.Value());

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          decoder.UtteranceFailed();
          continue;
        }
        const Matrix<BaseFloat> *online_ivectors = NULL;
        const Vector<BaseFloat> *ivector = NULL;
        if (!ivector_rspecifier.empty()) {
          if (!ivector_reader.HasKey(utt)) {
            KALDI_WARN << "No iVector available for utterance " << utt;
            decoder.UtteranceFailed();
            continue;
          } else {
            ivector = &ivector_reader.Value(utt);
          }
        }
        if (!online_ivector_rspecifier.empty()) {
          if (!online_ivector_reader.HasKey(utt)) {
            KALDI_WARN << "No online iVector available for utterance " << utt;
            decoder.UtteranceFailed();
            continue;
          } else {
            online_ivectors = &online_ivector_reader.Value(utt);
          }
        }

        decoder.AcceptInput(utt, features, ivector, online_ivectors,
                            online_ivector_period);

        HandleOutput(decoder_opts.determinize_lattice, word_syms, &decoder,
                     &compact_lattice_writer, &lattice_writer);
      }
      num_success = decoder.Finished();
      HandleOutput(decoder_opts.determinize_lattice, word_syms, &decoder,
                   &compact_lattice_writer, &lattice_writer);

      // At this point the decoder and batch-computer objects will print
      // diagnostics from their destructors (they are going out of scope).
    }
    delete decode_fst;
    delete word_syms;

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
