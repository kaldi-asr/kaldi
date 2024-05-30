// nnet3bin/nnet3-latgen-faster-parallel.cc

// Copyright 2012-2016   Johns Hopkins University (author: Daniel Povey)
//                2014   Guoguo Chen
//                2021   Xiaomi Corporation (Author: Zhao Yan)

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
#include "nnet3/decodable-batch-looped.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"



int main(int argc, char *argv[]) {
  // This program processes utterances in parallel, in looped mode.
  // Additionally, multiple nnet computation requests which from 
  // different decoding threads are batched to run parallelly in GPU.
  // Note that the audio streams represented by these requests can be 
  // different between two consecutive computaion, and the audio streams 
  // of batch can be asynchronous in timing.
  // 
  // First, the computer(type of NnetBatchLoopedComputer) is initialized,
  // start a thread for listening computation request from other threads.
  // The computer batches multiple computation requests and runs inference
  // on GPU, so this program needs CUDA available.
  //
  // Second, the decoder and decodable are constructed for every utterance.
  // The decoding task using decoder and decodable is submited to 
  // TaskSequencer. 
  //
  // Third, multiple decoding tasks run in parallel.
  // the decodable(type of DecodableAmNnetBatchSimpleLoopedParallel)
  // prepare computation request and post it to computer. Then, the thread 
  // is blocked until the computer finishes the computation and wakes it up.
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

#if HAVE_CUDA
    const char *usage =
        "Generate lattices using nnet3 neural net model.  This version supports\n"
        "multiple decoding threads (using a shared decoding graph.)\n"
        "Usage: nnet3-latgen-faster-looped-parallel [options] <nnet-in> <fst-in|fsts-rspecifier> <features-rspecifier>"
        " <lattice-wspecifier> [ <words-wspecifier> [<alignments-wspecifier>] ]\n"
        "See also: nnet3-latgen-faster-parallel nnet3-latgen-faster-batch (which supports GPUs)\n";
    ParseOptions po(usage);

    Timer timer;
    bool allow_partial = false;
    TaskSequencerConfig sequencer_config; // has --num-threads option
    LatticeFasterDecoderConfig config;
    NnetBatchLoopedComputationOptions decodable_opts;

    std::string word_syms_filename;
    std::string ivector_rspecifier,
        online_ivector_rspecifier,
        utt2spk_rspecifier;
    int32 online_ivector_period = 0;
    sequencer_config.Register(&po);
    config.Register(&po);
    decodable_opts.Register(&po);
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

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    CuDevice::Instantiate().AllowMultithreading();
    CuDevice::Instantiate().SelectGpuId("yes");

    TaskSequencer<DecodeUtteranceLatticeFasterClass> sequencer(sequencer_config);
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

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    DecodableNnetBatchLoopedInfo decodable_info(decodable_opts, 
                                                &(am_nnet.GetNnet()));
    NnetBatchLoopedComputer computer(decodable_info);

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset();

      {
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

          LatticeFasterDecoder *decoder =
              new LatticeFasterDecoder(*decode_fst, config);

          DecodableInterface *nnet_decodable = new
              DecodableAmNnetBatchSimpleLoopedParallel(
                  &computer, trans_model,
                  features, ivector, online_ivectors,
                  online_ivector_period);

          DecodeUtteranceLatticeFasterClass *task =
              new DecodeUtteranceLatticeFasterClass(
                  decoder, nnet_decodable, // takes ownership of these two.
                  trans_model, word_syms, utt, decodable_opts.acoustic_scale,
                  determinize, allow_partial, &alignment_writer, &words_writer,
                   &compact_lattice_writer, &lattice_writer,
                   &tot_like, &frame_count, &num_success, &num_fail, NULL);

          sequencer.Run(task); // takes ownership of "task",
                               // and will delete it when done.
        }
      }
      sequencer.Wait(); // Waits for all tasks to be done.
      delete decode_fst;
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no features available.";
          num_fail++;
          continue;
        }
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
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

        // the following constructor takes ownership of the FST pointer so that
        // it is deleted when 'decoder' is deleted.
        LatticeFasterDecoder *decoder =
            new LatticeFasterDecoder(config, fst_reader.Value().Copy());

        DecodableInterface *nnet_decodable = new
            DecodableAmNnetBatchSimpleLoopedParallel(
                &computer, trans_model, 
                features, ivector, online_ivectors,
                online_ivector_period);

        DecodeUtteranceLatticeFasterClass *task =
            new DecodeUtteranceLatticeFasterClass(
                decoder, nnet_decodable, // takes ownership of these two.
                trans_model, word_syms, utt, decodable_opts.acoustic_scale,
                determinize, allow_partial, &alignment_writer, &words_writer,
                &compact_lattice_writer, &lattice_writer,
                &tot_like, &frame_count, &num_success, &num_fail, NULL);

        sequencer.Run(task); // takes ownership of "task",
        // and will delete it when done.
      }
      sequencer.Wait(); // Waits for all tasks to be done.
    }

    kaldi::int64 input_frame_count =
        frame_count * decodable_opts.frame_subsampling_factor;

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken " << elapsed
              << "s: real-time factor assuming 100 feature frames/sec is "
              << (sequencer_config.num_threads * elapsed * 100.0 /
                  input_frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over "
              << frame_count << " frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
#else
    KALDI_ERR << "This program requires CUDA available.";
    return 1;
#endif
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
