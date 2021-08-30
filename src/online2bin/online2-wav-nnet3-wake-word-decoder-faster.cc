// online2bin/online2-wav-nnet3-wake-word-decoder-faster.cc

// Copyright 2019-2020  Daniel Povey
//           2019-2020  Yiming Wang

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

#include "feat/wave-reader.h"
#include "online2/online-nnet3-wake-word-faster-decoder.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "nnet3/decodable-online-looped.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-nnet3-wake-word-faster-decoder.h"

/** This code is modified from online2bin/online2-wav-nnet3-latgen-faster.cc,
    for wake word detection decoding. There is no lattice generation.
*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding for wake word with neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker adaptation.\n"
        "Once the wake word has been detected, or all the feature frames has been processed,\n"
        "the decoding terminates and write the decoded outputs to files.\n"
        "Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet3-wake-word-decoder-faster [options] <nnet3-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <word-symbol-table>  "
        "<transcript-wspecifier> <alignments-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n";

    ParseOptions po(usage);

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    OnlineWakeWordFasterDecoderOpts decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = 1.0;
    bool online = true;
    int32 wake_word_id = 2;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("wake-word-id", &wake_word_id, "Wake word id.");
    po.Register("online", &online,
                "You can set this to false to disable online iVector estimation "
                "and have all the data for each utterance used, even at "
                "utterance start.  This is useful where you just want the best "
                "results and don't care about online operation.  Setting this to "
                "false has the same effect as setting "
                "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                "in the file given to --ivector-extraction-config, and "
                "--chunk-length=-1.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po, true);
    endpoint_opts.Register(&po);


    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        spk2utt_rspecifier = po.GetArg(3),
        wav_rspecifier = po.GetArg(4),
        word_syms_rxfilename = po.GetArg(5),
        words_wspecifier = po.GetArg(6),
        alignment_wspecifier = po.GetArg(7);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);
    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    Matrix<double> global_cmvn_stats;
    if (feature_opts.global_cmvn_stats_rxfilename != "")
      ReadKaldiObject(feature_opts.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats);

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    int32 num_done = 0, num_err = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);

    OnlineTimingStats timing_stats;

    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
      KALDI_ERR << "Could not read symbol table from file "
                << word_syms_rxfilename;

    VectorFst<LatticeArc> out_fst;
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();

      OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);
      OnlineCmvnState cmvn_state(global_cmvn_stats);

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }
        const WaveData &wave_data = wav_reader.Value(utt);
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);
        feature_pipeline.SetCmvnState(cmvn_state);

        nnet3::DecodableAmNnetLoopedOnline decodable(trans_model,
            decodable_info, feature_pipeline.InputFeature(),
            feature_pipeline.IvectorFeature());

        OnlineWakeWordFasterDecoder decoder(*decode_fst, decoder_opts);

        OnlineTimer decoding_timer(utt);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
          chunk_length = int32(samp_freq * chunk_length_secs);
          if (chunk_length == 0) chunk_length = 1;
        } else {
          chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;

        bool partial_res = false;
        decoder.InitDecoding();
        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);

          samp_offset += num_samp;
          decoding_timer.WaitUntil(samp_offset / samp_freq);
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
          }

          decoder.AdvanceDecoding(&decodable);
          if (decodable.IsLastFrame(decoder.NumFramesDecoded() - 1)) {
            std::vector<int32> word_ids;
            decoder.FinishTraceBack(&out_fst);
            fst::GetLinearSymbolSequence(out_fst,
                                         static_cast<std::vector<int32> *>(0),
                                         &word_ids,
                                         static_cast<LatticeArc::Weight*>(0));
            PrintPartialResult(word_ids, word_syms, partial_res || word_ids.size());
            partial_res = false;
            decoder.GetBestPath(&out_fst);
            std::vector<int32> tids;
            fst::GetLinearSymbolSequence(out_fst,
                                         &tids,
                                         &word_ids,
                                         static_cast<LatticeArc::Weight*>(0));
            //if (!word_ids.empty())
            words_writer.Write(utt, word_ids);
            alignment_writer.Write(utt, tids);
            break;
          } else {
            std::vector<int32> word_ids;
            if (decoder.PartialTraceback(&out_fst)) {
              fst::GetLinearSymbolSequence(out_fst,
                                           static_cast<std::vector<int32> *>(0),
                                           &word_ids,
                                           static_cast<LatticeArc::Weight*>(0));
              PrintPartialResult(word_ids, word_syms, false);
              if (!partial_res)
                partial_res = (word_ids.size() > 0);
              if (std::find(word_ids.begin(), word_ids.end(), wake_word_id) !=
                  word_ids.end()) {
                decoder.GetBestPath(&out_fst);
                std::vector<int32> tids;
                fst::GetLinearSymbolSequence(out_fst,
                                             &tids,
                                             &word_ids,
                                             static_cast<LatticeArc::Weight*>(0));
                words_writer.Write(utt, word_ids);
                alignment_writer.Write(utt, tids);
                break;
              }
            }
          }
        }

        decoding_timer.OutputStats(&timing_stats);

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        feature_pipeline.GetAdaptationState(&adaptation_state);
        feature_pipeline.GetCmvnState(&cmvn_state);

        KALDI_LOG << "Decoded utterance " << utt;
        num_done++;
      }
    }
    timing_stats.Print(online);

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
