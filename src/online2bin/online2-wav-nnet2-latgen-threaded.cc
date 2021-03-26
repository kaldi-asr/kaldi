// online2bin/online2-wav-nnet2-latgen-threaded.cc

// Copyright 2014-2015  Johns Hopkins University (author: Daniel Povey)

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
#include "online2/online-nnet2-decoding-threaded.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"

namespace kaldi {

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";

  if (word_syms != NULL) {
    std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      std::cerr << s << ' ';
    }
    std::cerr << std::endl;
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet2 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  This version uses multiple threads for decoding.\n"
        "Note: some configuration values and inputs are set via config files\n"
        "whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet2-latgen-threaded [options] <nnet2-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n"
        "See egs/rm/s5/local/run_online_decoding_nnet2.sh for example\n"
        "See also online2-wav-nnet2-latgen-faster\n";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    OnlineEndpointConfig endpoint_config;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    OnlineNnet2DecodingThreadedConfig nnet2_decoding_config;

    BaseFloat chunk_length_secs = 0.05;
    bool do_endpointing = false;
    bool modify_ivector_config = false;
    bool simulate_realtime_decoding = true;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we provide each time to the "
                "decoder.  The actual chunk sizes it processes for various stages "
                "of decoding are dynamically determinated, and unrelated to this");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    po.Register("modify-ivector-config", &modify_ivector_config,
                "If true, modifies the iVector configuration from the config files "
                "by setting --use-most-recent-ivector=true and --greedy-ivector-extractor=true. "
                "This will give the best possible results, but the results may become dependent "
                "on the speed of your machine (slower machine -> better results).  Compare "
                "to the --online option in online2-wav-nnet2-latgen-faster");
    po.Register("simulate-realtime-decoding", &simulate_realtime_decoding,
                "If true, simulate real-time decoding scenario by providing the "
                "data incrementally, calling sleep() until each piece is ready. "
                "If false, don't sleep (so it will be faster).");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.  ");

    feature_opts.Register(&po);
    nnet2_decoding_config.Register(&po);
    endpoint_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet2_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        spk2utt_rspecifier = po.GetArg(3),
        wav_rspecifier = po.GetArg(4),
        clat_wspecifier = po.GetArg(5);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    if (modify_ivector_config) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
    }

    Matrix<double> global_cmvn_stats;
    if (feature_opts.global_cmvn_stats_rxfilename != "")
      ReadKaldiObject(feature_opts.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats);

    TransitionModel trans_model;
    nnet2::AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet2_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;
    Timer global_timer;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    CompactLatticeWriter clat_writer(clat_wspecifier);

    OnlineTimingStats timing_stats;

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

        SingleUtteranceNnet2DecoderThreaded decoder(
            nnet2_decoding_config, trans_model, am_nnet,
            *decode_fst, feature_info, adaptation_state, cmvn_state);

        OnlineTimer decoding_timer(utt);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        KALDI_ASSERT(chunk_length_secs > 0);
        chunk_length = int32(samp_freq * chunk_length_secs);
        if (chunk_length == 0) chunk_length = 1;

        int32 samp_offset = 0;
        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);

          // The endpointing code won't work if we let the waveform be given to
          // the decoder all at once, because we'll exit this while loop, and
          // the endpointing happens inside this while loop.  The next statement
          // is intended to prevent this from happening.
          while (do_endpointing &&
                 decoder.NumWaveformPiecesPending() * chunk_length_secs > 2.0)
            Sleep(0.5f);

          decoder.AcceptWaveform(samp_freq, wave_part);

          samp_offset += num_samp;

          if (simulate_realtime_decoding) {
            // Note: the next call may actually call sleep().
            decoding_timer.SleepUntil(samp_offset / samp_freq);
          }
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            decoder.InputFinished();
          }

          if (do_endpointing && decoder.EndpointDetected(endpoint_config)) {
            decoder.TerminateDecoding();
            break;
          }
        }
        Timer timer;
        decoder.Wait();
        if (simulate_realtime_decoding) {
          KALDI_VLOG(1) << "Waited " << timer.Elapsed() << " seconds for decoder to "
                        << "finish after giving it last chunk.";
        }
        decoder.FinalizeDecoding();

        CompactLattice clat;
        bool end_of_utterance = true;
        decoder.GetLattice(end_of_utterance, &clat, NULL);

        GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                     &num_frames, &tot_like);

        decoding_timer.OutputStats(&timing_stats);

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        decoder.GetAdaptationState(&adaptation_state);
        decoder.GetCmvnState(&cmvn_state);

        // we want to output the lattice with un-scaled acoustics.
        BaseFloat inv_acoustic_scale =
            1.0 / nnet2_decoding_config.acoustic_scale;
        ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

        if (simulate_realtime_decoding) {
          KALDI_VLOG(1) << "Adding the various end-of-utterance tasks took the "
                        << "total latency to " << timer.Elapsed() << " seconds.";
        }
        clat_writer.Write(utt, clat);
        KALDI_LOG << "Decoded utterance " << utt;

        num_done++;
      }
    }
    bool online = true;

    if (simulate_realtime_decoding) {
      timing_stats.Print(online);
    } else {
      BaseFloat frame_shift = 0.01;
      BaseFloat real_time_factor =
          global_timer.Elapsed() / (frame_shift * num_frames);
      if (num_frames > 0)
        KALDI_LOG << "Real-time factor was " << real_time_factor
                  << " assuming frame shift of " << frame_shift;
    }

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
