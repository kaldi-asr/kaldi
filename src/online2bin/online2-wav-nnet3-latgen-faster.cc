// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

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
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

#ifndef TEST_TIME
#include <sys/time.h>
#define TEST_TIME(times) do{\
        struct timeval cur_time;\
	    gettimeofday(&cur_time, NULL);\
	    times = (cur_time.tv_sec * 1000000llu + cur_time.tv_usec) / 1000llu;\
	}while(0)
#endif

extern unsigned long long advance_chunk_time;
extern unsigned long long AcceptWaveform_ComputeFeatures_PushBack_time;

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
                << " frames, = " << (-weight.Value1() / num_frames)
                << ',' << (weight.Value2() / num_frames);

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

    unsigned long long start_time = 0, end_time = 0;
    unsigned long long load_params_time = 0, load_trans_model_nnet_time = 0;
    unsigned long long precomputed_stuff_time = 0, read_fst_time = 0;
    unsigned long long read_spk2utt_wav_time = 0;
    unsigned long long get_wave_data_before_time = 0, get_wave_data_after_time = 0;
    unsigned long long get_frame_wave_data_before_time = 0, get_frame_wave_data_after_time = 0;
    unsigned long long get_frame_feature_time = 0, total_frame_feature_time = 0,get_frame_decoding_time = 0, total_frame_decoding_time = 0;
    unsigned long long get_final_decode_after_time = 0, get_final_decode_before_time = 0;
    unsigned long long after_decode_time = 0;
    unsigned long long total_AcceptWaveform_ComputeFeatures_PushBack_time = 0;
    uint32 loop_time = 0;
    TEST_TIME(start_time);

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = 0.185;
    // BaseFloat chunk_length_secs = 4;
    bool do_endpointing = false;
    bool online = true;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
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
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);


    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        spk2utt_rspecifier = po.GetArg(3),
        wav_rspecifier = po.GetArg(4),
        clat_wspecifier = po.GetArg(5);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);
    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    Matrix<double> global_cmvn_stats;
    if (feature_info.global_cmvn_stats_rxfilename != "")
      ReadKaldiObject(feature_info.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats);

    TEST_TIME(load_params_time);
    std::cout <<"\033[0;31mLoad params time: " << load_params_time - start_time << " ms. \033[0;39m" << std::endl;

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

    TEST_TIME(load_trans_model_nnet_time);
    std::cout <<"\033[0;31mLoad trans model and nnet time " << load_trans_model_nnet_time - load_params_time << " ms. \033[0;39m" << std::endl;

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);

    TEST_TIME(precomputed_stuff_time);
    std::cout <<"\033[0;31mPrecomputed stuff time " << precomputed_stuff_time - load_trans_model_nnet_time << " ms. \033[0;39m" << std::endl;

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    TEST_TIME(read_fst_time);
    std::cout <<"\033[0;31mLoad fst time " << read_fst_time - precomputed_stuff_time << " ms. \033[0;39m" << std::endl;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    CompactLatticeWriter clat_writer(clat_wspecifier);

    TEST_TIME(read_spk2utt_wav_time); 
    std::cout <<"\033[0;31mLoad spk2utt wav time " << read_spk2utt_wav_time - read_fst_time << " ms. \033[0;39m\n" << std::endl;
    std::cout <<"\033[0;31mTotal init time " << read_spk2utt_wav_time - start_time << " ms. \033[0;39m\n" << std::endl;

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

        TEST_TIME(get_wave_data_before_time);

        const WaveData &wave_data = wav_reader.Value(utt);
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);

        TEST_TIME(get_wave_data_after_time);     
        std::cout <<"\033[0;31mLoad wave date time " << get_wave_data_after_time - get_wave_data_before_time << " ms. \033[0;39m" << std::endl;

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);
        feature_pipeline.SetCmvnState(cmvn_state);

        OnlineSilenceWeighting silence_weighting(
            trans_model,
            feature_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor);

        SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                            decodable_info,
                                            *decode_fst, &feature_pipeline);
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
        std::vector<std::pair<int32, BaseFloat> > delta_weights;

        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          TEST_TIME(get_frame_wave_data_before_time);
          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);

          TEST_TIME(get_frame_wave_data_after_time);
          std::cout <<"\033[0;31mLoad wave date per frame time " << get_frame_wave_data_after_time - get_frame_wave_data_before_time << " ms. \033[0;39m" << std::endl;

          feature_pipeline.AcceptWaveform(samp_freq, wave_part);
          TEST_TIME(get_frame_feature_time);
          std::cout <<"\033[0;31mAcceptWaveform -> ComputeFeatures -> PushBack: pushback time " << AcceptWaveform_ComputeFeatures_PushBack_time << " ms. \033[0;39m" << std::endl;
          std::cout <<"\033[0;31mAcceptWaveform per frame time " << get_frame_feature_time - get_frame_wave_data_after_time << " ms. \033[0;39m" << std::endl;
          total_AcceptWaveform_ComputeFeatures_PushBack_time += AcceptWaveform_ComputeFeatures_PushBack_time;
          total_frame_feature_time +=  get_frame_feature_time - get_frame_wave_data_after_time;

          samp_offset += num_samp;
          loop_time += 1;
          decoding_timer.WaitUntil(samp_offset / samp_freq);
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
          }

          if (silence_weighting.Active() &&
              feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
          }

          decoder.AdvanceDecoding();
          TEST_TIME(get_frame_decoding_time);
          std::cout <<"\033[0;31mAccept decode per frame time " << get_frame_decoding_time - get_frame_feature_time << " ms. \033[0;39m\n" << std::endl;
          total_frame_decoding_time += get_frame_decoding_time - get_frame_feature_time;

          if (do_endpointing && decoder.EndpointDetected(endpoint_opts)) {
            break;
          }
        }

        std::cout <<"\n\033[0;34mDo AdvanceChunk: " << advance_chunk_time << " ms. \033[0;39m" << std::endl;
        std::cout <<"\033[0;31mAcceptWaveform -> ComputeFeatures -> PushBack: [Total]push back time " << total_AcceptWaveform_ComputeFeatures_PushBack_time << " ms. \033[0;39m" << std::endl;
        std::cout <<"\033[0;31mTotal feature frames time " << total_frame_feature_time << " ms. \033[0;39m" << std::endl;
        std::cout <<"\033[0;31mTotal decode frames time " << total_frame_decoding_time << " ms. \033[0;39m" << std::endl;
        std::cout <<"\033[0;31mTotal loop time " << loop_time << "\033[0;39m\n" << std::endl;
        TEST_TIME(get_final_decode_before_time);
        decoder.FinalizeDecoding();
        TEST_TIME(get_final_decode_after_time);
        std::cout <<"\033[0;31mFinal Decode time " <<  get_final_decode_after_time - get_final_decode_before_time << " ms. \033[0;39m" << std::endl;

        CompactLattice clat;
        bool end_of_utterance = true;
        decoder.GetLattice(end_of_utterance, &clat);

        GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                     &num_frames, &tot_like);

        decoding_timer.OutputStats(&timing_stats);

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        feature_pipeline.GetAdaptationState(&adaptation_state);
        feature_pipeline.GetCmvnState(&cmvn_state);

        // we want to output the lattice with un-scaled acoustics.
        BaseFloat inv_acoustic_scale =
            1.0 / decodable_opts.acoustic_scale;
        ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

        clat_writer.Write(utt, clat);
        KALDI_LOG << "Decoded utterance " << utt;
        num_done++;

        TEST_TIME(after_decode_time);
        std::cout <<"\033[0;31mAfter decode time " << after_decode_time - get_final_decode_after_time << " ms. \033[0;39m" << std::endl;
      }
    }
    timing_stats.Print(online);

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.

    TEST_TIME(end_time);
    std::cout <<"\033[0;31mTotal decode time: " << end_time - read_spk2utt_wav_time << " ms. \033[0;39m" << std::endl;
    std::cout <<"\033[0;31mTotal time: " << end_time - start_time << " ms. \033[0;39m" << std::endl;
    
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
