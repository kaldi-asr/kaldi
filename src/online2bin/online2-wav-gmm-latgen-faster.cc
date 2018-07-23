// online2bin/online2-wav-gmm-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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
#include "online2/online-feature-pipeline.h"
#include "online2/online-gmm-decoding.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"

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
        "Reads in wav file(s) and simulates online decoding, including\n"
        "basis-fMLLR adaptation and endpointing.  Writes lattices.\n"
        "Models are specified via options.\n"
        "\n"
        "Usage: online2-wav-gmm-latgen-faster [options] <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
        "Run egs/rm/s5/local/run_online_decoding.sh for example\n";
    
    ParseOptions po(usage);
    
    std::string word_syms_rxfilename;
    
    OnlineEndpointConfig endpoint_config;
    OnlineFeaturePipelineCommandLineConfig feature_cmdline_config;
    OnlineGmmDecodingConfig decode_config;
    
    BaseFloat chunk_length_secs = 0.05;
    bool do_endpointing = false;
    std::string use_gpu = "no";
    
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    
    feature_cmdline_config.Register(&po);
    decode_config.Register(&po);
    endpoint_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }
    
    std::string fst_rxfilename = po.GetArg(1),
        spk2utt_rspecifier = po.GetArg(2),
        wav_rspecifier = po.GetArg(3),
        clat_wspecifier = po.GetArg(4);
    
    OnlineFeaturePipelineConfig feature_config(feature_cmdline_config);
    OnlineFeaturePipeline pipeline_prototype(feature_config);
    // The following object initializes the models we use in decoding.
    OnlineGmmDecodingModels gmm_models(decode_config);
    
    
    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);
    
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
    
    OnlineTimingStats timing_stats;
    
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      OnlineGmmAdaptationState adaptation_state;
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
        
        SingleUtteranceGmmDecoder decoder(decode_config,
                                          gmm_models,
                                          pipeline_prototype,
                                          *decode_fst,
                                          adaptation_state);
        
        OnlineTimer decoding_timer(utt);
        
        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length = int32(samp_freq * chunk_length_secs);
        if (chunk_length == 0) chunk_length = 1;
        
        int32 samp_offset = 0;
        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;
          
          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          decoder.FeaturePipeline().AcceptWaveform(samp_freq, wave_part);
          
          samp_offset += num_samp;
          decoding_timer.WaitUntil(samp_offset / samp_freq);
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            decoder.FeaturePipeline().InputFinished();
          }
          decoder.AdvanceDecoding();
          
          if (do_endpointing && decoder.EndpointDetected(endpoint_config))
            break;
        }
        decoder.FinalizeDecoding();

        bool end_of_utterance = true;
        decoder.EstimateFmllr(end_of_utterance);
        CompactLattice clat;
        bool rescore_if_needed = true;
        decoder.GetLattice(rescore_if_needed, end_of_utterance, &clat);
        
        GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                     &num_frames, &tot_like);
        
        decoding_timer.OutputStats(&timing_stats);
        
        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        decoder.GetAdaptationState(&adaptation_state);
        
        // we want to output the lattice with un-scaled acoustics.
        if (decode_config.acoustic_scale != 0.0) {
          BaseFloat inv_acoustic_scale = 1.0 / decode_config.acoustic_scale;
          ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);
        }
        clat_writer.Write(utt, clat);
        KALDI_LOG << "Decoded utterance " << utt;
        num_done++;
      }
    }
    timing_stats.Print();    
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
