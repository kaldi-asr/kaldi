// onlinebin/online2-wav-nnet2-decode-faster.cc

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
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "nnet2/online-nnet2-decodable.h"
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
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet2 setup), without speaker adaptation and with optional endpointing.\n"
        "\n"
        "Usage: online2-wav-nnet2-latgen-faster [options] <nnet2-in> <fst-in> "
        "<wav-rspecifier> <lattice-wspecifier>\n"
        "Run ^/egs/rm/s5/local/run_online_decoding_nnet2.sh for example\n";
    
    ParseOptions po(usage);
    
    std::string word_syms_rxfilename;
    
    OnlineEndpointConfig endpoint_config;
    OnlineFeaturePipelineCommandLineConfig feature_cmdline_config;

    // Options for the decoder, including --beam and --lattice-beam.
    LatticeFasterDecoderConfig faster_decoder_opts;
    
    // Options for the Decodable object, including --acoustic-scale
    nnet2::DecodableNnet2OnlineOptions decodable_opts;

    BaseFloat chunk_length_secs = 0.05;
    bool do_endpointing = false;
    
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    
    feature_cmdline_config.Register(&po);
    faster_decoder_opts.Register(&po);
    decodable_opts.Register(&po);
    endpoint_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }
    
    std::string nnet2_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        wav_rspecifier = po.GetArg(3),
        clat_wspecifier = po.GetArg(4);
    
    OnlineFeaturePipelineConfig feature_config(feature_cmdline_config);
    OnlineFeaturePipeline pipeline_prototype(feature_config);

    TransitionModel trans_model;
    nnet2::AmNnet nnet;
    {
      bool binary;
      Input ki(nnet2_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      nnet.Read(ki.Stream(), binary);
    }
    
    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(fst_rxfilename);
    
    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;
    
    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;
    
    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    CompactLatticeWriter clat_writer(clat_wspecifier);
    
    OnlineTimingStats timing_stats;
    
    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string utt = wav_reader.Key();
      const WaveData &wave_data = wav_reader.Value();
      // get the data for channel zero (if the signal is not mono, we only
      // take the first channel).
      SubVector<BaseFloat> data(wave_data.Data(), 0);

      OnlineFeaturePipeline pipeline(pipeline_prototype);

      nnet2::DecodableNnet2Online decodable(nnet, trans_model,
                                            decodable_opts,
                                            &pipeline);

      LatticeFasterOnlineDecoder decoder(*decode_fst, faster_decoder_opts);
      
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
        pipeline.AcceptWaveform(samp_freq, wave_part);
        
        samp_offset += num_samp;
        decoding_timer.WaitUntil(samp_offset / samp_freq);
        if (samp_offset == data.Dim()) {
          // no more input. flush out last frames
          pipeline.InputFinished();
        }
        decoder.AdvanceDecoding(&decodable);
        
        if (do_endpointing && EndpointDetected(endpoint_config, trans_model,
                                               pipeline.FrameShiftInSeconds(),
                                               decoder)) {
            break;
        }
      }

      Lattice lat;
      decoder.GetRawLattice(&lat);
      CompactLattice clat;
      if (!DeterminizeLatticePhonePrunedWrapper(
              trans_model,
              &lat,
              faster_decoder_opts.lattice_beam,
              &clat,
              faster_decoder_opts.det_opts)) {
        KALDI_WARN << "Determinization finished earlier than the beam for "
                   << "utterance " << utt;
      }

      decoding_timer.OutputStats(&timing_stats);
      
      GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                   &num_frames, &tot_like);
      
      KALDI_ASSERT(decodable_opts.acoustic_scale != 0.0);
      // We'll write the lattice without acoustic scaling.
      BaseFloat inv_scale = 1.0 / decodable_opts.acoustic_scale;
      fst::ScaleLattice(fst::AcousticLatticeScale(inv_scale), &clat);
      
      clat_writer.Write(utt, clat);
      
      KALDI_LOG << "Decoded utterance " << utt;
      num_done++;
    }
    timing_stats.Print();
    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Average likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
