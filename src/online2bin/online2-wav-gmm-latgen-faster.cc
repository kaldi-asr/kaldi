// onlinebin/online2-wav-gmm-decode-faster.cc

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
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    

    const char *usage =
        "Reads in wav file(s) and simulates online decoding, including\n"
        "basis-fMLLR adaptation.  Writes lattices.  Models are provided via\n"
        "options.\n"
        "\n"
        "Usage: online2-wav-gmm-decode-faster [options] <fst-in> <spk2utt-rspecifier> "
        "<wav-rspecifier> <lattice-wspecifier> "
        "e.g.: ... \n"
        "[TODO]\n";
    ParseOptions po(usage);

    std::string word_syms_rxfilename;
    OnlineFeaturePipelineCommandLineConfig feature_cmdline_config;
    OnlineGmmDecodingConfig decode_config;
    
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    
    feature_cmdline_config.Register(&po);
    decode_config.Register(&po);
    
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
    
    
    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(fst_rxfilename);
    

    // For first utterance, re-estimate fMLLR every two seconds.
    int32 fmllr_interval_first_utt = 200;
    

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;
    
    int32 num_done = 0, num_err = 0;
    double tot_like;
    int64 num_frames;
   
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    CompactLatticeWriter clat_writer(clat_wspecifier);
      
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {      
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      SpeakerAdaptationState adaptation_state;
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find features for utterance " << utt;
          num_err++;
          continue;
        }
        SingleUtteranceGmmDecoder decoder(decode_config,
                                          gmm_models,
                                          pipeline_prototype,
                                          *decode_fst,
                                          adaptation_state);
        const WaveData &wave_data = wav_reader.Value(utt);
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);
        // Very arbitrarily, we decide to process at most one second
        // at a time.
        int32 samp_offset = 0, max_samp = wave_data.SampFreq();
        while (samp_offset < data.Dim()) {
          // This randomness is just for testing purposes and to demonstrate
          // that you can process arbitrary amounts of data.
          int32 this_num_samp = rand() % max_samp;
          if (this_num_samp == 0) this_num_samp = 1;
          if (this_num_samp > data.Dim() - samp_offset)
            this_num_samp = data.Dim() - samp_offset;
          SubVector<BaseFloat> wave_part(data, samp_offset, this_num_samp);
          int32 old_frames_ready = decoder.FeaturePipeline().NumFramesReady();
          decoder.FeaturePipeline().AcceptWaveform(wave_data.SampFreq(),
                                                   wave_part);
          decoder.AdvanceFirstPass();
            
          int32 new_frames_ready = decoder.FeaturePipeline().NumFramesReady();
          bool end_of_utterance = false;
          if (i == 0 &&
              old_frames_ready / fmllr_interval_first_utt !=
              new_frames_ready / fmllr_interval_first_utt)
            decoder.EstimateFmllr(end_of_utterance);
            
          samp_offset += this_num_samp;
        }
        bool end_of_utterance = true;
        decoder.EstimateFmllr(end_of_utterance);
        CompactLattice clat;
        bool rescore_if_needed = true;
        decoder.GetLattice(rescore_if_needed,
                           end_of_utterance,
                           &clat);

        GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                     &num_frames, &tot_like);
        
        // In an application you might avoid updating the adptation state if you
        // felt the utterance had low confidence.  See lat/confidence.h
        decoder.GetAdaptationState(&adaptation_state);
        
        // we want to output the lattice with un-scaled acoustics.
        if (decode_config.acoustic_scale != 0.0)
          ScaleLattice(AcousticLatticeScale(1.0 / decode_config.acoustic_scale),
                       &clat);
        clat_writer.Write(utt, clat);
        KALDI_LOG << "Decoded utterance " << utt;
        num_done++;
      }
    }
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
