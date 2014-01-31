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
        "Usage: online-wav-gmm-decode-faster [options] <fst-in> <wav-rspecifier> "
        "<lattice-wspecifier> "
        "e.g.: ... \n"
        "[TODO]\n"
    ParseOptions po(usage);

    std::string spk2utt;
    OnlineFeaturePipelineConfig feature_config;
    OnlineGmmDecodingConfig decode_config;
    
    feature_config.Register(&po);
    decode_config.Register(&po);
    po.Register("spk2utt", &spk2utt, "rspecifier for spk2utt mapping, "
                "used when multiple utterances of the same speaker are "
                "to be processed in incremental-adaptation mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      return 1;
    }

    // pipeline_prototype will be used for its Copy() operator
    // to copy an identical pipeline for each utterance.
    OnlineFeaturePipeline pipeline_prototype(feature_config);
    // The following object initializes the models we use in decoding.
    OnlineGmmDecodingModels gmm_models(decode_config);
    
    std::string fst_rxfilename = po.GetArg(1),
        wav_rspecifier = po.GetArg(2),
        lat_wspecifier = po.GetArg(3);

    fst::Fst<fst::StdArc> decode_fst = *ReadDecodeGraph(fst_rspecifier);


    int32 num_done = 0, num_err = 0;
    
    if (spk2utt_rxfilename == "") {
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
      // TODO.
    } else {
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
      
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {      
        std::string spk = spk2utt_reader.Key();
        const std::vector<std::string> &uttlist = spk2utt_reader.Value();
        bool started = false;
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
            KALDI_ASSERT(data.SampFreq() == decoder.FeaturePipeline().SampFreq());
            decoder.FeaturePipeline().AcceptWaveform(wave_data.SampFreq(),
                                                     wave_part);
            decoder.AdvanceFirstPass();
            samp_offset += this_num_samp;
          }
          decoder.SecondPass(); // note: you could call this at any point if
                                // you wanted the second-pass output.

          
        }  
    
    
    
    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    Matrix<BaseFloat> lda_transform;
    if (lda_mat_rspecifier != "") {
      bool binary_in;
      Input ki(lda_mat_rspecifier, &binary_in);
      lda_transform.Read(ki.Stream(), binary_in);
    }

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
        bool binary;
        Input ki(model_rspecifier, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_gmm.Read(ki.Stream(), binary);
    }

    fst::SymbolTable *word_syms = NULL;
    if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                    << word_syms_filename;

    fst::Fst<fst::StdArc> *decode_fst = ReadDecodeGraph(fst_rspecifier);

    // We are not properly registering/exposing MFCC and frame extraction options,
    // because there are parts of the online decoding code, where some of these
    // options are hardwired(ToDo: we should fix this at some point)
    MfccOptions mfcc_opts;
    mfcc_opts.use_energy = false;
    int32 frame_length = mfcc_opts.frame_opts.frame_length_ms = 25;
    int32 frame_shift = mfcc_opts.frame_opts.frame_shift_ms = 10;

    int32 window_size = right_context + left_context + 1;
    decoder_opts.batch_size = std::max(decoder_opts.batch_size, window_size);

    OnlineFasterDecoder decoder(*decode_fst, decoder_opts,
                                silence_phones, trans_model);
    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    VectorFst<LatticeArc> out_fst;
    for (; !reader.Done(); reader.Next()) {
      std::string wav_key = reader.Key();
      std::cerr << "File: " << wav_key << std::endl;
      const WaveData &wav_data = reader.Value();
      if(wav_data.SampFreq() != 16000)
        KALDI_ERR << "Sampling rates other than 16kHz are not supported!";
      int32 num_chan = wav_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << wav_key << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            continue;
          }
        }
      }
      OnlineVectorSource au_src(wav_data.Data().Row(this_chan));
      Mfcc mfcc(mfcc_opts);
      FeInput fe_input(&au_src, &mfcc,
                       frame_length*(wav_data.SampFreq()/1000),
                       frame_shift*(wav_data.SampFreq()/1000));
      OnlineCmnInput cmn_input(&fe_input, cmn_window, min_cmn_window);
      OnlineFeatInputItf *feat_transform = 0;
      if (lda_mat_rspecifier != "") {
        feat_transform = new OnlineLdaInput(
            &cmn_input, lda_transform,
            left_context, right_context);
      } else {
        DeltaFeaturesOptions opts;
        opts.order = kDeltaOrder;
        feat_transform = new OnlineDeltaInput(opts, &cmn_input);
      }

      // feature_reading_opts contains number of retries, batch size.
      OnlineFeatureMatrix feature_matrix(feature_reading_opts,
                                         feat_transform);

      OnlineDecodableDiagGmmScaled decodable(am_gmm, trans_model, acoustic_scale,
                                             &feature_matrix);
      int32 start_frame = 0;
      bool partial_res = false;
      while (1) {
        OnlineFasterDecoder::DecodeState dstate = decoder.Decode(&decodable);
        if (dstate & (decoder.kEndFeats | decoder.kEndUtt)) {
          std::vector<int32> word_ids;
          decoder.FinishTraceBack(&out_fst);
          fst::GetLinearSymbolSequence(out_fst,
                                       static_cast<vector<int32> *>(0),
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
          std::stringstream res_key;
          res_key << wav_key << '_' << start_frame << '-' << decoder.frame();
          if (!word_ids.empty())
            words_writer.Write(res_key.str(), word_ids);
          alignment_writer.Write(res_key.str(), tids);
          if (dstate == decoder.kEndFeats)
            break;
          start_frame = decoder.frame();
        } else {
          std::vector<int32> word_ids;
          if (decoder.PartialTraceback(&out_fst)) {
            fst::GetLinearSymbolSequence(out_fst,
                                        static_cast<vector<int32> *>(0),
                                        &word_ids,
                                        static_cast<LatticeArc::Weight*>(0));
            PrintPartialResult(word_ids, word_syms, false);
            if (!partial_res)
              partial_res = (word_ids.size() > 0);
          }
        }
      }
      if (feat_transform) delete feat_transform;
    }
    if (word_syms) delete word_syms;
    if (decode_fst) delete decode_fst;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
