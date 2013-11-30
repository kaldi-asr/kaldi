// onlinebin/online-wav-gmm-decode-faster.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "online/online-audio-source.h"
#include "online/online-feat-input.h"
#include "online/online-decodable.h"
#include "online/online-faster-decoder.h"
#include "online/onlinebin-util.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef OnlineFeInput<Mfcc> FeInput;

    // up to delta-delta derivative features are calculated (unless LDA is used)
    const int32 kDeltaOrder = 2;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding.\n"
        "Writes .tra and .ali files for WER computation. Utterance "
        "segmentation is done on-the-fly.\n"
        "Feature splicing/LDA transform is used, if the optional(last) argument "
        "is given.\n"
        "Otherwise delta/delta-delta(i.e. 2-nd order) features are produced.\n"
        "Caution: the last few frames of the wav file may not be decoded properly.\n"
        "Hence, don't use one wav file per utterance, but "
        "rather use one wav file per show.\n\n"
        "Usage: online-wav-gmm-decode-faster [options] wav-rspecifier model-in"
        "fst-in word-symbol-table silence-phones transcript-wspecifier "
        "alignments-wspecifier [lda-matrix-in]\n\n"
        "Example: ./online-wav-gmm-decode-faster --rt-min=0.3 --rt-max=0.5 "
        "--max-active=4000 --beam=12.0 --acoustic-scale=0.0769 "
        "scp:wav.scp model HCLG.fst words.txt '1:2:3:4:5' ark,t:trans.txt ark,t:ali.txt";
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    int32 cmn_window = 600,
      min_cmn_window = 100; // adds 1 second latency, only at utterance start.
    int32 channel = -1;
    int32 right_context = 4, left_context = 4;

    OnlineFasterDecoderOpts decoder_opts;
    decoder_opts.Register(&po, true);
    OnlineFeatureMatrixOptions feature_reading_opts;
    feature_reading_opts.Register(&po);
    
    po.Register("left-context", &left_context, "Number of frames of left context");
    po.Register("right-context", &right_context, "Number of frames of right context");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("cmn-window", &cmn_window,
        "Number of feat. vectors used in the running average CMN calculation");
    po.Register("min-cmn-window", &min_cmn_window,
                "Minumum CMN window used at start of decoding (adds "
                "latency only at start)");
    po.Register("channel", &channel,
        "Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)");
    po.Read(argc, argv);
    if (po.NumArgs() != 7 && po.NumArgs() != 8) {
      po.PrintUsage();
      return 1;
    }
    
    std::string wav_rspecifier = po.GetArg(1),
        model_rspecifier = po.GetArg(2),
        fst_rspecifier = po.GetArg(3),
        word_syms_filename = po.GetArg(4),
        silence_phones_str = po.GetArg(5),
        words_wspecifier = po.GetArg(6),
        alignment_wspecifier = po.GetArg(7),
        lda_mat_rspecifier = po.GetOptArg(8);

    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
        KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    if (silence_phones.empty())
        KALDI_ERR << "No silence phones given!";

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
