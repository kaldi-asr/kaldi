// onlinebin/online-server-gmm-decode-faster.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)
//           2012 Vassil Panayotov
//           2013 Johns Hopkins University (author: Daniel Povey)

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
#include "online/online-feat-input.h"
#include "online/online-decodable.h"
#include "online/online-faster-decoder.h"
#include "online/onlinebin-util.h"

namespace kaldi {

void SendPartialResult(const std::vector<int32>& words,
                       const fst::SymbolTable *word_syms,
                       const bool line_break,
                       const int32 serv_sock,
                       const sockaddr_in &client_addr) {
  KALDI_ASSERT(word_syms != NULL);
  std::stringstream sstream;
  for (size_t i = 0; i < words.size(); i++) {
    std::string word = word_syms->Find(words[i]);
    if (word == "")
      KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
    sstream << word << ' ';
  }
  if (line_break)
    sstream << "\n\n";

  ssize_t sent = sendto(serv_sock, sstream.str().c_str(), sstream.str().size(),
                        0, reinterpret_cast<const sockaddr*>(&client_addr),
                        sizeof(client_addr));
  if (sent == -1)
    KALDI_WARN << "sendto() call failed when tried to send recognition results";
}

} // namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;

    // Up to delta-delta derivative features are calculated (unless LDA is used)
    const int32 kDeltaOrder = 2;

    const char *usage =
        "Decode speech, using feature batches received over a network connection\n\n"
        "Utterance segmentation is done on-the-fly.\n"
        "Feature splicing/LDA transform is used, if the optional(last) argument "
        "is given.\n"
        "Otherwise delta/delta-delta(2-nd order) features are produced.\n\n"
        "Usage: online-server-gmm-decode-faster [options] model-in"
        "fst-in word-symbol-table silence-phones udp-port [lda-matrix-in]\n\n"
        "Example: online-server-gmm-decode-faster --rt-min=0.3 --rt-max=0.5 "
        "--max-active=4000 --beam=12.0 --acoustic-scale=0.0769 "
        "model HCLG.fst words.txt '1:2:3:4:5' 1234 lda-matrix";
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    int32 cmn_window = 600,
      min_cmn_window = 100; // adds 1 second latency, only at utterance start.
    int32 right_context = 4, left_context = 4;

    kaldi::DeltaFeaturesOptions delta_opts;
    delta_opts.Register(&po);
    OnlineFasterDecoderOpts decoder_opts;
    OnlineFeatureMatrixOptions feature_reading_opts;
    decoder_opts.Register(&po, true);
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

    po.Read(argc, argv);
    if (po.NumArgs() != 5 && po.NumArgs() != 6) {
      po.PrintUsage();
      return 1;
    }

    std::string model_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        word_syms_filename = po.GetArg(3),
        silence_phones_str = po.GetArg(4),
        lda_mat_rspecifier = po.GetOptArg(6);
    int32 udp_port = atoi(po.GetArg(5).c_str());

    Matrix<BaseFloat> lda_transform;
    if (lda_mat_rspecifier != "") {
      bool binary_in;
      Input ki(lda_mat_rspecifier, &binary_in);
      lda_transform.Read(ki.Stream(), binary_in);
    }

    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
        KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    if (silence_phones.empty())
        KALDI_ERR << "No silence phones given!";

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
        bool binary;
        Input ki(model_rxfilename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_gmm.Read(ki.Stream(), binary);
    }

    fst::SymbolTable *word_syms = NULL;
    if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                    << word_syms_filename;

    fst::Fst<fst::StdArc> *decode_fst = ReadDecodeGraph(fst_rxfilename);

    // We are not properly registering/exposing MFCC and frame extraction options,
    // because there are parts of the online decoding code, where some of these
    // options are hardwired(ToDo: we should fix this at some point)
    MfccOptions mfcc_opts;
    mfcc_opts.use_energy = false;

    OnlineFasterDecoder decoder(*decode_fst, decoder_opts,
                                silence_phones, trans_model);
    VectorFst<LatticeArc> out_fst;
    int32 feature_dim = mfcc_opts.num_ceps; // default to 13 right now.
    OnlineUdpInput udp_input(udp_port, feature_dim);
    OnlineCmnInput cmn_input(&udp_input, cmn_window, min_cmn_window);
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

    std::cerr << std::endl << "Listening on UDP port "
              << udp_port << " ... " << std::endl;
    bool partial_res = false;
    while (1) {
      OnlineFasterDecoder::DecodeState dstate = decoder.Decode(&decodable);
      std::vector<int32> word_ids;
      if (dstate & (decoder.kEndFeats | decoder.kEndUtt)) {
        decoder.FinishTraceBack(&out_fst);
        fst::GetLinearSymbolSequence(out_fst,
                                     static_cast<vector<int32> *>(0),
                                     &word_ids,
                                     static_cast<LatticeArc::Weight*>(0));
        SendPartialResult(word_ids, word_syms, partial_res || word_ids.size(),
                          udp_input.descriptor(), udp_input.client_addr());
        partial_res = false;
      } else {
        if (decoder.PartialTraceback(&out_fst)) {
          fst::GetLinearSymbolSequence(out_fst,
                                       static_cast<vector<int32> *>(0),
                                       &word_ids,
                                       static_cast<LatticeArc::Weight*>(0));
          SendPartialResult(word_ids, word_syms, false,
                            udp_input.descriptor(), udp_input.client_addr());
          if (!partial_res)
            partial_res = (word_ids.size() > 0);
        }
      }
    }

    delete feat_transform;
    delete word_syms;
    delete decode_fst;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
