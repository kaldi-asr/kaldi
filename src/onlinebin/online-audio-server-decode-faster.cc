// onlinebin/online-audio-server-decode-faster.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)
// Copyright 2013 Polish-Japanese Institute of Information Technology (author: Danijel Korzinek)

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
#include "online/online-tcp-source.h"
#include "online/online-feat-input.h"
#include "online/online-decodable.h"
#include "online/online-faster-decoder.h"
#include "online/onlinebin-util.h"
#include "matrix/kaldi-vector.h"
#include "lat/word-align-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/sausages.h"
#include "lat/determinize-lattice-pruned.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctime>
#include <signal.h>

namespace kaldi {
/*
 * This class is for a very simple TCP server implementation
 * in UNIX sockets.
 */
class TcpServer {
 public:
  TcpServer();
  ~TcpServer();

  bool Listen(int32 port);  //start listening on a given port
  int32 Accept();  //accept a client and return its descriptor

 private:
  struct sockaddr_in h_addr_;
  int32 server_desc_;
};

//write a line of text to socket
bool WriteLine(int32 socket, std::string line);

//constant allowing to convert frame count to time
const float kFramesPerSecond = 100.0f;
}  // namespace kaldi

int32 main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace fst;

  try {
    typedef kaldi::int32 int32;
    typedef OnlineFeInput<Mfcc> FeInput;
    TcpServer tcp_server;
    signal(SIGPIPE, SIG_IGN);

    // up to delta-delta derivative features are calculated (unless LDA is used)
    const int32 kDeltaOrder = 2;

    const char *usage =
        "Starts a TCP server that receives RAW audio and outputs aligned words.\n"
            "A sample client can be found in: onlinebin/online-audio-client\n\n"
            "Usage: online-audio-server-decode-faster [options] model-in "
            "fst-in word-symbol-table silence-phones word_boundary_file tcp-port [lda-matrix-in]\n\n"
            "example: online-audio-server-decode-faster --verbose=1 --rt-min=0.5 --rt-max=3.0 --max-active=6000\n"
            "--beam=72.0 --acoustic-scale=0.0769 final.mdl graph/HCLG.fst graph/words.txt '1:2:3:4:5'\n"
            "graph/word_boundary.int 5000 final.mat\n\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    int32 cmn_window = 600, min_cmn_window = 100;  // adds 1 second latency, only at utterance start.
    int32 right_context = 4, left_context = 4;
    BaseFloat frame_shift = 0.01;

    OnlineFasterDecoderOpts decoder_opts;
    decoder_opts.Register(&po, true);
    OnlineFeatureMatrixOptions feature_reading_opts;
    feature_reading_opts.Register(&po);

    po.Register("left-context", &left_context,
                "Number of frames of left context");
    po.Register("right-context", &right_context,
                "Number of frames of right context");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register(
        "cmn-window", &cmn_window,
        "Number of feat. vectors used in the running average CMN calculation");
    po.Register("min-cmn-window", &min_cmn_window,
                "Minumum CMN window used at start of decoding (adds "
                "latency only at start)");
    po.Register("frame-shift", &frame_shift,
                "Time in seconds between frames.\n");

    WordBoundaryInfoNewOpts opts;
    opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() < 6 || po.NumArgs() > 7) {
      po.PrintUsage();
      return 1;
    }

    std::string model_rspecifier = po.GetArg(1), fst_rspecifier = po.GetArg(2),
        word_syms_filename = po.GetArg(3), silence_phones_str = po.GetArg(4),
        word_boundary_file = po.GetArg(5), lda_mat_rspecifier = "";

    if (po.NumArgs() == 7)
      lda_mat_rspecifier = po.GetOptArg(7);

    int32 port = strtol(po.GetArg(6).c_str(), 0, 10);

    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    if (silence_phones.empty())
      KALDI_ERR << "No silence phones given!";

    if (!tcp_server.Listen(port))
      return 0;

    std::cout << "Reading LDA matrix: " << lda_mat_rspecifier << "..."
        << std::endl;
    Matrix < BaseFloat > lda_transform;
    if (lda_mat_rspecifier != "") {
      bool binary_in;
      Input ki(lda_mat_rspecifier, &binary_in);
      lda_transform.Read(ki.Stream(), binary_in);
    }

    std::cout << "Reading acoustic model: " << model_rspecifier << "..."
        << std::endl;
    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rspecifier, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    std::cout << "Reading word list: " << word_syms_filename << "..."
        << std::endl;
    fst::SymbolTable *word_syms = NULL;
    if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
      KALDI_ERR << "Could not read symbol table from file "
          << word_syms_filename;

    std::cout << "Reading word boundary file: " << word_boundary_file << "..."
        << std::endl;
    WordBoundaryInfo info(opts, word_boundary_file);

    std::cout << "Reading FST: " << fst_rspecifier << "..." << std::endl;
    fst::Fst < fst::StdArc > *decode_fst = ReadDecodeGraph(fst_rspecifier);

    // We are not properly registering/exposing MFCC and frame extraction options,
    // because there are parts of the online decoding code, where some of these
    // options are hardwired(ToDo: we should fix this at some point)
    MfccOptions mfcc_opts;
    mfcc_opts.use_energy = false;
    int32 frame_length = mfcc_opts.frame_opts.frame_length_ms = 25;
    int32 mfcc_frame_shift = mfcc_opts.frame_opts.frame_shift_ms = 10;

    int32 window_size = right_context + left_context + 1;
    decoder_opts.batch_size = std::max(decoder_opts.batch_size, window_size);

    DeterminizeLatticePrunedOptions det_opts;
    det_opts.max_mem = 50000000;
    det_opts.max_loop = 0;

    VectorFst < LatticeArc > out_fst;
    Lattice out_lat;
    CompactLattice det_lat, aligned_lat;
    OnlineTcpVectorSource* au_src = NULL;
    int32 client_socket = -1;

    while (true) {
      if (au_src == NULL || !au_src->IsConnected()) {
        if (au_src) {
          std::cout << "Client disconnected!" << std::endl;
          delete au_src;
        }
        client_socket = tcp_server.Accept();
        au_src = new OnlineTcpVectorSource(client_socket);
      }

      //re-initalizing decoder for each utterance
      OnlineFasterDecoder decoder(*decode_fst, decoder_opts, silence_phones,
                                  trans_model);

      Mfcc mfcc(mfcc_opts);
      FeInput fe_input(au_src, &mfcc, frame_length * (16000 / 1000),
                       mfcc_frame_shift * (16000 / 1000));  //we always assume 16 kHz Fs on input
      OnlineCmnInput cmn_input(&fe_input, cmn_window, min_cmn_window);
      OnlineFeatInputItf *feat_transform = 0;
      if (lda_mat_rspecifier != "") {
        feat_transform = new OnlineLdaInput(&cmn_input, lda_transform,
                                            left_context, right_context);
      } else {
        DeltaFeaturesOptions opts;
        opts.order = kDeltaOrder;
        feat_transform = new OnlineDeltaInput(opts, &cmn_input);
      }

      // feature_reading_opts contains number of retries, batch size.
      OnlineFeatureMatrix feature_matrix(feature_reading_opts, feat_transform);

      OnlineDecodableDiagGmmScaled decodable(am_gmm, trans_model,
                                             acoustic_scale, &feature_matrix);

      clock_t start = clock();
      int32 decoder_offset = 0;

      while (1) {
        if (!au_src->IsConnected())
          break;

        OnlineFasterDecoder::DecodeState dstate = decoder.Decode(&decodable);

        if (!au_src->IsConnected()) {
          break;
        }

        if (dstate & (decoder.kEndFeats | decoder.kEndUtt)) {
          std::vector<int32> word_ids, times, lengths;

          decoder.FinishTraceBack(&out_fst);
          decoder.GetBestPath(&out_fst);

          ConvertLattice(out_fst, &out_lat);

          Invert(&out_lat);
          //TopSort(&out_lat);
          //ArcSort(&out_lat, ILabelCompare<LatticeArc>());

          DeterminizeLatticePruned(out_lat, 10.0f, &det_lat, det_opts);

          WordAlignLattice(det_lat, trans_model, info, 0, &aligned_lat);

          CompactLatticeToWordAlignment(aligned_lat, &word_ids, &times,
                                        &lengths);

          //count number of non-sil words
          int32 words_num = 0;
          for (size_t i = 0; i < word_ids.size(); i++)
            if (word_ids[i] != 0)
              words_num++;

          if (words_num > 0) {

            float dur = (clock() - start) / (float) CLOCKS_PER_SEC;
            float input_dur = au_src->SamplesProcessed() / 16000.0;

            start = clock();
            au_src->ResetSamples();

            std::stringstream sstr;
            sstr << "RESULT:NUM=" << words_num << ",FORMAT=WSE,RECO-DUR=" << dur
                << ",INPUT-DUR=" << input_dur;

            WriteLine(client_socket, sstr.str());

            for (size_t i = 0; i < word_ids.size(); i++) {
              if (word_ids[i] == 0)
                continue;  //skip silences...

              std::string word = word_syms->Find(word_ids[i]);
              if (word.empty())
                word = "???";

              float start = (times[i] + decoder_offset) / kFramesPerSecond;
              float len = lengths[i] / kFramesPerSecond;

              std::stringstream wstr;
              wstr << word << "," << start << "," << (start + len);

              WriteLine(client_socket, wstr.str());
            }
          }

          if (dstate == decoder.kEndFeats) {
            WriteLine(client_socket, "RESULT:DONE");
            break;
          }

          decoder_offset = decoder.frame();
        } else {
          std::vector<int32> word_ids;
          if (decoder.PartialTraceback(&out_fst)) {
            GetLinearSymbolSequence(out_fst, static_cast<vector<int32> *>(0),
                                    &word_ids,
                                    static_cast<LatticeArc::Weight*>(0));
            for (size_t i = 0; i < word_ids.size(); i++) {
              if (word_ids[i] != 0) {
                WriteLine(client_socket,
                          "PARTIAL:" + word_syms->Find(word_ids[i]));
              }
            }
          }
        }
      }
      delete feat_transform;
    }

    std::cout << "Deinitizalizing..." << std::endl;

    delete word_syms;
    delete decode_fst;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}  // main()

namespace kaldi {
// IMPLEMENTATION OF THE CLASSES/METHODS ABOVE MAIN
TcpServer::TcpServer() {
  server_desc_ = -1;
}

bool TcpServer::Listen(int32 port) {
  h_addr_.sin_addr.s_addr = INADDR_ANY;
  h_addr_.sin_port = htons(port);
  h_addr_.sin_family = AF_INET;

  server_desc_ = socket(AF_INET, SOCK_STREAM, 0);

  if (server_desc_ == -1) {
    KALDI_ERR << "Cannot create TCP socket!";
    return false;
  }

  int32 flag = 1;
  int32 len = sizeof(int32);
  if( setsockopt(server_desc_, SOL_SOCKET, SO_REUSEADDR, &flag, len) == -1){
    KALDI_ERR << "Cannot set socket options!\n";
    return false;
  }

  if (bind(server_desc_, (struct sockaddr*) &h_addr_, sizeof(h_addr_)) == -1) {
    KALDI_ERR << "Cannot bind to port: " << port << " (is it taken?)";
    return false;
  }

  if (listen(server_desc_, 1) == -1) {
    KALDI_ERR << "Cannot listen on port!";
    return false;
  }

  std::cout << "TcpServer: Listening on port: " << port << std::endl;

  return true;

}

TcpServer::~TcpServer() {
  if (server_desc_ != -1)
    close(server_desc_);
}

int32 TcpServer::Accept() {
  std::cout << "Waiting for client..." << std::endl;

  socklen_t len;

  len = sizeof(struct sockaddr);
  int32 client_desc = accept(server_desc_, (struct sockaddr*) &h_addr_, &len);

  struct sockaddr_storage addr;
  char ipstr[20];

  len = sizeof addr;
  getpeername(client_desc, (struct sockaddr*) &addr, &len);

  struct sockaddr_in *s = (struct sockaddr_in *) &addr;
  inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);

  std::cout << "TcpServer: Accepted connection from: " << ipstr << std::endl;

  return client_desc;
}

bool WriteLine(int32 socket, std::string line) {
  line = line + "\n";

  const char* p = line.c_str();
  int32 to_write = line.size();
  int32 wrote = 0;
  while (to_write > 0) {
    int32 ret = write(socket, p + wrote, to_write);
    if (ret <= 0)
      return false;

    to_write -= ret;
    wrote += ret;
  }

  return true;
}
}  // namespace kaldi
