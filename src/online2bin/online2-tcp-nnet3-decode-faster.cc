// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)
//           2018  Polish-Japanese Academy of Information Technology (Author: Danijel Korzinek)

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

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <poll.h>
#include <signal.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>

namespace kaldi {

class TcpServer {
 public:
  TcpServer(int read_timeout);
  ~TcpServer();

  bool Listen(int32 port);  //start listening on a given port
  int32 Accept();  //accept a client and return its descriptor

  bool ReadChunk(size_t len); //get more data and return false if end-of-stream

  Vector<BaseFloat> GetChunk(); //get the data read by above method

  bool Write(std::string msg); //write to accepted client

  void Disconnect();

 private:
  struct ::sockaddr_in h_addr_;
  int32 server_desc_, client_desc_;
  int16 *samp_buf_;
  size_t buf_len_, has_read_;
  pollfd client_set_[1];
  int read_timeout_;
};

std::string LatticeToString(const Lattice &lat, fst::SymbolTable *word_syms) {
  LatticeWeight weight;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(lat, &alignment, &words, &weight);

  std::string msg = "";
  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms->Find(words[i]);
    if (s == "") {
      KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      msg += "<#" + std::to_string(i) + "> ";
    } else
      msg += s + " ";
  }
  return msg;
}

std::string LatticeToString(const CompactLattice &clat, fst::SymbolTable *word_syms) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return "";
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);
  return LatticeToString(best_path_lat, word_syms);
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in audio from a network socket and performs online decoding with neural nets\n"
        "(nnet3 setup), with iVector-based speaker adaptation and\n"
        "endpointing.  Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-net-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
        "<word-symbol-table> <listen-port>\n";

    ParseOptions po(usage);


    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = 0.18;
    BaseFloat output_freq = 1;
    BaseFloat samp_freq = 16000.0;
    int read_timeout = 3;
    bool adapt_speaker = false;

    po.Register("samp-freq", &samp_freq,
                "Sampling frequency of the input signal (coded as 16-bit slinear).");
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");
    po.Register("output-freq", &output_freq,
                "How often in seconds, do we check for changes in output.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");
    po.Register("adapt-spk", &adapt_speaker,
                "Adapt to a single speaker. Otherwise, treat each segment as a new speaker.");
    po.Register("read-timeout", &read_timeout,
                "Number of seconds of timout for TCP audio data to appear on the stream. Use -1 for blocking.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        word_syms_rxfilename = po.GetArg(3),
        port_rspecifier = po.GetArg(4);

    int port_num = std::stoi(port_rspecifier);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    KALDI_LOG << "Loading AM...";

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

    KALDI_LOG << "Loading FST...";

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (!word_syms_rxfilename.empty())
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    KALDI_LOG << "Loaded evertyhing. Waiting for data...";

    signal(SIGPIPE, SIG_IGN); //ignore SIGPIPE to avoid crashing when socket forecfully disconnected

    TcpServer server(read_timeout);

    server.Listen(port_num);

    while (true) {

      server.Accept();

      int32 samp_count = 0;//this is used for output refresh rate
      size_t chunk_len = static_cast<size_t>(chunk_length_secs * samp_freq);

      int32 check_update = static_cast<int32>(samp_freq * output_freq);
      int32 next_check = check_update;

      int32 frame_offset = 0;

      bool eos = false;

      OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);

      OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
      SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                          decodable_info,
                                          *decode_fst, &feature_pipeline);

      while (!eos) {

        decoder.InitDecoding(frame_offset);

//        if (!adapt_speaker)
//          feature_pipeline.SetAdaptationState(adaptation_state); //reset adaptation state to intital one

        OnlineSilenceWeighting silence_weighting(
            trans_model,
            feature_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor);

        std::vector<std::pair<int32, BaseFloat>> delta_weights;

        int32 samp_pipeline = 0;//this is used to figure out the offset for the remainder

        while (true) {

          eos = !server.ReadChunk(chunk_len);

          if (!eos) {
            Vector<BaseFloat> wave_part = server.GetChunk();
            feature_pipeline.AcceptWaveform(samp_freq, wave_part);
            samp_pipeline += wave_part.Dim();
            samp_count += chunk_len;

            if (silence_weighting.Active() &&
                feature_pipeline.IvectorFeature() != NULL) {
              feature_pipeline.UpdateFrameWeights(delta_weights,
                                                  frame_offset * decodable_opts.frame_subsampling_factor);
            }

            decoder.AdvanceDecoding();

            if (samp_count > next_check) {
              if (decoder.NumFramesDecoded() > 0) {
                Lattice lat;
                decoder.GetBestPath(false, &lat);

                std::string msg = LatticeToString(lat, word_syms);

                server.Write(msg + '\r');
              }

              next_check += check_update;
            }

          } else {
            feature_pipeline.InputFinished();
            decoder.AdvanceDecoding();

            decoder.FinalizeDecoding();
            frame_offset += decoder.NumFramesDecoded();

            if (decoder.NumFramesDecoded() > 0) {
              CompactLattice lat;
              decoder.GetLattice(true, &lat);

              std::string msg = LatticeToString(lat, word_syms);

              server.Write(msg + "\n");
            } else
              server.Write("\n");

            server.Disconnect();
            break;

          }

          if (decoder.EndpointDetected(endpoint_opts)) {

            decoder.FinalizeDecoding();
            frame_offset += decoder.NumFramesDecoded();

            CompactLattice lat;
            decoder.GetLattice(true, &lat);
            std::string msg = LatticeToString(lat, word_syms);

            server.Write(msg + "\n");
            break;
          }
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
} // main()


namespace kaldi {
TcpServer::TcpServer(int read_timeout) {
  server_desc_ = -1;
  client_desc_ = -1;
  samp_buf_ = NULL;
  buf_len_ = 0;
  read_timeout_ = 1000 * read_timeout;
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
  if (setsockopt(server_desc_, SOL_SOCKET, SO_REUSEADDR, &flag, len) == -1) {
    KALDI_ERR << "Cannot set socket options!\n";
    return false;
  }

  if (bind(server_desc_, (struct sockaddr *) &h_addr_, sizeof(h_addr_)) == -1) {
    KALDI_ERR << "Cannot bind to port: " << port << " (is it taken?)";
    return false;
  }

  if (listen(server_desc_, 1) == -1) {
    KALDI_ERR << "Cannot listen on port!";
    return false;
  }

  KALDI_LOG << "TcpServer: Listening on port: " << port << std::endl;

  return true;

}

TcpServer::~TcpServer() {
  Disconnect();
  if (server_desc_ != -1)
    close(server_desc_);
  delete[] samp_buf_;
}

int32 TcpServer::Accept() {
  KALDI_LOG << "Waiting for client..." << std::endl;

  socklen_t len;

  len = sizeof(struct sockaddr);
  client_desc_ = accept(server_desc_, (struct sockaddr *) &h_addr_, &len);

  struct sockaddr_storage addr;
  char ipstr[20];

  len = sizeof addr;
  getpeername(client_desc_, (struct sockaddr *) &addr, &len);

  struct sockaddr_in *s = (struct sockaddr_in *) &addr;
  inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);

  client_set_[0].fd = client_desc_;
  client_set_[0].events = POLLIN;

  KALDI_LOG << "Accepted connection from: " << ipstr << std::endl;

  return client_desc_;
}

bool TcpServer::ReadChunk(size_t len) {
  if (buf_len_ != len) {
    buf_len_ = len;
    delete[] samp_buf_;
    samp_buf_ = new int16[len];
  }

  ssize_t ret;
  int poll_ret;
  size_t to_read = len;
  has_read_ = 0;
  while (to_read > 0) {
    poll_ret = poll(client_set_, 1, read_timeout_);
    if (poll_ret == 0) {
      KALDI_WARN << "Socket timeout! Disconnecting..." << std::endl;
      break;
    }
    if (client_set_[0].revents != POLLIN) {
      KALDI_WARN << "Socket error! Disconnecting..." << std::endl;
      break;
    }
    ret = read(client_desc_, static_cast<void *>(samp_buf_ + has_read_), to_read * sizeof(int16));
    if (ret <= 0) {
      KALDI_WARN << "Stream over..." << std::endl;
      break;
    }
    to_read -= ret / sizeof(int16);
    has_read_ += ret / sizeof(int16);
  }

  return has_read_ == len;
}

Vector<BaseFloat> TcpServer::GetChunk() {
  Vector<BaseFloat> buf;

  buf.Resize(static_cast<MatrixIndexT>(has_read_));

  for (size_t i = 0; i < has_read_; i++)
    buf(i) = static_cast<BaseFloat>(samp_buf_[i]);

  return buf;
}

bool TcpServer::Write(std::string msg) {

  const char *p = msg.c_str();
  size_t to_write = msg.size();
  size_t wrote = 0;
  while (to_write > 0) {
    ssize_t ret = write(client_desc_, static_cast<const void *>(p + wrote), to_write);
    if (ret <= 0)
      return false;

    to_write -= ret;
    wrote += ret;
  }

  return true;
}

void TcpServer::Disconnect() {
  if (client_desc_ != -1) {
    close(client_desc_);
    client_desc_ = -1;
  }
}
}  // namespace kaldi