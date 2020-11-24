// online2bin/online2-tcp-nnet3-decode-faster-reorganized.cc
// This a copy of online2bin/online2-tcp-nnet3-decode-faster, where the online
// decoder has been extracted from the main function and put in a class for
// build to Web Assembly with the emscripten toolchain.

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
  explicit TcpServer(int read_timeout);
  ~TcpServer();

  bool Listen(int32 port);  // start listening on a given port
  int32 Accept();  // accept a client and return its descriptor

  bool ReadChunk(size_t len); // get more data and return false if end-of-stream

  Vector<BaseFloat> GetChunk(); // get the data read by above method

  bool Write(const std::string &msg); // write to accepted client
  bool WriteLn(const std::string &msg, const std::string &eol = "\n"); // write line to accepted client

  void Disconnect();

 private:
  struct ::sockaddr_in h_addr_;
  int32 server_desc_, client_desc_;
  int16 *samp_buf_;
  size_t buf_len_, has_read_;
  pollfd client_set_[1];
  int read_timeout_;
};

std::string LatticeToString(const Lattice &lat, const fst::SymbolTable &word_syms) {
  LatticeWeight weight;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(lat, &alignment, &words, &weight);

  std::ostringstream msg;
  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms.Find(words[i]);
    if (s.empty()) {
      KALDI_WARN << "Word-id " << words[i] << " not in symbol table.";
      msg << "<#" << std::to_string(i) << "> ";
    } else
      msg << s << " ";
  }
  return msg.str();
}

std::string GetTimeString(int32 t_beg, int32 t_end, BaseFloat time_unit) {
  char buffer[100];
  double t_beg2 = t_beg * time_unit;
  double t_end2 = t_end * time_unit;
  snprintf(buffer, 100, "%.2f %.2f", t_beg2, t_end2);
  return std::string(buffer);
}

int32 GetLatticeTimeSpan(const Lattice& lat) {
  std::vector<int32> times;
  LatticeStateTimes(lat, &times);
  return times.back();
}

std::string LatticeToString(const CompactLattice &clat, const fst::SymbolTable &word_syms) {
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

struct OnlineASROptionParser: public ParseOptions {
  OnlineASROptionParser();
  explicit OnlineASROptionParser(int argc, const char* const* argv);
  int Read(int, const char* const*);

  // Members
  static constexpr const char *usage =
    "Reads in audio from a network socket and performs online\n"
    "decoding with neural nets (nnet3 setup), with iVector-based\n"
    "speaker adaptation and endpointing.\n"
    "Note: some configuration values and inputs are set via config\n"
    "files whose filenames are passed as options\n"
    "\n"
    "Usage: online2-tcp-nnet3-decode-faster [options] <nnet3-in> "
    "<fst-in> <word-symbol-table>\n";
  // ASR stuff
  BaseFloat output_period = 1;
  bool produce_time = false;
  BaseFloat samp_freq = 16000.0;
  OnlineEndpointConfig endpoint_opts;
  OnlineNnet2FeaturePipelineConfig feature_opts;
  nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
  LatticeFasterDecoderConfig decoder_opts;
  std::string nnet3_rxfilename;
  std::string fst_rxfilename;
  std::string word_syms_filename;
  // TCP stuff
  BaseFloat chunk_length_secs = 0.18;
  int port_num = 5050;
  int read_timeout = 3;
};

class OnlineASR {
 public:
  static constexpr const char eou {'\n'};
  static constexpr const char tmp_eou {'\r'};

  explicit OnlineASR(int argc, const char *const argv[]);
  explicit OnlineASR(const std::vector<std::string> &args);
  explicit OnlineASR(const OnlineASROptionParser& po);
  std::string ProcessBuffer(int16 *, size_t);
  std::string ProcessSTLVector(const std::vector<int16>&);
  std::string ProcessVector(const Vector<BaseFloat>&);
  std::string Reset();
  ~OnlineASR();

 private:
  BaseFloat samp_freq;
  int32 frame_offset {0};
  int32 check_period;
  int32 samp_count {0};
  bool produce_time {false};
  // Model related members
  nnet3::DecodableNnetSimpleLoopedInfo *decodable_info = nullptr;
  OnlineNnet2FeaturePipelineInfo *feature_info = nullptr;
  LatticeFasterDecoderConfig decoder_opts;
  nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
  fst::Fst<fst::StdArc> *decode_fst = nullptr;
  TransitionModel trans_model;
  nnet3::AmNnetSimple am_nnet;
  fst::SymbolTable *word_syms = nullptr;
  OnlineEndpointConfig endpoint_opts;
  // Stream parameters
  OnlineNnet2FeaturePipeline *feature_pipeline = nullptr;
  SingleUtteranceNnet3Decoder *decoder = nullptr;
  // Utterance parameters
  OnlineSilenceWeighting *silence_weighting = nullptr;
  std::vector<std::pair<int32, BaseFloat> > delta_weights;

  // private methods
  void InitClass(const OnlineASROptionParser& parser);
  void InitWords(const std::string& filename);
  void UpdateDecoder(const Vector<BaseFloat>&);
  std::string CheckDecoderOutput();
  std::string PrependTimestamps(const std::string&);
  void ResetStreamDecoder();
  void ResetUtteranceDecoder();
};
}  // namespace kaldi

#ifndef __EMSCRIPTEN__

int main(int argc, const char* const* argv) {
  using kaldi::int32;
  using kaldi::int64;
  using kaldi::OnlineASR;
  using kaldi::OnlineASROptionParser;
  using kaldi::Vector;
  using kaldi::BaseFloat;
  using kaldi::TcpServer;

  OnlineASROptionParser po;

  try {
    po.Read(argc, argv);
    OnlineASR onlineASR(po);

    // ignore SIGPIPE to avoid crashing when socket forcefully disconnected
    signal(SIGPIPE, SIG_IGN);

    size_t chunk_len = static_cast<size_t>(po.chunk_length_secs * po.samp_freq);
    TcpServer server(po.read_timeout);
    server.Listen(po.port_num);

    while (true) {
      server.Accept();
      bool eos {false};

      while (!eos) {
        while (true) {
          eos = !server.ReadChunk(chunk_len);
          if (eos) {
            std::string msg { onlineASR.Reset() };
            KALDI_VLOG(1) << "EndOfAudio, sending message: " << msg;
            server.Write(msg);
            server.Disconnect();
            break;
          }
          Vector<BaseFloat> wave_part = server.GetChunk();
          std::string msg { onlineASR.ProcessVector(wave_part) };
          if (msg != "") {
            server.Write(msg);
            if (msg[msg.length() - 1] == onlineASR.tmp_eou) {
              KALDI_VLOG(1) << "Temporary transcript: " << msg;
            } else {
              KALDI_VLOG(1) << "Endpoint, sending message: " << msg;
              break;
            }
          }
        }
      }
    }
  } catch (const std::invalid_argument& e) {
    po.PrintUsage();
    return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
#else

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <iterator>

using std::vector;
using kaldi::int16;
using emscripten::val;
using emscripten::class_;
using emscripten::optional_override;
using emscripten::register_vector;

/* Convert JS Int16Array to C++ std::vector<kaldi::int16> without copy of data
*/
vector<int16> typed_array_to_vector(const val &int16_array) {
  unsigned int length = int16_array["length"].as<unsigned int>();
  vector<int16> vec(length);

  val memory = val::module_property("HEAP16")["buffer"];
  val memoryView = val::global("Int16Array").new_(memory,
      reinterpret_cast<std::uintptr_t>(vec.data()), length);

  memoryView.call<void>("set", int16_array);

  return vec;
}

EMSCRIPTEN_BINDINGS(asr) {
  class_<kaldi::OnlineASR>("OnlineASR")
    .constructor<const std::vector<std::string>& >()
    // Inject lambda before class method call to adapt I/O types
    .function("processBuffer", optional_override(
      [](kaldi::OnlineASR& self, const val& int16_array) {
        vector<int16> vect_array = typed_array_to_vector(int16_array);
        return self.ProcessSTLVector(vect_array);
      })
    )
    .function("reset", &kaldi::OnlineASR::Reset);
  // Define JS class StringList to be understood as vector<string> in C++
  register_vector<std::string>("StringList");
}

#endif

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
    KALDI_ERR << "Cannot set socket options!";
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

  KALDI_LOG << "TcpServer: Listening on port: " << port;

  return true;
}

TcpServer::~TcpServer() {
  Disconnect();
  if (server_desc_ != -1)
    close(server_desc_);
  delete[] samp_buf_;
}

int32 TcpServer::Accept() {
  KALDI_LOG << "Waiting for client...";

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

  KALDI_LOG << "Accepted connection from: " << ipstr;

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
  char *samp_buf_p = reinterpret_cast<char *>(samp_buf_);
  size_t to_read = len * sizeof(int16);
  has_read_ = 0;
  while (to_read > 0) {
    poll_ret = poll(client_set_, 1, read_timeout_);
    if (poll_ret == 0) {
      KALDI_WARN << "Socket timeout! Disconnecting..." << "(has_read_ = " << has_read_ << ")";
      break;
    }
    if (poll_ret < 0) {
      KALDI_WARN << "Socket error! Disconnecting...";
      break;
    }
    ret = read(client_desc_, static_cast<void *>(samp_buf_p + has_read_), to_read);
    if (ret <= 0) {
      KALDI_WARN << "Stream over...";
      break;
    }
    to_read -= ret;
    has_read_ += ret;
  }
  has_read_ /= sizeof(int16);

  return has_read_ > 0;
}

Vector<BaseFloat> TcpServer::GetChunk() {
  Vector<BaseFloat> buf;

  buf.Resize(static_cast<MatrixIndexT>(has_read_));

  for (int i = 0; i < has_read_; i++)
    buf(i) = static_cast<BaseFloat>(samp_buf_[i]);

  return buf;
}

bool TcpServer::Write(const std::string &msg) {
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

bool TcpServer::WriteLn(const std::string &msg, const std::string &eol) {
  if (Write(msg))
    return Write(eol);
  else return false;
}

void TcpServer::Disconnect() {
  if (client_desc_ != -1) {
    close(client_desc_);
    client_desc_ = -1;
  }
}

OnlineASROptionParser::OnlineASROptionParser(): ParseOptions{usage} {
  g_num_threads = 0;

  Register("samp-freq", &samp_freq,
           "Sampling frequency of the input signal (coded as 16-bit slinear).");
  Register("chunk-length", &chunk_length_secs,
           "Length of chunk size in seconds, that we process.");
  Register("output-period", &output_period,
           "How often in seconds, do we check for changes in output.");
  Register("num-threads-startup", &g_num_threads,
           "Number of threads used when initializing iVector extractor.");
  Register("read-timeout", &read_timeout,
           "Number of seconds of timeout for TCP audio data to appear on the "
           "stream. Use -1 for blocking.");
  Register("port-num", &port_num,
           "Portnumber the server will listen on.");
  Register("produce-time", &produce_time,
           "Prepend begin/end times between endpoints (e.g. '5.46 6.81"
           " <text_output>', in seconds)");

  endpoint_opts.Register(this);
  feature_opts.Register(this);
  decodable_opts.Register(this);
  decoder_opts.Register(this);
}

OnlineASROptionParser::OnlineASROptionParser(int argc,
                                             const char* const * argv):
  OnlineASROptionParser() {
    Read(argc, argv);
  }

int OnlineASROptionParser::Read(int argc, const char* const* argv) {
  int read_value = ParseOptions::Read(argc, argv);
  if (NumArgs() != 3)
    throw std::invalid_argument("Wrong number of arguments\n");

  nnet3_rxfilename = GetArg(1);
  fst_rxfilename = GetArg(2);
  word_syms_filename = GetArg(3);

  return read_value;
}

OnlineASR::OnlineASR(int argc, const char *const argv[]):
  OnlineASR(OnlineASROptionParser(argc, argv)) {
  }

OnlineASR::OnlineASR(const std::vector<std::string> &args) {
  // Convert args to const char* const *
  std::vector<const char*> char_array;
  char_array.reserve(args.size());
  for (int i = 0; i < args.size(); ++i)
    char_array.push_back(const_cast<char*>(args[i].c_str()));

  int argc { static_cast<int>(char_array.size()) };
  OnlineASROptionParser parser {argc, &char_array[0]};
  InitClass(parser);
}

OnlineASR::OnlineASR(const OnlineASROptionParser& parser) {
  InitClass(parser);
}

void OnlineASR::InitClass(const OnlineASROptionParser& parser) {
  decodable_opts = parser.decodable_opts;
  decoder_opts = parser.decoder_opts;
  endpoint_opts = parser.endpoint_opts;
  samp_freq = parser.samp_freq;
  check_period = static_cast<int32>(samp_freq * parser.output_period);
  produce_time = parser.produce_time;

  feature_info = new OnlineNnet2FeaturePipelineInfo(parser.feature_opts);
  InitWords(parser.word_syms_filename);

  KALDI_VLOG(1) << "Loading AM...";
  {
    bool binary;
    Input ki(parser.nnet3_rxfilename, &binary);
    trans_model.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
    SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
    SetDropoutTestMode(true, &(am_nnet.GetNnet()));
    nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
  }

  KALDI_VLOG(1) << "Loading FST...";
  decode_fst = fst::ReadFstKaldiGeneric(parser.fst_rxfilename);
  // this object contains precomputed stuff that is used by all decodable
  // objects.  It takes a pointer to am_nnet because if it has iVectors it has
  // to modify the nnet to accept iVectors at intervals.
  decodable_info = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts,
      &am_nnet);

  ResetStreamDecoder();
}

void OnlineASR::ResetStreamDecoder() {
  frame_offset = 0;

  delete feature_pipeline;
  feature_pipeline = new OnlineNnet2FeaturePipeline(*feature_info);

  delete decoder;
  decoder = new SingleUtteranceNnet3Decoder(decoder_opts, trans_model,
                                            *decodable_info, *decode_fst,
                                            feature_pipeline);
  ResetUtteranceDecoder();
}

void OnlineASR::InitWords(const std::string &filename) {
  if (!filename.empty())
    if (!(word_syms = fst::SymbolTable::ReadText(filename)))
      KALDI_ERR << "Could not read symbol table from file "
                << filename;
}

void OnlineASR::ResetUtteranceDecoder() {
  decoder->InitDecoding(frame_offset);
  delete silence_weighting;
  silence_weighting = new OnlineSilenceWeighting(
      trans_model,
      feature_info->silence_weighting_config,
      decodable_opts.frame_subsampling_factor);
  delta_weights = std::vector<std::pair<int32, BaseFloat> >();
}

std::string OnlineASR::ProcessBuffer(int16 *samp_buf, size_t buf_len) {
  Vector<BaseFloat> buf;
  buf.Resize(static_cast<MatrixIndexT>(buf_len));
  for (int i = 0; i < buf_len; ++i)
    buf(i) = static_cast<BaseFloat>(samp_buf[i]);

  return ProcessVector(buf);
}

std::string OnlineASR::ProcessVector(const Vector<BaseFloat>& buf) {
  UpdateDecoder(buf);
  return CheckDecoderOutput();
}

void OnlineASR::UpdateDecoder(const Vector<BaseFloat>& buf) {
  feature_pipeline->AcceptWaveform(samp_freq, buf);
  samp_count += buf.Dim();

  if (silence_weighting->Active() &&
      feature_pipeline->IvectorFeature() != NULL) {
    silence_weighting->ComputeCurrentTraceback(decoder->Decoder());
    silence_weighting->GetDeltaWeights(feature_pipeline->NumFramesReady(),
                                       frame_offset * decodable_opts.frame_subsampling_factor,
                                       &delta_weights);
    feature_pipeline->UpdateFrameWeights(delta_weights);
  }

  decoder->AdvanceDecoding();
}

std::string OnlineASR::CheckDecoderOutput() {
  if (decoder->EndpointDetected(endpoint_opts)) {
    samp_count %= check_period;
    decoder->FinalizeDecoding();
    frame_offset += decoder->NumFramesDecoded();
    CompactLattice lat;
    decoder->GetLattice(true, &lat);
    std::string msg = LatticeToString(lat, *word_syms);
    if (produce_time) msg = PrependTimestamps(msg);

    ResetUtteranceDecoder();
    return msg + eou;
  }

  // Force temporary result
  if (samp_count > check_period) {
    samp_count %= check_period;
    if (decoder->NumFramesDecoded() > 0) {
      Lattice lat;
      decoder->GetBestPath(false, &lat);
      TopSort(&lat);  // for LatticeStateTimes(),
      std::string msg = LatticeToString(lat, *word_syms);

      if (produce_time) {
        int32 frame_subsampling { decodable_opts.frame_subsampling_factor };
        BaseFloat frame_shift { feature_info->FrameShiftInSeconds() };
        int32 t_beg = frame_offset;
        int32 t_end = frame_offset + GetLatticeTimeSpan(lat);
        msg = GetTimeString(t_beg, t_end, frame_shift * frame_subsampling) + " "
              + msg;
      }
      return msg + tmp_eou;
    }
  }

  return "";
}

std::string OnlineASR::PrependTimestamps(const std::string& msg) {
  int32 frame_subsampling { decodable_opts.frame_subsampling_factor };
  BaseFloat frame_shift { feature_info->FrameShiftInSeconds() };
  int32 t_beg = frame_offset - decoder->NumFramesDecoded();
  int32 t_end = frame_offset;
  return GetTimeString(t_beg, t_end, frame_shift * frame_subsampling) + " "
         + msg;
}

std::string OnlineASR::ProcessSTLVector(const std::vector<int16>& samp_buf) {
  // cast input to float
  Vector<BaseFloat> buf;
  size_t buf_len = samp_buf.size();
  buf.Resize(static_cast<MatrixIndexT>(buf_len));
  for (int i = 0; i < buf_len; ++i)
    buf(i) = static_cast<BaseFloat>(samp_buf[i]);

  return ProcessVector(buf);
}

std::string OnlineASR::Reset() {
  feature_pipeline->InputFinished();

  if (silence_weighting->Active() &&
      feature_pipeline->IvectorFeature() != NULL) {
    silence_weighting->ComputeCurrentTraceback(decoder->Decoder());
    silence_weighting->GetDeltaWeights(feature_pipeline->NumFramesReady(),
                                       frame_offset * decodable_opts.frame_subsampling_factor,
                                       &delta_weights);
    feature_pipeline->UpdateFrameWeights(delta_weights);
  }

  decoder->AdvanceDecoding();
  decoder->FinalizeDecoding();

  std::string msg {""};
  frame_offset += decoder->NumFramesDecoded();
  if (decoder->NumFramesDecoded() > 0) {
    CompactLattice lat;
    decoder->GetLattice(true, &lat);
    msg = LatticeToString(lat, *word_syms);
    if (produce_time) msg = PrependTimestamps(msg);
  }

  ResetStreamDecoder();
  return msg + eou;
}

OnlineASR::~OnlineASR() {
  delete feature_info;
  delete feature_pipeline;
  delete decoder;
  delete decodable_info;
  delete word_syms;
  delete silence_weighting;
  delete decode_fst;
}
}  // namespace kaldi
