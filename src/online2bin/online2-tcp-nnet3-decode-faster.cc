// online2bin/online2-tcp-nnet3-decode-faster.cc

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
#include "lat/word-align-lattice.h"
#include "lat/sausages.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <poll.h>
#include <signal.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>

#include <jansson.h>

/* JSON_REAL_PRECISION is a macro from libjansson 2.7. Ubuntu 12.04 only has 2.2.1-1 */
#ifndef JSON_REAL_PRECISION
#define JSON_REAL_PRECISION(n)  (((n) & 0x1F) << 11)
#endif // JSON_REAL_PRECISION

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

typedef struct _WordAlignmentInfo WordAlignmentInfo;

struct _WordAlignmentInfo {
  int32 word_id;
  int32 start_frame;
  int32 length_in_frames;
  double confidence;
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

std::string LatticeToJson(const Lattice &lat, 
                          const fst::SymbolTable &word_syms,
                          const WordBoundaryInfo &word_boundary,
                          const TransitionModel &trans_model,
                          const OnlineNnet2FeaturePipelineInfo &feature_info,
                          const nnet3::NnetSimpleLoopedComputationOptions &decodable_opts,
                          int32 frame_offset,
                          bool final) {
  LatticeWeight weight;
  std::vector<int32> alignment;
  std::vector<int32> words;
  std::vector<WordAlignmentInfo> word_alignment;
  std::vector<int32> times, lengths;

  GetLinearSymbolSequence(lat, &alignment, &words, &weight);

  // Get the transcript
  std::ostringstream tr;
  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms.Find(words[i]);
    if (s.empty()) {
      KALDI_WARN << "Word-id " << words[i] << " not in symbol table.";
      tr << "<#" << std::to_string(i) << "> ";
    } else
      tr << s << " ";
  }
  std::string transcript = tr.str();
  size_t end = transcript.find_last_not_of(" \n\r\t\f\v");
	transcript = (end == std::string::npos) ? "" : transcript.substr(0, end + 1);

  // Break here if we have no words, this skips the segment totally
  if (transcript.empty())
    return "";

  BaseFloat likelihood = -(weight.Value1() + weight.Value2());
  int num_frames = alignment.size();

  CompactLattice clat;
  ConvertLattice(lat, &clat);
  CompactLattice aligned_clat;

  // Get confidences
  MinimumBayesRiskOptions mbr_opts;
  mbr_opts.decode_mbr = false; // we just want confidences
  mbr_opts.print_silence = false; 
  MinimumBayesRisk *mbr = new MinimumBayesRisk(clat, words, mbr_opts);
  std::vector<BaseFloat> confidences = mbr->GetOneBestConfidences();
  delete mbr;

  // Word-align the lattice (can be phone-aligned too if needed)
  if (!WordAlignLattice(clat, trans_model, word_boundary, 0, &aligned_clat)) {
    KALDI_WARN << "Failed to word-align the lattice";
    return "";
  }

  if (!CompactLatticeToWordAlignment(aligned_clat, &words, &times, &lengths)) {
    KALDI_WARN << "Failed to do word alignment";
    return "";
  }

  // Let's assume the vectors are all the same size
  KALDI_ASSERT(words.size() == times.size() && words.size() == lengths.size());

  // Populate the word_alignment structure
  int confidence_i = 0;
  for (size_t i = 0; i < words.size(); i++) {
    if (words[i] == 0)  {
      // Don't output anything for <eps> links, which
      continue; // correspond to silence....
    }
    WordAlignmentInfo alignment_info;
    alignment_info.word_id = words[i];
    alignment_info.start_frame = times[i];
    alignment_info.length_in_frames = lengths[i];
    if (confidences.size() > 0) {
      alignment_info.confidence = confidences[confidence_i++];
    }
    word_alignment.push_back(alignment_info);
  }

  // Construct the returned json object
  json_t *root = json_object();
  json_t *result_json_object = json_object();
  json_object_set_new(root, "status", json_integer(0));
  json_object_set_new(root, "result", result_json_object);

  if (final)
    json_object_set_new(result_json_object, "final", json_true());
  else
    json_object_set_new(result_json_object, "final", json_false());

  BaseFloat frame_shift = feature_info.FrameShiftInSeconds();
  frame_shift *= decodable_opts.frame_subsampling_factor;

  json_object_set_new(root, "segment-start",  json_real((frame_offset - num_frames) * frame_shift));
  json_object_set_new(root, "segment-length",  json_real(num_frames * frame_shift));
  json_object_set_new(root, "total-length",  json_real(frame_offset * frame_shift));

  json_t *nbest_json_arr = json_array();
  json_t *nbest_result_json_object = json_object();
  json_object_set_new(nbest_result_json_object, "transcript", json_string(transcript.c_str()));
  json_object_set_new(nbest_result_json_object, "likelihood",  json_real(likelihood));
  json_array_append(nbest_json_arr, nbest_result_json_object);
  
  json_t *word_alignment_json_arr = json_array();
  for (size_t j = 0; j < word_alignment.size(); j++) {
    WordAlignmentInfo alignment_info = word_alignment[j];
    json_t *alignment_info_json_object = json_object();
    std::string word = word_syms.Find(alignment_info.word_id);
    json_object_set_new(alignment_info_json_object, "word", json_string(word.c_str()));
    json_object_set_new(alignment_info_json_object, "start", json_real(alignment_info.start_frame * frame_shift));
    json_object_set_new(alignment_info_json_object, "length", json_real(alignment_info.length_in_frames * frame_shift));
    json_object_set_new(alignment_info_json_object, "confidence", json_real(alignment_info.confidence));
    json_array_append(word_alignment_json_arr, alignment_info_json_object);
  }
  json_object_set_new(nbest_result_json_object, "word-alignment", word_alignment_json_arr);
  json_object_set_new(result_json_object, "hypotheses", nbest_json_arr);

  char *ret_strings = json_dumps(root, JSON_REAL_PRECISION(6));
  json_decref(root);
  std::string json;
  json = ret_strings;
  return json;
}

std::string LatticeToJson(const CompactLattice &clat, 
                          const fst::SymbolTable &word_syms, 
                          const WordBoundaryInfo &word_boundary,
                          const TransitionModel &trans_model,
                          const OnlineNnet2FeaturePipelineInfo &feature_info,
                          const nnet3::NnetSimpleLoopedComputationOptions &decodable_opts,
                          int32 frame_offset,
                          bool final) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return "";
  }

  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);
  return LatticeToJson(best_path_lat, word_syms, word_boundary, trans_model,
                       feature_info, decodable_opts, frame_offset, final);
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in audio from a network socket and performs online\n"
        "decoding with neural nets (nnet3 setup), with iVector-based\n"
        "speaker adaptation and endpointing.\n"
        "Note: some configuration values and inputs are set via config\n"
        "files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-tcp-nnet3-decode-faster [options] <model-dir>\n";

    ParseOptions po(usage);


    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = 0.18;
    BaseFloat output_period = 1;
    BaseFloat samp_freq = 16000.0;
    int port_num = 5050;
    int read_timeout = 3;

    po.Register("samp-freq", &samp_freq,
                "Sampling frequency of the input signal (coded as 16-bit slinear).");
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");
    po.Register("output-period", &output_period,
                "How often in seconds, do we check for changes in output.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");
    po.Register("read-timeout", &read_timeout,
                "Number of seconds of timout for TCP audio data to appear on the stream. Use -1 for blocking.");
    po.Register("port-num", &port_num,
                "Port number the server will listen on.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      return 1;
    }

    std::string model_dir = po.GetArg(1);
    std::string nnet3_rxfilename = model_dir + "/final.mdl";
    std::string fst_rxfilename = model_dir + "/HCLG.fst.map";
    std::string word_syms_filename = model_dir + "/words.txt";
    std::string word_boundary_filename = model_dir + "/phones/word_boundary.int";

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    KALDI_VLOG(1) << "Loading AM...";

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

    KALDI_VLOG(1) << "Loading FST...";

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (!word_syms_filename.empty())
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    WordBoundaryInfoNewOpts opts;
    WordBoundaryInfo *word_boundary = new WordBoundaryInfo(opts, word_boundary_filename);

    signal(SIGPIPE, SIG_IGN); // ignore SIGPIPE to avoid crashing when socket forcefully disconnected

    TcpServer server(read_timeout);

    server.Listen(port_num);

    while (true) {

      server.Accept();

      int32 samp_count = 0;// this is used for output refresh rate
      size_t chunk_len = static_cast<size_t>(chunk_length_secs * samp_freq);
      int32 check_period = static_cast<int32>(samp_freq * output_period);
      int32 check_count = check_period;

      int32 frame_offset = 0;

      bool eos = false;

      OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
      SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                          decodable_info,
                                          *decode_fst, &feature_pipeline);

      while (!eos) {

        decoder.InitDecoding(frame_offset);
        OnlineSilenceWeighting silence_weighting(
            trans_model,
            feature_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor);
        std::vector<std::pair<int32, BaseFloat>> delta_weights;

        while (true) {
          eos = !server.ReadChunk(chunk_len);

          if (eos) {
            feature_pipeline.InputFinished();
            decoder.AdvanceDecoding();
            decoder.FinalizeDecoding();
            frame_offset += decoder.NumFramesDecoded();
            if (decoder.NumFramesDecoded() > 0) {
              CompactLattice lat;
              decoder.GetLattice(true, &lat);
              //std::string msg = LatticeToString(lat, *word_syms);
              std::string msg = LatticeToJson(lat, *word_syms, *word_boundary, trans_model,
                                              feature_info, decodable_opts, frame_offset, true);
              if (msg.size() > 0)
                server.WriteLn(msg);
            } else
              server.Write("\n");
            server.Disconnect();
            break;
          }

          Vector<BaseFloat> wave_part = server.GetChunk();
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);
          samp_count += chunk_len;

          if (silence_weighting.Active() &&
              feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.UpdateFrameWeights(delta_weights,
                                                frame_offset * decodable_opts.frame_subsampling_factor);
          }

          decoder.AdvanceDecoding();

          if (samp_count > check_count) {
            if (decoder.NumFramesDecoded() > 0) {
              Lattice lat;
              decoder.GetBestPath(false, &lat);
              //std::string msg = LatticeToString(lat, *word_syms);
              std::string msg = LatticeToJson(lat, *word_syms, *word_boundary, trans_model,
                                              feature_info, decodable_opts, 
                                              frame_offset + decoder.NumFramesDecoded(), false);
              if (msg.size() > 0)
                server.WriteLn(msg);
            }
            check_count += check_period;
          }

          if (decoder.EndpointDetected(endpoint_opts)) {
            decoder.FinalizeDecoding();
            frame_offset += decoder.NumFramesDecoded();
            CompactLattice lat;
            decoder.GetLattice(true, &lat);
            //std::string msg = LatticeToString(lat, *word_syms);
            std::string msg = LatticeToJson(lat, *word_syms, *word_boundary, trans_model,
                                              feature_info, decodable_opts, frame_offset, true);
            if (msg.size() > 0)
              server.WriteLn(msg);
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
  size_t to_read = len;
  has_read_ = 0;
  while (to_read > 0) {
    poll_ret = poll(client_set_, 1, read_timeout_);
    if (poll_ret == 0) {
      KALDI_WARN << "Socket timeout! Disconnecting...";
      break;
    }
    if (client_set_[0].revents != POLLIN) {
      KALDI_WARN << "Socket error! Disconnecting...";
      break;
    }
    ret = read(client_desc_, static_cast<void *>(samp_buf_ + has_read_), to_read * sizeof(int16));
    if (ret <= 0) {
      KALDI_WARN << "Stream over...";
      break;
    }
    to_read -= ret / sizeof(int16);
    has_read_ += ret / sizeof(int16);
  }

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
}  // namespace kaldi