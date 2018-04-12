// online2bin/online2-audio-nnet2-latgen-faster.cc
// a modified version of:
// online2bin/online2-wav-nnet2-latgen-faster.cc
// original author: Daniel Povey, Johns Hopkins University

// Copyright 2015-2018  the ARIA-VALUSPA project
//        (authors: B. Potard, E. Coutinho, A. Mousa)

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

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <unistd.h>

#include "feat/wave-reader.h"
#include "online2/online-nnet2-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "online2/online-tcp-source.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"

namespace kaldi {

/*
 * This class is for a very simple TCP server implementation
 * in UNIX sockets.
 */
  class TcpServer {
  public:
    TcpServer();
    ~TcpServer();

    bool Listen(int32 port);  // Start listening on a given port
    int32 Accept();  // Accept a client and return its descriptor

  private:
    struct sockaddr_in h_addr_;
    int32 server_desc_;
  };

  clock_t start;
  int32 s_sample = 0;  // Start sample of an audio segment, used
  // to help calculating the exact time start and end of a
  // recognized audio segment

// Write a line of text to socket
  bool WriteLine(int32 socket, std::string line);

// Constant allowing to convert frame count to time
  const float kFramesPerSecond = 100.0f;
// Constant allowing to convert sample count to time
  const float kSamplingFrequency = 16000.0f;

  void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                    const fst::SymbolTable *word_syms,
                                    const CompactLattice &clat,
                                    int64 *tot_num_frames,
                                    double *tot_like,
                                    int client_socket,
                                    OnlineTcpVectorSource* au_src,
                                    int nbest,
                                    bool partial_out) {
    if (clat.NumStates() == 0) {
      KALDI_WARN << "Empty lattice.";
      return;
    }

    Lattice lat;
    ConvertLattice(clat, &lat);  // Convert CompactLattice to Lattice

    Lattice nbest_lat;                // One lattice to hold all the nbest
    std::vector<Lattice> nbest_lats;  // A vector of lattices to hold the
    // nbest lattices as a vector

    fst::ShortestPath(lat, &nbest_lat, nbest);  // Extract n best paths and
    // store in one lattice nbest_lat
    fst::ConvertNbestToVector(nbest_lat, &nbest_lats);  // Get a vector of
    // nbest lattices of type Lattice

    if (nbest_lats.empty()) {
      KALDI_WARN << "Empty lattice (no N-best entries).";
      return;
    }

    for (int32 k = 0; k < static_cast<int32>(nbest_lats.size()); k++) {
      LatticeWeight weight;
      std::vector<int32> alignment;
      std::vector<int32> words;
      int32 num_frames;
      GetLinearSymbolSequence(nbest_lats[k], &alignment, &words, &weight);
      num_frames = alignment.size();
      if (!partial_out && k == 0) {  // If it is not a call for partial output
        double likelihood;
        likelihood = -(weight.Value1() + weight.Value2());
        *tot_num_frames += num_frames;
        *tot_like += likelihood;
        KALDI_VLOG(2) << "Likelihood per frame for utterance "
                      << utt << " is " << (likelihood / num_frames)
                      << " over " << num_frames << " frames.";
      }
      int32 words_num = 0;  // Count number of non-sil words
      if (word_syms != NULL) {
        if (partial_out) {
          if (words.size() > 0) std::cerr << "\33[2K\r" << utt << "_PART: ";
        } else {
          std::cerr << utt << ' ';
        }
        for (size_t i = 0; i < words.size(); i++) {
          if (words[i] != 0) words_num++;  // Count number of non-sil words
          std::string s = word_syms->Find(words[i]);
          if (s == "") {
            KALDI_ERR << "Word-id " << words[i]
                      << " not in symbol table.";
          }
          std::cerr << s << ' ';
        }
        std::cerr << std::endl;
        float dur = (clock() - start) / static_cast<float>(CLOCKS_PER_SEC);
        float input_dur = au_src->SamplesProcessed() / kSamplingFrequency;
        float s_time = s_sample / kSamplingFrequency;
        float e_time = s_time + input_dur;
        std::stringstream sstr;
        sstr << "RESULT:NUM=" << words_num << ",FORMAT=WSE,RECO-DUR=" << dur
             << ",INPUT-DUR=" << input_dur << ",INPUT-TIME-START=" << s_time
             << ",INPUT-TIME-END=" << e_time;

        WriteLine(client_socket, sstr.str());
        if (words_num > 0) {
          for (size_t i = 0; i < words.size(); i++) {
            if (words[i] == 0)
              continue;  // Skip silences...

            std::string word = word_syms->Find(words[i]);
            if (word.empty())
              word = "???";
            std::stringstream wstr;
            wstr << word;
            WriteLine(client_socket, wstr.str());
          }
        }
        if (nbest > 1) {
          std::stringstream rstr;  // Result string
          rstr << "(" << k << ")";
          WriteLine(client_socket, rstr.str());
        }
      }  // if
    }  // for
    if (partial_out) {
      WriteLine(client_socket, "RESULT:PART");
    } else {
      WriteLine(client_socket, "RESULT:DONE");
    }
    if (!partial_out) {
      start = clock();
      s_sample += au_src->SamplesProcessed();
      au_src->ResetSamples();
    }
  }  // function
}  // namespace

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Listens on the given socket number for audio stream input\n"
      "and simulates (1-best) online decoding with neural nets (nnet2 setup),\n"
      "with optional iVector-based speaker adaptation, optional endpointing.\n"
      "Note: some configuration values and inputs are set via config files\n"
      "whose filenames are passed as options.\n"
      "\n"
      "Usage: online2-audio-nnet2-latgen-faster [options] <nnet2-in> <fst-in> "
      "<socket-number> \n"
      "See also online2-audio-nnet3-latgen-faster\n"
      "and online2-wav-nnet2-latgen-faster.\n";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    TcpServer tcp_server;
    OnlineEndpointConfig endpoint_config;

    // feature_config includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_config;
    OnlineNnet2DecodingConfig nnet2_decoding_config;

    BaseFloat chunk_length_secs = 1.0f / kFramesPerSecond;
    bool do_endpointing = false;
    bool online = true;
    bool streaming = false;
    int nbest = 1;
    std::string use_gpu = "optional";

    po.Register(
      "chunk-length", &chunk_length_secs,
      "Length of chunk size in seconds, that we process.  Set to <= 0 "
      "to use all input in one chunk.");
    po.Register(
      "word-symbol-table", &word_syms_rxfilename,
      "Symbol table for words [for debug output]");
    po.Register(
      "do-endpointing", &do_endpointing,
      "If true, apply endpoint detection");
    po.Register(
      "online", &online,
      "You can set this to false to disable online iVector estimation "
      "and have all the data for each utterance used, even at "
      "utterance start.  This is useful where you just want the best "
      "results and don't care about online operation.  Setting this to "
      "false has the same effect as setting "
      "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
      "in the file given to --ivector-extraction-config, and "
      "--chunk-length=-1.");
    po.Register(
      "num-threads-startup", &g_num_threads,
      "Number of threads used when initializing iVector extractor.");
    po.Register(
      "n-best", &nbest,
      "Number of top ASR results to output.");
    po.Register(
      "streaming", &streaming,
      "If true, output partial results as well as final ones.");
    po.Register(
      "use-gpu", &use_gpu,
      "yes|no|optional|wait, only has effect if compiled with CUDA");

    feature_config.Register(&po);
    nnet2_decoding_config.Register(&po);
    endpoint_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      return 1;
    }
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet2_rxfilename = po.GetArg(1),
      fst_rxfilename = po.GetArg(2);
    int32 port = strtol(po.GetArg(3).c_str(), 0, 10);
    const int32 step_size = 100;  // 1s

    OnlineNnet2FeaturePipelineInfo feature_info(feature_config);

    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

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

    // Starting tcp server
    if (!tcp_server.Listen(port))
      return 0;

    OnlineTimingStats timing_stats;

    // Copied from online-audio-server
    OnlineTcpVectorSource* au_src = NULL;
    int32 client_socket = -1;
    int32 segment = 0;

    // Define one adaptation state that is being updated via
    // (Set & Get) during the whole dialogue session
    // so that the whole session is considered for the
    // same speaker.
    OnlineIvectorExtractorAdaptationState adaptation_state(
      feature_info.ivector_extractor_info);

    while (true) {
      if (au_src == NULL || !au_src->IsConnected()) {
        if (au_src) {
          std::cout << "Client disconnected!" << std::endl;
          delete au_src;
        }
        client_socket = tcp_server.Accept();
        au_src = new OnlineTcpVectorSource(client_socket);
        segment = 0;
        start = clock();
      }

      std::ostringstream s;
      s << "segment_" << segment;
      std::string utt = s.str();

      // Initialise feature pipeline
      OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
      feature_pipeline.SetAdaptationState(adaptation_state);

      OnlineSilenceWeighting silence_weighting(
        trans_model,
        feature_info.silence_weighting_config);

      SingleUtteranceNnet2Decoder decoder(nnet2_decoding_config,
                                          trans_model,
                                          nnet,
                                          *decode_fst,
                                          &feature_pipeline);
      OnlineTimer decoding_timer(utt);

      // Hacky
      BaseFloat samp_freq = kSamplingFrequency;
      int32 chunk_length;
      if (chunk_length_secs > 0) {
        chunk_length = static_cast<int32>(samp_freq * chunk_length_secs);
        if (chunk_length == 0) chunk_length = 1;
      } else {
        chunk_length = std::numeric_limits<int32>::max();
      }

      int32 samp_offset = 0;
      std::vector<std::pair<int32, BaseFloat> > delta_weights;

      bool more_data = true;
      int32 prev_decoded_frames = 0;
      while (more_data) {
        if (!au_src->IsConnected())
          break;

        // Get data from socket
        Vector<BaseFloat> wave_part(chunk_length);
        more_data = au_src->Read(&wave_part);

        int32 new_offset = au_src->SamplesProcessed();
        int32 nb_received = new_offset - samp_offset;
        if (nb_received > 0) {
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);
        }

        samp_offset = new_offset;
        decoding_timer.WaitUntil(samp_offset / samp_freq);
        if (!more_data && samp_offset > 0) {
          // no more input. flush out last frames
          feature_pipeline.InputFinished();
        }
        if (silence_weighting.Active() &&
            feature_pipeline.IvectorFeature() != NULL) {
          silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
          silence_weighting.GetDeltaWeights(
            feature_pipeline.IvectorFeature()->NumFramesReady(),
            &delta_weights);
          feature_pipeline.IvectorFeature()->UpdateFrameWeights(
            delta_weights);
        }

        decoder.AdvanceDecoding();

        if (do_endpointing && decoder.EndpointDetected(endpoint_config))
          break;

        if (streaming && more_data) {
          int32 decoded_frames = decoder.NumFramesDecoded();
          if (decoded_frames >= prev_decoded_frames + step_size) {
            // output every some frames
            CompactLattice partial_clat;
            decoder.GetLattice(false, &partial_clat);
            GetDiagnosticsAndPrintOutput(
              utt, word_syms, partial_clat,
              &num_frames, &tot_like, client_socket, au_src, 1, true);
            prev_decoded_frames = decoded_frames;
          }  // if
        }  // if
      }
      // Deal with this a bit better
      if (samp_offset == 0) continue;
      decoder.FinalizeDecoding();
      std::cerr << "End of segment detected.";

      CompactLattice clat;
      bool end_of_utterance = true;
      decoder.GetLattice(end_of_utterance, &clat);

      GetDiagnosticsAndPrintOutput(
        utt, word_syms, clat,
        &num_frames, &tot_like, client_socket, au_src, nbest, false);

      decoding_timer.OutputStats(&timing_stats);
      au_src->ResetSamples();

      // In an application you might avoid updating the adaptation state if
      // you felt the utterance had low confidence.  See lat/confidence.h
      feature_pipeline.GetAdaptationState(&adaptation_state);

      KALDI_LOG << "Decoded utterance " << utt;
      num_done++;
      segment += 1;
    }
    timing_stats.Print(online);

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    delete decode_fst;
    delete word_syms;  // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
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
    int32 len = sizeof(flag);
    if (setsockopt(server_desc_, SOL_SOCKET, SO_REUSEADDR, &flag, len) == -1) {
      KALDI_ERR << "Cannot set socket options!\n";
      return false;
    }

    if (bind(server_desc_, (struct sockaddr*) &h_addr_,
             sizeof(h_addr_)) == -1) {
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
    int32 to_write = strlen(p);
    int32 wrote = 0;
    while (to_write > 0) {
      int32 ret = write(socket, p + wrote, to_write);
      if (ret <= 0)
        return false;

      to_write -= ret;
      wrote += ret;
    }
    int flag = 1;
    setsockopt(socket, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<char *>(&flag), sizeof(flag));

    return true;
  }
}  // namespace kaldi

