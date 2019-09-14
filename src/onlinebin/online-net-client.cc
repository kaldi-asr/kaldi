// onlinebin/online-net-client.cc

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

#include <netdb.h>
#include <fcntl.h>

#include "feat/feature-mfcc.h"
#include "online/online-audio-source.h"
#include "online/online-feat-input.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    typedef kaldi::int32 int32;
    typedef OnlineFeInput<Mfcc> FeInput;

    // Time out interval for the PortAudio source
    const int32 kTimeout = 500; // half second
    // PortAudio sampling rate
    const int32 kSampleFreq = 16000;
    // PortAudio's internal ring buffer size in bytes
    const int32 kPaRingSize = 32768;
    // Report interval for PortAudio buffer overflows in number of feat. batches
    const int32 kPaReportInt = 4;

    const char *usage =
        "Takes input using a microphone(PortAudio), extracts features and sends them\n"
        "to a speech recognition server over a network connection\n\n"
        "Usage: online-net-client server-address server-port\n\n";
    ParseOptions po(usage);
    int32 batch_size = 27;
    po.Register("batch-size", &batch_size,
                "The number of feature vectors to be extracted and sent in one go");
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      return 1;
    }

    std::string server_addr_str = po.GetArg(1);
    std::string server_port_str = po.GetArg(2);

    addrinfo *server_addr, hints;
    hints.ai_family = AF_INET;
    hints.ai_protocol = IPPROTO_UDP;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_ADDRCONFIG;
    if (getaddrinfo(server_addr_str.c_str(), server_port_str.c_str(),
                    &hints, &server_addr) != 0)
      KALDI_ERR << "getaddrinfo() call failed!";
    int32 sock_desc = socket(server_addr->ai_family,
                             server_addr->ai_socktype,
                             server_addr->ai_protocol);
    if (sock_desc == -1)
      KALDI_ERR << "socket() call failed!";
    int32 flags = fcntl(sock_desc, F_GETFL);
    flags |= O_NONBLOCK;
    if (fcntl(sock_desc, F_SETFL, flags) == -1)
      KALDI_ERR << "fcntl() failed to put the socket in non-blocking mode!";

    // We are not properly registering/exposing MFCC and frame extraction options,
    // because there are parts of the online decoding code, where some of these
    // options are hardwired(ToDo: we should fix this at some point)
    MfccOptions mfcc_opts;
    mfcc_opts.use_energy = false;
    int32 frame_length = mfcc_opts.frame_opts.frame_length_ms = 25;
    int32 frame_shift = mfcc_opts.frame_opts.frame_shift_ms = 10;
    OnlinePaSource au_src(kTimeout, kSampleFreq, kPaRingSize, kPaReportInt);
    Mfcc mfcc(mfcc_opts);
    FeInput fe_input(&au_src, &mfcc,
                     frame_length * (kSampleFreq / 1000),
                     frame_shift * (kSampleFreq / 1000));
    std::cerr << std::endl << "Sending features to " << server_addr_str
              << ':' << server_port_str << " ... " << std::endl;
    char buf[65535];
    Matrix<BaseFloat> feats;
    while (1) {
      feats.Resize(batch_size, mfcc_opts.num_ceps, kUndefined);
      bool more_feats = fe_input.Compute(&feats);
      if (feats.NumRows() > 0) {
        std::stringstream ss;
        feats.Write(ss, true); // serialize features as binary data
        ssize_t sent = sendto(sock_desc,
                              ss.str().c_str(),
                              ss.str().size(), 0,
                              server_addr->ai_addr,
                              server_addr->ai_addrlen);
        if (sent == -1)
          KALDI_ERR << "sendto() call failed!";
        ssize_t rcvd = recvfrom(sock_desc, buf, sizeof(buf), 0,
                                server_addr->ai_addr, &server_addr->ai_addrlen);
        if (rcvd == -1 && errno != EWOULDBLOCK && errno != EAGAIN) {
          KALDI_ERR << "recvfrom() failed unexpectedly!";
        } else if (rcvd > 0) {
          buf[rcvd] = 0;
          std::cout << buf;
          std::cout.flush();
        }
      }
      if (!more_feats) break;
    }
    freeaddrinfo(server_addr);
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
