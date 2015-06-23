// nnet/nnet-io-socket.h

// Copyright 2015  Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_NNET_IO_SOCKET_H_
#define KALDI_NNET_IO_SOCKET_H_

#include <sstream>

#ifdef __linux__
  #include <unistd.h>
  #include <string.h>
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netdb.h>
#endif

#include "base/kaldi-common.h"
#include "util/common-utils.h"


namespace kaldi {
namespace nnet1 {


/**
 * Class Socket, for bidirectional communication over TCP/IP socket,
 * using blocking read/write operations : send(), recv().
 * 
 * Each message begins with its size in bytes encoded as int64 
 * (does not include the int64).
 *
 * The messages are atomic, meaninig the process is recieving
 * into buffer, until message the complete. Then it is put into 
 * 'ostringsream', from which it is parsed by data-type specific 
 * 'Read()' methods. 
 * 
 * It is used for clients (Asynchronous-SGD training), 
 * can be extended to servers. Currently only for linux.
 */
class Socket {
 public:
  Socket() : buf_size_(8388608) {} // 8MB buffer size (from Olda),
  Socket(std::string host, int32 port) : buf_size_(8388608) {
    Connect(host, port);
  }
  ~Socket(); // disconnect,

  std::istringstream& IStream() { 
    return iss_;
  }
  std::ostringstream& OStream() {
    return oss_;
  }

  void Connect(std::string host, int32 port);
  void Recv();
  void Send();

 private:
  std::istringstream iss_;
  std::ostringstream oss_;

#ifdef __linux__
  std::string host_;
  int sockfd_, port_;
  struct sockaddr_in serv_addr_;
  struct hostent *server_;
  const size_t buf_size_;
#endif
};
  
}
}

#endif
