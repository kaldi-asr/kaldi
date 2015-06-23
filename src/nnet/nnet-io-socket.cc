// nnet/nnet-io-socket.cc

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

#include "nnet/nnet-io-socket.h"

namespace kaldi {
namespace nnet1 {


Socket::~Socket() {
  KALDI_LOG << "Disconnecting " << host_ << ", port " << port_;
  int32 ret = shutdown(sockfd_, SHUT_RDWR);
  if (ret != 0) { KALDI_ERR << strerror(errno); }
  ret = close(sockfd_);
  if (ret != 0) { KALDI_ERR << strerror(errno); }
  free(server_);
}


void Socket::Connect(std::string host, int32 port) {
#ifdef __linux__
  host_ = host;
  port_ = port;
  // create socket,
  sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd_ < 0) 
    KALDI_ERR << "Cannot create socket!"; 
  // get structure with hostname,
  server_ = gethostbyname(host.c_str());
  if (server_ == NULL) 
    KALDI_ERR << "No such host" << host;\
  // prepare 'struct sockaddr_in' for connect(),
  bzero(reinterpret_cast<char*>(&serv_addr_), sizeof(serv_addr_)); // fill with zeros,
  serv_addr_.sin_family = AF_INET;
  bcopy(reinterpret_cast<char *>(server_->h_addr), 
        reinterpret_cast<char *>(&serv_addr_.sin_addr.s_addr),
        server_->h_length);
  serv_addr_.sin_port = htons(port);
  // connect,
  int32 ret = connect(sockfd_,reinterpret_cast<struct sockaddr *>(&serv_addr_), sizeof(serv_addr_));
  if (ret < 0) {
    KALDI_ERR << "Cannot connect to " << host << " port " << port;
  }
  KALDI_LOG << "Connected to " << host_ << ", port " << port_;
#else
  KALDI_ERR << "Only linux supported, sorry...";
#endif
}


void Socket::Recv() { 
#ifdef __linux__
  KALDI_ASSERT(iss_.str().empty());
  // header is the size of the message,
  int64 msg_size;
  int32 ret = recv(sockfd_, &msg_size, 8, 0);
  KALDI_ASSERT(ret == 8);
  KALDI_LOG << "Message size is " << msg_size; // DEBUG,
  KALDI_ASSERT(msg_size > 0);
  KALDI_ASSERT(msg_size < 1e10); // Can be removed,
  // buffer for recieving,
  char buffer[buf_size_];
  // put whole message to string 's',
  std::string s;
  s.reserve(msg_size);
  int64 remain_size = msg_size;
  while (remain_size != 0) {
    int32 chunk_size = (remain_size > buf_size_ ? buf_size_ : remain_size);
    int32 ret = recv(sockfd_, buffer, chunk_size, 0);
    s.append(buffer, ret);
    remain_size -= ret;
    KALDI_ASSERT(remain_size >= 0);
  }
  // put the message to the stringstream,
  iss_.str(s);
  // check,
  KALDI_ASSERT(iss_.str().size() == msg_size);
#else
  KALDI_ERR << "Only linux supported, sorry...";
#endif
}

 
void Socket::Send() {
#ifdef __linux__
  KALDI_ASSERT(!oss_.str().empty());
  int64 msg_size = oss_.str().size();
  int32 ret = send(sockfd_, &msg_size, 8, 0);
  KALDI_ASSERT(ret == 8);
  KALDI_LOG << "Message size is " << msg_size; // DEBUG,
  // send per blokcs,
  int64 remain_size = msg_size;
  const char* msg_ptr(oss_.str().c_str());
  while (remain_size != 0) {
    int32 chunk_size = (remain_size > buf_size_ ? buf_size_ : remain_size);
    int32 ret = send(sockfd_, msg_ptr, chunk_size, 0);
    remain_size -= ret;
    msg_ptr += ret;
    KALDI_ASSERT(remain_size >= 0);
  }
  // clear the ostringstream,
  oss_.str("");
#else
  KALDI_ERR << "Only linux supported, sorry...";
#endif
}


} // namespace nnet1
} // namespace kaldi

