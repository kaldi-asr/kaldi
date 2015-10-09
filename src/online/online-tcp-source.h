// online/online-tcp-source.h

// Copyright 2013 Polish-Japanese Institute of Information Technology (author: Danijel Korzinek)

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

#ifndef KALDI_ONLINE_ONLINE_TCP_SOURCE_H_
#define KALDI_ONLINE_ONLINE_TCP_SOURCE_H_

#if !defined(_MSC_VER)

#include "online-audio-source.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {
/*
 * This class implements a VectorSource that reads audio data in a special format from a socket descriptor.
 *
 * The documentation and "interface" for this class is given in online-audio-source.h
 */
class OnlineTcpVectorSource : public OnlineAudioSourceItf {
 public:
  OnlineTcpVectorSource(int32 socket);
  ~OnlineTcpVectorSource();

  // Implementation of the OnlineAudioSourceItf
  bool Read(Vector<BaseFloat> *data);

  //returns if the socket is still connected
  bool IsConnected();

  //returns the number of samples read since the last reset
  size_t SamplesProcessed();
  //resets the number of samples
  void ResetSamples();

 private:
  int32 socket_desc;
  bool connected;
  char* pack;
  int32 pack_size;
  char* frame;
  int32 frame_size;

  int32 last_pack_size, last_pack_rem;

  size_t samples_processed;

  //runs the built-in "read" method as many times as needed to fill "buf" with "len" bytes
  bool ReadFull(char* buf, int32 len);
  //gets the next packet of bytes and returns its size
  int32 GetNextPack();
  //runs "getNextPack" enough times to fill the frame with "size" bytes
  int32 FillFrame(int32 size);

  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineTcpVectorSource);
};

}  // namespace kaldi

#endif // !defined(_MSC_VER)

#endif // KALDI_ONLINE_ONLINE_TCP_SOURCE_H_
