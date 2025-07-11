// online/online-tcp-source.cc

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

#if !defined(_MSC_VER)

#include "online-tcp-source.h"
#include <unistd.h>

namespace kaldi {

typedef kaldi::int32 int32;

OnlineTcpVectorSource::OnlineTcpVectorSource(int32 socket)
    : socket_desc(socket),
      connected(true),
      pack_size(512),
      frame_size(512),
      last_pack_size(0),
      last_pack_rem(0),
      samples_processed(0) {
  pack = new char[pack_size];
  frame = new char[frame_size];
}

OnlineTcpVectorSource::~OnlineTcpVectorSource() {
  delete[] pack;
  delete[] frame;
}

size_t OnlineTcpVectorSource::SamplesProcessed() {
  return samples_processed;
}
void OnlineTcpVectorSource::ResetSamples() {
  samples_processed = 0;
}

bool OnlineTcpVectorSource::ReadFull(char* buf, int32 len) {
  int32 to_read = len;
  int32 has_read = 0;
  int32 ret;

  while (to_read > 0) {
    ret = read(socket_desc, buf + has_read, to_read);
    if (ret <= 0) {
      connected = false;
      return false;
    }
    to_read -= ret;
    has_read += ret;
  }

  return true;
}

int OnlineTcpVectorSource::GetNextPack() {
  int32 size = 0;
  if (!ReadFull((char*) &size, 4))
    return 0;

  if (size % 2 != 0) {
    KALDI_ERR << "TCPVectorSource: Pack size must be even!";
    return 0;
  }

  if (pack_size < size) {
    pack_size = size;
    delete[] pack;
    pack = new char[pack_size];
  }

  if (!ReadFull(pack, size))
    return 0;

  return size;
}

int OnlineTcpVectorSource::FillFrame(int32 get_size) {
  int32 frame_offset = 0;
  if (last_pack_rem > 0) {
    int pack_offset = last_pack_size - last_pack_rem;
    int size = last_pack_rem < get_size ? last_pack_rem : get_size;

    memcpy(frame, pack + pack_offset, size);

    last_pack_rem -= size;
    get_size -= size;
    frame_offset += size;
  }

  while (get_size > 0) {
    int32 ret = GetNextPack();

    if (ret == 0)
      return frame_offset;

    int32 size = ret < get_size ? ret : get_size;

    memcpy(frame + frame_offset, pack, size);

    last_pack_size = ret;
    last_pack_rem = last_pack_size - size;
    get_size -= size;
    frame_offset += size;
  }

  return frame_offset;
}

bool OnlineTcpVectorSource::Read(Vector<BaseFloat> *data) {
  if (!connected)
    return false;

  int32 n_elem = static_cast<uint32>(data->Dim());

  int32 n_bytes = n_elem * 2;

  if (frame_size < n_bytes) {
    frame_size = n_bytes;
    delete[] frame;
    frame = new char[frame_size];
  }

  int32 b_read = FillFrame(n_bytes);
  int32 n_read = b_read / 2;

  short* s_frame = (short*) frame;
  data->Resize(n_read);
  for (int32 i = 0; i < n_read; i++)
    (*data)(i) = s_frame[i];

  samples_processed += n_read;

  return (n_read == n_elem);
}

bool OnlineTcpVectorSource::IsConnected() {
  return connected;
}

}  // namespace kaldi

#endif // !defined(_MSC_VER)
