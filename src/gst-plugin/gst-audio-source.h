// gst-plugin/gst-audio-source.h

// Copyright 2013  Tanel Alumae, Tallinn University of Technology

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

#ifndef GST_ONLINE_AUDIO_SOURCE_H_
#define GST_ONLINE_AUDIO_SOURCE_H_

#include <matrix/kaldi-vector.h>
#include <gst/gst.h>

namespace kaldi {


// OnlineAudioSource implementation using a queue of Gst Buffers
class GstBufferSource {
 public:
  typedef int16 SampleType; // hardcoded 16-bit audio

  GstBufferSource();

  // Implementation of the OnlineAudioSource "interface"
  bool Read(Vector<BaseFloat> *data, uint32 timeout = 0);

  void PushBuffer(GstBuffer *buf);
  
  void SetEnded(bool ended);
  
  ~GstBufferSource();

 private:
  
  GAsyncQueue* buf_queue_;
  gint pos_in_current_buf_;
  GstBuffer *current_buffer_;
  bool ended_;
  GMutex lock_;
  GCond data_cond_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(GstBufferSource);
};

} // namespace kaldi

#endif // GST_ONLINE_AUDIO_SOURCE_H_
