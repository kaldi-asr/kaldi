// gst-plugin/gst-audio-source.h

// Copyright 2013  Tanel Alumae, Tallinn University of Technology

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

#ifndef KALDI_GST_PLUGIN_GST_AUDIO_SOURCE_H_
#define KALDI_GST_PLUGIN_GST_AUDIO_SOURCE_H_

#include <online/online-audio-source.h>
#include <matrix/kaldi-vector.h>
#include <gst/gst.h>

namespace kaldi {


// OnlineAudioSourceItf implementation using a queue of Gst Buffers
class GstBufferSource : public OnlineAudioSourceItf {
 public:
  typedef int16 SampleType;  // hardcoded 16-bit audio

  GstBufferSource();

  // Implementation of the OnlineAudioSourceItf
  bool Read(Vector<BaseFloat> *data);

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

}  // namespace kaldi

#endif  // KALDI_GST_PLUGIN_GST_AUDIO_SOURCE_H_
