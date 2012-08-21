// online/online-audio-source.h

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

#ifndef KALDI_ONLINE_AUDIO_SOURCE_H_
#define KALDI_ONLINE_AUDIO_SOURCE_H_

#include <portaudio.h>
#include <pa_ringbuffer.h>

#include "matrix/kaldi-vector.h"

namespace kaldi {

// There is in fact no real hierarchy of classes w/ virtual methods etc.,
// as C++ templates are used instead. The class below is given just
// to document the interface.
class OnlineAudioSource {
 public:
  // Reads from the audio source, and writes the samples converted to BaseFloat
  // into the vector pointed by "data". The function assumes, that "data" already
  // has the right size - i.e. its length is equal to the count of samples
  // requested. The original contents will be overwritten.
  // The function blocks until data->Dim() samples are read, unless
  // no more data is available for some reason(EOF?), or "*timeout" (in ms)
  // expires. In each case the function returns the number of samples actually
  // read. If the returned number is less than data->Dim(), the contents of the
  // remainder of the vector are undefined. If timer expires "data" contains
  // the samples read until timeout occured and *timeout contains zero.
  // In case timeout is not reached the contents of "*timeout" are left unchanged.
  // The timeout is considered to be a hint only and one should not rely on it
  // to be completely accurate. The "timeout" is not considered if this pointer
  // is zero and the function can block for indefinitely long period.
  // In case an unexpected and unrecoverable error occurs the function throws
  // an exception of type std::runtime_error (e.g. by using KALDI_ERR macro).
  int32 Read(VectorBase<BaseFloat> *data, int32 *timeout = 0) { return 0; }
};


// OnlineAudioSource implementation using PortAudio to read samples in real-time
// from a sound card/microphone.
class OnlinePaSource {
 public:
  typedef int16 SampleType; // hardcoded 16-bit audio
  typedef ring_buffer_size_t rbs_t;

  // PortAudio is initialized here, so it may throw an exception on error
  // "sample_rate": the input rate to request from PortAudio
  // "rb_size": requested size of PA's ring buffer - will be round up to
  //           power of 2
  // "report_interval": if not 0, PA ring buffer overflow will be reported
  //                    at every ovfw_msg_interval-th call to Read().
  //                    Putting 0 into this argument disables the reporting.
  OnlinePaSource(const uint32 sample_rate, const uint32 rb_size,
                 const uint32 report_interval);

  // Implementation of the OnlineAudioSource "interface" - see above
  int32 Read(VectorBase<BaseFloat> *data, uint32 *timeout = 0);

  // Making friends with the callback so it will be able to access a private
  // member function to delegate the processing
  friend int PaCallback(const void *input, void *output,
                        long unsigned frame_count,
                        const PaStreamCallbackTimeInfo *time_info,
                        PaStreamCallbackFlags status_flags,
                        void *user_data);

  ~OnlinePaSource();

 private:
  // The real PortAudio callback delegates to this one
  int Callback(const void *input, void *output,
               ring_buffer_size_t frame_count,
               const PaStreamCallbackTimeInfo *time_info,
               PaStreamCallbackFlags status_flags);

  uint32 sample_rate_; // the sampling rate of the input audio
  int32 rb_size_;
  char *ring_buffer_; // points to the actual buffer used by PA to store samples
  PaUtilRingBuffer pa_ringbuf_; // a data structure used to wrap the ring buffer
  PaStream *pa_stream_;
  bool pa_started_; // becomes "true" after "pa_stream_" is started
  uint32 report_interval_; // interval (in Read() calls) to report PA rb overflows
  uint32 nread_calls_; // number of Read() calls so far
  uint32 noverflows_; // number of the ringbuf overflows since the last report
  uint32 samples_lost_; // samples lost, due to PA ring buffer overflow
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlinePaSource);
};

// The actual PortAudio callback - delegates to OnlinePaSource->PaCallback()
int PaCallback(const void *input, void *output,
               long unsigned frame_count,
               const PaStreamCallbackTimeInfo *time_info,
               PaStreamCallbackFlags status_flags,
               void *user_data);


// Simulates audio input, by returning data from a Vector.
// This class is mostly meant to be used for online decoder testing using
// pre-recorded audio
class OnlineVectorSource {
 public:
  OnlineVectorSource(const VectorBase<BaseFloat> &input)
      : src_(input), pos_(0) {}

  // Implementation of the OnlineAudioSource "interface" - see above
  int32 Read(VectorBase<BaseFloat> *data, uint32 *timeout = 0);

 private:
  Vector<BaseFloat> src_;
  uint32 pos_; // the index of the first element, not yet consumed
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineVectorSource);
};

} // namespace kaldi

#endif // KALDI_ONLINE_AUDIO_SOURCE_H_
