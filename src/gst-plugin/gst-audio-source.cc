// gst-plugin/gst-audio-source.cc

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

#include <algorithm>

#include "gst-plugin/gst-audio-source.h"

namespace kaldi {


GstBufferSource::GstBufferSource() :
  ended_(false) {
  buf_queue_ = g_async_queue_new();
  current_buffer_ = NULL;
  pos_in_current_buf_ = 0;

  // Monophone, 16-bit input hardcoded
  KALDI_ASSERT(sizeof(SampleType) == 2 &&
      "The current GstBufferSource code assumes 16-bit input");
  g_cond_init(&data_cond_);
  g_mutex_init(&lock_);
}

GstBufferSource::~GstBufferSource() {
  g_cond_clear(&data_cond_);
  g_mutex_clear(&lock_);
  g_async_queue_unref(buf_queue_);
  if (current_buffer_) {
    gst_buffer_unref(current_buffer_);
    current_buffer_ = NULL;
  }
}

void GstBufferSource::PushBuffer(GstBuffer *buf) {
  g_mutex_lock(&lock_);
  gst_buffer_ref(buf);
  g_async_queue_push(buf_queue_, buf);
  g_cond_signal(&data_cond_);
  g_mutex_unlock(&lock_);
}

void GstBufferSource::SetEnded(bool ended) {
  ended_ = ended;
  g_mutex_lock(&lock_);
  g_cond_signal(&data_cond_);
  g_mutex_unlock(&lock_);
}


bool GstBufferSource::Read(Vector<BaseFloat> *data) {
  uint32 nsamples_req = data->Dim();  // (16bit) samples requested
  int16 buf[data->Dim()];
  uint32 nbytes_transferred = 0;

  while ((nbytes_transferred  < nsamples_req * sizeof(SampleType))) {
    g_mutex_lock(&lock_);
    while ((current_buffer_ == NULL) &&
        !((g_async_queue_length(buf_queue_) == 0) && ended_)) {
      current_buffer_ = reinterpret_cast<GstBuffer*>(g_async_queue_try_pop(buf_queue_));
      if (current_buffer_ == NULL) {
        g_cond_wait(&data_cond_, &lock_);
      }
    }
    g_mutex_unlock(&lock_);
    if (current_buffer_ == NULL) {
      break;
    }
    uint32 nbytes_from_current =
        std::min(nsamples_req * sizeof(SampleType) - nbytes_transferred,
                 (gst_buffer_get_size(current_buffer_) - pos_in_current_buf_));
    uint32 nbytes_extracted =
        gst_buffer_extract(current_buffer_, pos_in_current_buf_,
                           (reinterpret_cast<char *>(buf)) + nbytes_transferred,
                           nbytes_from_current);
    KALDI_ASSERT(nbytes_extracted == nbytes_from_current
                 && "Unexpected number of bytes extracted from Gst buffer");

    nbytes_transferred += nbytes_from_current;
    pos_in_current_buf_ += nbytes_from_current;
    if (pos_in_current_buf_ == gst_buffer_get_size(current_buffer_)) {
      // we are done with the current buffer
      gst_buffer_unref(current_buffer_);
      current_buffer_ = NULL;
      pos_in_current_buf_ = 0;
    }
  }

  for (int i = 0; i < nbytes_transferred / sizeof(SampleType) ; ++i) {
    (*data)(i) = static_cast<BaseFloat>(buf[i]);
  }
  return !((g_async_queue_length(buf_queue_) < sizeof(SampleType))
      && ended_
      && (current_buffer_ == NULL));
}
}
