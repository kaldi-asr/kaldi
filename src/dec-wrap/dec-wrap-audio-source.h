// dec-wrap/dec-wrap-audio-source.cc

/* Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
 *                     2012-2013  Vassil Panayotov
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
 * MERCHANTABLITY OR NON-INFRINGEMENT.
 * See the Apache 2 License for the specific language governing permissions and
 * limitations under the License. */



#ifndef KALDI_DEC_WRAP_DEC_WRAP_AUDIO_SOURCE_H_
#define KALDI_DEC_WRAP_DEC_WRAP_AUDIO_SOURCE_H_

#include "matrix/kaldi-vector.h"
#include "itf/options-itf.h"

namespace kaldi {


// Copied and renamed from online -> in order to get rid of portaudio dependency
class OnlAudioSourceItf {
 public:
  // Reads from the audio source, and writes the samples converted to BaseFloat
  // into the vector pointed by "data".
  // The user sets data->Dim() as a way of requesting that many samples.
  // The function returns number of actually read data.
  // Ideal scenerio the return value is equal to data->Dim()
  virtual MatrixIndexT Read(Vector<BaseFloat> *data) = 0;

  virtual ~OnlAudioSourceItf() { }
};

struct OnlBuffSourceOptions {
  int32 bits_per_sample;
  OnlBuffSourceOptions(): bits_per_sample(16) { }
  void Register(OptionsItf *po) {
    po->Register("bits-per-sample", &bits_per_sample,
                 "Number of bits for storing one audio sample. Typically 8, 16, 32");
  }
};

/** @brief Proxy Audio Input. Acts like a buffer.
 *
 *  OnlAudioSource implementation.
 *  It expects to be fed with the audio frame by frame.
 *  Supports only one channel. */
class OnlBuffSource: public OnlAudioSourceItf {
 public:

  /// Creates the OnlBuffSource empty "buffer"
  OnlBuffSource(const OnlBuffSourceOptions &opts): opts_(opts) { }

  size_t BufferSize() { return src_.size(); }
  size_t frame_size;


 MatrixIndexT Read(Vector<BaseFloat> *data);

  /// Converts and buffers  the data based on bits_per_sample 
  /// @param data [in] the single channel pcm audio data
  /// @param num_samples [in] number of samples in data array
  void Write(unsigned char *data, size_t num_samples);


  void Reset();

 private:
  const OnlBuffSourceOptions opts_;
  std::vector<BaseFloat> src_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlBuffSource);
};

} // namespace kaldi


#endif // KALDI_DEC_WRAP_DEC_WRAP_AUDIO_SOURCE_H_
