// online2/online-speex-wrapper.h

// Copyright   2014  IMSL, PKU-HKUST (author: Wei Shi)

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


#ifndef KALDI_ONLINE2_ONLINE_SPEEX_WRAPPER_H_
#define KALDI_ONLINE2_ONLINE_SPEEX_WRAPPER_H_

#ifdef HAVE_SPEEX
  #include <speex/speex.h>
  typedef SpeexBits SPEEXBITS;
#else
  typedef char SPEEXBITS;
#endif

#include "matrix/kaldi-vector.h"
#include "itf/options-itf.h"

namespace kaldi {

struct SpeexOptions {
  /// The sample frequency of the waveform, it decides which Speex mode
  /// to use. Often 8kHz---> narrow band, 16kHz---> wide band and 32kHz
  /// ---> ultra wide band
  BaseFloat sample_rate;

  /// Ranges from 0 to 10, the higher the quality is better. In my preliminary
  /// tests with the RM recipe, if set it to 8, I observed the WER incresed by
  /// 0.1%; while set it to 10, the WER almost kept unchanged.
  int32 speex_quality;

  /// In bytes.
  /// Should be set according to speex_quality. Just name a few here(wideband):
  ///     quality            size(in bytes)
  ///        8                  70
  ///        9                  86
  ///        10                 106
  int32 speex_bits_frame_size;

  /// In samples.
  /// The Speex toolkit uses a 20ms long window by default
  int32 speex_wave_frame_size;

  SpeexOptions(): sample_rate(16000.0),
                  speex_quality(10),
                  speex_bits_frame_size(106),
                  speex_wave_frame_size(320) { }

  void Register(OptionsItf *opts) {
    opts->Register("sample-rate", &sample_rate, "Sample frequency of the waveform.");
    opts->Register("speex-quality", &speex_quality, "Speex speech quality.");
    opts->Register("speex-bits-frame-size", &speex_bits_frame_size,
                   "#bytes of each Speex compressed frame.");
    opts->Register("speex-wave-frame-size", &speex_wave_frame_size,
                   "#samples of each waveform frame.");
  }
};

class OnlineSpeexEncoder {
  public:
    OnlineSpeexEncoder(const SpeexOptions &config);
    ~OnlineSpeexEncoder();

    void AcceptWaveform(int32 sample_rate,
           const VectorBase<BaseFloat> &waveform);

    void InputFinished();

    void GetSpeexBits(std::vector<char> *spx_bits) {  // call it after AcceptWaveform
      *spx_bits = speex_encoded_char_bits_;
      speex_encoded_char_bits_.clear();
    }
  private:
    int32 speex_frame_size_;  // in bytes, will be different according to the quality
    int32 speex_encoded_frame_size_;  // in samples, typically 320 in wideband mode, 16kHz
#ifdef HAVE_SPEEX
    void *speex_state_;  // Holds the state of the speex encoder
#endif
    SPEEXBITS speex_bits_;

    Vector<BaseFloat> waveform_remainder_;      // Holds the waveform that have not been processed

    // Holds the Speex-encoded char bits, will be peaked by GetSpeexBits().
    // We use a vector container rather than a char-type pointer because
    // it's a little easier to expand.
    std::vector<char> speex_encoded_char_bits_;

    BaseFloat sample_rate_;
    bool input_finished_;

    void Encode(const VectorBase<BaseFloat> &wave,
                std::vector<char> *speex_encoder_bits) ;
};

class OnlineSpeexDecoder {
  public:
    OnlineSpeexDecoder(const SpeexOptions &config);
    ~OnlineSpeexDecoder();

    void AcceptSpeexBits(const std::vector<char> &spx_enc_bits);

    void GetWaveform(Vector<BaseFloat> *waveform) {  // call it after AcceptSpeexBits
      *waveform = waveform_;
      waveform_.Resize(0);
    }
  private:
    int32 speex_frame_size_;  // in bytes, will be different according to the quality
    int32 speex_decoded_frame_size_;  // in samples, typically 320 in wideband mode, 16kHz

#ifdef HAVE_SPEEX
    void *speex_state_;  // Holds the state of the speex decoder
#endif
    SPEEXBITS speex_bits_;


    Vector<BaseFloat> waveform_;  // Holds the waveform decoded from speex bits
    std::vector<char> speex_bits_remainder_;

    void Decode(const std::vector<char> &speex_char_bits,
                Vector<BaseFloat> *decoded_wav) ;
};

}  // namespace kaldi

#endif  // KALDI_ONLINE2_ONLINE_SPEEX_WRAPPER_H_
