// online2/online-speex-wrapper.cc

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

#include <cstring>
#include "online-speex-wrapper.h"

namespace kaldi {

OnlineSpeexEncoder::OnlineSpeexEncoder(const SpeexOptions &config):
 input_finished_(false) {
  speex_frame_size_ = config.speex_bits_frame_size;
  speex_encoded_frame_size_ = config.speex_wave_frame_size;
  sample_rate_ = config.sample_rate;

  if(sizeof(speex_bits_) == 1) {
    KALDI_ERR << "OnlineSpeexEncoder called but Speex not installed."
      << "You should run tools/extras/install_speex.sh first, then "
      << "re-run configure in src/ and then make Kaldi again.\n";
  }

#ifdef HAVE_SPEEX
  speex_state_ = speex_encoder_init(&speex_wb_mode);  // init speex with wideband mode
  int32 tmp = config.speex_quality;
  speex_encoder_ctl(speex_state_, SPEEX_SET_QUALITY, &tmp);
  tmp = (int)(sample_rate_);
  speex_encoder_ctl(speex_state_, SPEEX_SET_SAMPLING_RATE, &tmp);
  speex_bits_init(&speex_bits_);
#endif
}

OnlineSpeexEncoder::~OnlineSpeexEncoder() {
#ifdef HAVE_SPEEX
  speex_bits_destroy(&speex_bits_);
  speex_encoder_destroy(speex_state_);
#endif
}

void OnlineSpeexEncoder::AcceptWaveform(
 int32 sample_rate, const VectorBase<BaseFloat> &waveform) {
  if (waveform.Dim() == 0) {
    return;   // Nothing to do.
  }
  if (input_finished_) {
    KALDI_ERR << "AcceptWaveform called after InputFinished() was called.";
  }
  if (sample_rate != sample_rate_) {
    KALDI_ERR << "Sampling frequency mismatch, expected "
              << sample_rate_ << ", got " << sample_rate;
  }

  Vector<BaseFloat> appended_wave;
  const VectorBase<BaseFloat> &wave_to_use = (waveform_remainder_.Dim() != 0 ?
                                              appended_wave : waveform);
  if (waveform_remainder_.Dim() != 0) {
    appended_wave.Resize(waveform_remainder_.Dim() +
                         waveform.Dim());
    appended_wave.Range(0, waveform_remainder_.Dim()).CopyFromVec(
        waveform_remainder_);
    appended_wave.Range(waveform_remainder_.Dim(),
                        waveform.Dim()).CopyFromVec(waveform);
  }
  waveform_remainder_.Resize(0);

  std::vector<char> spx_bits;
  Encode(wave_to_use, &spx_bits);

  if (spx_bits.size() > 0) {
    speex_encoded_char_bits_.insert(speex_encoded_char_bits_.end(),
                                    spx_bits.begin(), spx_bits.end());
  }
}

// Deal with the last frame, pad zeros
void OnlineSpeexEncoder::InputFinished() {
  input_finished_ = true;

  int32 dim = waveform_remainder_.Dim();
  if (dim != 0) {
    KALDI_ASSERT(dim <= speex_encoded_frame_size_);
    Vector<BaseFloat> wave_last(speex_encoded_frame_size_);
    std::vector<char> spx_bits;
    wave_last.Range(0, dim).CopyFromVec(waveform_remainder_);
    Encode(wave_last, &spx_bits);

    speex_encoded_char_bits_.insert(speex_encoded_char_bits_.end(),
                                    spx_bits.begin(), spx_bits.end());
  }
}

void OnlineSpeexEncoder::Encode(const VectorBase<BaseFloat> &wave,
 std::vector<char> *speex_encoder_bits) {
  if (wave.Dim() == 0) {
    return;
  }

  int32 to_encode = wave.Dim();
  int32 has_encode = 0;
  char cbits[200];
  std::vector<char> encoded_bits;
  while (to_encode > speex_encoded_frame_size_) {
    SubVector<BaseFloat> wave_frame(wave, has_encode,
      speex_encoded_frame_size_);
    int32 nbytes = 0;
#ifdef HAVE_SPEEX
    speex_bits_reset(&speex_bits_);
    speex_encode(speex_state_, wave_frame.Data(), &speex_bits_);
    nbytes = speex_bits_nbytes(&speex_bits_);
    if (nbytes != speex_frame_size_) {
      KALDI_ERR << "The number of bytes of Speex encoded frame mismatch,"
        << "expected " << speex_frame_size_ << ", got " << nbytes;
    }
    nbytes = speex_bits_write(&speex_bits_, cbits, 200);
#endif

    int32 encoded_bits_len = encoded_bits.size();
    encoded_bits.resize(encoded_bits_len + nbytes);
    for (int32 i = 0; i < nbytes; i++) {
      encoded_bits[i+encoded_bits_len] = cbits[i];
    }

    has_encode += speex_encoded_frame_size_;
    to_encode -= speex_encoded_frame_size_;
  }

  if (to_encode > 0) {
    SubVector<BaseFloat> wave_left(wave, has_encode, to_encode);
    int32 dim = waveform_remainder_.Dim();
    if (dim != 0) {
      waveform_remainder_.Resize(dim + to_encode, kCopyData);
      waveform_remainder_.Range(dim, to_encode).CopyFromVec(wave_left);
    } else {
      waveform_remainder_ = wave_left;
    }
  }

  *speex_encoder_bits = encoded_bits;
}


OnlineSpeexDecoder::OnlineSpeexDecoder(const SpeexOptions &config) {
  speex_frame_size_ = config.speex_bits_frame_size;
  speex_decoded_frame_size_ = config.speex_wave_frame_size;

  if(sizeof(speex_bits_) == 1) {
    KALDI_ERR << "OnlineSpeexEncoder called but Speex not installed."
      << "You should run tools/extras/install_speex.sh first, then "
      << "re-run configure in src/ and then make Kaldi again.\n";
  }

#ifdef HAVE_SPEEX
  speex_state_ = speex_decoder_init(&speex_wb_mode);  // init speex with wideband mode
  int32 tmp = config.speex_quality;
  speex_decoder_ctl(speex_state_, SPEEX_SET_QUALITY, &tmp);
  tmp = (int)config.sample_rate;
  speex_decoder_ctl(speex_state_, SPEEX_SET_SAMPLING_RATE, &tmp);
  speex_bits_init(&speex_bits_);
#endif
}

OnlineSpeexDecoder::~OnlineSpeexDecoder() {
#ifdef HAVE_SPEEX
  speex_decoder_destroy(speex_state_);
  speex_bits_destroy(&speex_bits_);
#endif
}

void OnlineSpeexDecoder::AcceptSpeexBits(const std::vector<char> &spx_enc_bits) {
  if (spx_enc_bits.size() == 0) {
    return;                 // Nothing to do
  }

  std::vector<char> appended_bits;
  const std::vector<char> &bits_to_use = (speex_bits_remainder_.size() != 0 ?
                                              appended_bits : spx_enc_bits);
  if (speex_bits_remainder_.size() != 0) {
    appended_bits.insert(appended_bits.end(), speex_bits_remainder_.begin(),
                         speex_bits_remainder_.end());
    appended_bits.insert(appended_bits.end(), spx_enc_bits.begin(),
                         spx_enc_bits.end());
  }
  speex_bits_remainder_.clear();

  Vector<BaseFloat> waveform;
  Decode(bits_to_use, &waveform);
  if (waveform.Dim() == 0) {
    // Got nothing, maybe the decode has failed
    return;
  }
  int32 last_wav_size = waveform_.Dim();
  waveform_.Resize(last_wav_size + waveform.Dim(), kCopyData);
  waveform_.Range(last_wav_size, waveform.Dim()).CopyFromVec(waveform);
}

void OnlineSpeexDecoder::Decode(const std::vector<char> &speex_char_bits,
                                Vector<BaseFloat> *decoded_wav) {
  if (speex_char_bits.size() < speex_frame_size_) {
    return;                // Nothing to do, should never reach this
  }
  decoded_wav->Resize(0);

  char *cbits = new char[speex_frame_size_ + 10]();
  BaseFloat *wav = new BaseFloat[speex_decoded_frame_size_]();
  int32 to_decode = speex_char_bits.size();
  int32 has_decode = 0;

  while(to_decode > speex_frame_size_){
    memcpy(cbits, &speex_char_bits[has_decode], speex_frame_size_);
#ifdef HAVE_SPEEX
    speex_bits_read_from(&speex_bits_, cbits, speex_frame_size_);
    speex_decode(speex_state_, &speex_bits_, wav);
#endif

    int32 dim = decoded_wav->Dim();  // expanding decoded_wav each frame
    decoded_wav->Resize(dim + speex_decoded_frame_size_, kCopyData);
    // Cannot use CopyFromPtr at this moment
    // decoded_wav->Range(dim, speex_decoded_frame_size_).
    //  CopyFromPtr(wav, speex_decoded_frame_size_);
    for (int32 i = 0; i < speex_decoded_frame_size_; i++) {
      (*decoded_wav)(i+dim) = wav[i];
    }

    has_decode += speex_frame_size_;
    to_decode  -= speex_frame_size_;
  }

  if (to_decode > 0) {
    speex_bits_remainder_.insert(speex_bits_remainder_.end(),
      speex_char_bits.begin() + has_decode, speex_char_bits.end());
  }

  delete []cbits;
  delete []wav;
}

}
// namespace kaldi
