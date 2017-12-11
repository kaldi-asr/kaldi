// online2bin/extend-wav-with-silence.cc

// 2014  IMSL, PKU-HKUST (author: Wei Shi)
// 2015  Tom Ko

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"

namespace kaldi{
void FindQuietestSegment(const Vector<BaseFloat> &wav_in,
                         BaseFloat samp_rate,
                         Vector<BaseFloat> *wav_sil,
                         BaseFloat search_dur = 0.5,
                         BaseFloat seg_dur = 0.1,
                         BaseFloat seg_shift_dur = 0.05);

void ExtendWaveWithSilence(const Vector<BaseFloat> &wav_in,
                           BaseFloat samp_rate,
                           Vector<BaseFloat> *wav_out,
                           BaseFloat sil_search_len,
                           BaseFloat sil_extract_len,
                           BaseFloat sil_extract_shift);

}


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Extend wave data with a fairly long silence at the end (e.g. 5 seconds).\n"
        "The input waveforms are assumed having silences at the begin/end and those\n"
        "segments are extracted and appended to the end of the utterance.\n"
        "Note this is for use in testing endpointing in decoding.\n"
        "\n"
        "Usage: extend-wav-with-silence [options] <wav-rspecifier> <wav-wspecifier>\n";

    ParseOptions po(usage);
    BaseFloat sil_len = 5.0,
      sil_search_len = 0.5,
      sil_extract_len = 0.05,
      sil_extract_shift = 0.025;
    po.Register("extra-silence-length", &sil_len, "the length of silence that will be "
                "appended to the end of each waveform, in seconds.");
    po.Register("silence-search-length", &sil_search_len, "the length at the beginning "
                "or end of each waveform in which to search for the quietest segment of "
                "silence, in seconds.");
    po.Register("silence-extract-length", &sil_extract_len, "the length of silence segments "
                "to be extracted from the waveform, which must be smaller than silence-"
                "search-length, in seconds.");
    po.Register("silence-extract-shift", &sil_extract_shift, "the shift length when searching "
                "for segments of silences, typically samller than silence-extract-length, "
                "in seconds.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1);
    std::string wav_wspecifier = po.GetArg(2);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    TableWriter<WaveHolder> writer(wav_wspecifier);
    int32 num_success = 0;

    for(; !reader.Done(); reader.Next()){
      std::string wav_key = reader.Key();
      const WaveData &wave = reader.Value();
      BaseFloat samp_freq = wave.SampFreq();  // read sampling fequency
      const Matrix<BaseFloat> &wave_data = wave.Data();
      int32 num_chan = wave_data.NumRows(),       // number of channels in recording
        num_ext_samp  = (int32)(samp_freq * sil_len); // number of samples that will be extended
      KALDI_ASSERT(num_ext_samp > 0);
      Matrix<BaseFloat> new_wave(wave_data.NumRows(), wave_data.NumCols() + num_ext_samp);
      for(int32 i = 0; i < num_chan; i++){
        Vector<BaseFloat> wav_this_chan(wave_data.Row(i));
        Vector<BaseFloat> wav_extend(wav_this_chan.Dim() + num_ext_samp);
        ExtendWaveWithSilence(wav_this_chan, samp_freq, &wav_extend,
                              sil_search_len, sil_extract_len, sil_extract_shift);
        KALDI_ASSERT(wav_extend.Dim() == wav_this_chan.Dim() + num_ext_samp);
        new_wave.CopyRowFromVec(wav_extend, i);
      }
      WaveData wave_out(samp_freq, new_wave);
      writer.Write(wav_key, wave_out);
      num_success++;
    }
    KALDI_LOG << "Successfully extended " << num_success << " files.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

namespace kaldi{

void ExtendWaveWithSilence(const Vector<BaseFloat> &wav_in,
                           BaseFloat samp_rate,
                           Vector<BaseFloat> *wav_out,
                           BaseFloat sil_search_len,
                           BaseFloat sil_extract_len,
                           BaseFloat sil_extract_shift){
  Vector<BaseFloat> quietest_seg;
  FindQuietestSegment(wav_in, samp_rate, &quietest_seg,
                      sil_search_len, sil_extract_len, sil_extract_shift);

  int32 window_size = quietest_seg.Dim(),
    window_size_half = window_size / 2;
  KALDI_ASSERT(window_size > 0);
  Vector<BaseFloat> window(window_size);
  Vector<BaseFloat> windowed_silence(window_size);
  Vector<BaseFloat> half_window(window_size_half);
  for(int32 i = 0; i < window.Dim(); i++){
    BaseFloat i_fl = static_cast<BaseFloat>(i);
    window(i) = 0.54 - 0.46*cos(M_2PI * i_fl / (window_size-1));
  }
  half_window = window.Range(window_size_half, window_size_half);
  windowed_silence.AddVecVec(1.0, window, quietest_seg, 0.0);

  wav_out->Range(0, wav_in.Dim()).CopyFromVec(wav_in);
  SubVector<BaseFloat> wav_ext(*wav_out, wav_in.Dim() - window_size_half,
                                wav_out->Dim() - wav_in.Dim() + window_size_half);
  for(int32 i = 0; i < window_size_half; i++)    // windowing the first half window
    wav_ext(i) *= half_window(i);
  
  int32 tmp_offset = 0;
  for(; tmp_offset + window_size < wav_ext.Dim();) {
    wav_ext.Range(tmp_offset, window_size).AddVec(1.0, windowed_silence);
    tmp_offset += window_size_half;
  }

  for(int32 i = tmp_offset; i < wav_ext.Dim(); i++)
    wav_ext(i) += windowed_silence(i-tmp_offset);

}

// Try to find the quietest seq_dur(default 0.1) second segment in the
// search_dur(default 0.5) seconds at the beginning and the end
// of input waveform by simply find a segment with the least energy.
void FindQuietestSegment(const Vector<BaseFloat> &wav_in,
                         BaseFloat samp_rate,
                         Vector<BaseFloat> *wav_sil,
                         BaseFloat search_dur,
                         BaseFloat seg_dur,
                         BaseFloat seg_shift_dur){
  KALDI_ASSERT(seg_dur < search_dur);

  int32 search_len = (int32) (search_dur * samp_rate),
    seg_len = (int32) (seg_dur * samp_rate),
    seg_shift = (int32) (seg_shift_dur *samp_rate),
    start = 0;
  double min_energy;
  Vector<BaseFloat> wav_min_energy;
  Vector<BaseFloat> seg_tmp(wav_in.Range(0, seg_len));
  wav_min_energy = seg_tmp;
  min_energy = VecVec(seg_tmp, seg_tmp);
  for(start = 0; start + seg_len < search_len; ){
    SubVector<BaseFloat> seg_this(wav_in, start, seg_len);
    seg_tmp = seg_this;
    double energy_this = VecVec(seg_this, seg_this);
    if(energy_this < min_energy && energy_this > 0.0){
      min_energy = energy_this;
      wav_min_energy = seg_tmp;
    }
    start += seg_shift;
  }

  for(start = wav_in.Dim() - search_len; start + seg_len < wav_in.Dim(); ){
    SubVector<BaseFloat> seg_this(wav_in, start, seg_len);
    seg_tmp = seg_this;
    double energy_this = VecVec(seg_this, seg_this);
    if(energy_this < min_energy && energy_this > 0.0){
      min_energy = energy_this;
      wav_min_energy = seg_tmp;
    }
    start += seg_shift;
  }

  if (min_energy == 0.0) {
    KALDI_WARN << "Zero energy silence being used.";
  }
  *wav_sil = wav_min_energy;
}

}
