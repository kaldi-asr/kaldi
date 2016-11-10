// online2bin/compress-uncompress-speex.cc

// 2014  IMSL, PKU-HKUST (author: Wei Shi)

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

#include <vector>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "online2/online-speex-wrapper.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Demonstrating how to use the Speex wrapper in Kaldi by compressing input waveforms \n"
        "chunk by chunk and then decompressing them.\n"
        "\n"
        "Usage: compress-uncompress-speex [options] <wav-rspecifier> <wav-wspecifier>\n";

    ParseOptions po(usage);
    SpeexOptions spx_config;
    BaseFloat chunk_length_secs = 0.05;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");

    spx_config.Register(&po);

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
      int32 num_chan = wave_data.NumRows();       // number of channels in recording

      Matrix<BaseFloat> new_wave(wave_data.NumRows(), wave_data.NumCols());
      for(int32 i = 0; i < num_chan; i++){
        OnlineSpeexEncoder spx_encoder(spx_config);
        OnlineSpeexDecoder spx_decoder(spx_config);
        Vector<BaseFloat> wav_this_chan(wave_data.Row(i));
        Vector<BaseFloat> wav_decode(wav_this_chan.Dim());

        int32 samp_offset = 0, decode_sample_offset = 0,
          max_samp = samp_freq * chunk_length_secs;
        while (samp_offset < wav_this_chan.Dim()) {
          int32 this_num_samp = max_samp;
          if (this_num_samp > wav_this_chan.Dim() - samp_offset)
            this_num_samp = wav_this_chan.Dim() - samp_offset;
          SubVector<BaseFloat> wave_part(wav_this_chan, samp_offset,
                                         this_num_samp);

          spx_encoder.AcceptWaveform(samp_freq, wave_part);
          if (this_num_samp == wav_this_chan.Dim() - samp_offset)  // no more input.
            spx_encoder.InputFinished();
          std::vector<char> speex_bits_part;
          spx_encoder.GetSpeexBits(&speex_bits_part);

          Vector<BaseFloat> wave_part_spx;
          spx_decoder.AcceptSpeexBits(speex_bits_part);
          spx_decoder.GetWaveform(&wave_part_spx);

          int32 decode_num_samp = wave_part_spx.Dim();
          if (decode_sample_offset + decode_num_samp > wav_this_chan.Dim()) {
            int32 num_samp_last = wav_this_chan.Dim() - decode_sample_offset;
            SubVector<BaseFloat> wave_part_tmp(wave_part_spx,0,num_samp_last);

            wav_decode.Range(decode_sample_offset, num_samp_last).
              CopyFromVec(wave_part_tmp);
            decode_sample_offset += num_samp_last;
          } else {
            wav_decode.Range(decode_sample_offset, decode_num_samp).
              CopyFromVec(wave_part_spx);
            decode_sample_offset += wave_part_spx.Dim();
          }

          samp_offset += this_num_samp;
        }

        new_wave.CopyRowFromVec(wav_decode, i);
      }
      WaveData wave_out(samp_freq, new_wave);
      writer.Write(wav_key, wave_out);
      num_success++;
    }
    KALDI_LOG << "Successfully processed " << num_success << " files.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

