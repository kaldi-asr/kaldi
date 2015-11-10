// featbin/compute-inverse-stft.cc

// Copyright 2015  Hakan Erdogan

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
#include "feat/inverse-stft.h"
#include "feat/wave-reader.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        const char *usage =
            "Convert stft feature files to wave files by inverse fft and overlap-add.\n"
            "Usage:  compute-inverse-stft [options...] <feats-rspecifier> <wav-wspecifier>\n";
        std::string wav_durations_rspecifier;

        // construct all the global objects
        ParseOptions po(usage);
        StftOptions stft_opts;
        // Define defaults for gobal options

        // Register the option struct
        stft_opts.Register(&po);
        // Register the options
        po.Register("wav-durations", &wav_durations_rspecifier, "Durations for output wave files ");

        // OPTION PARSING ..........................................................
        //

        // parse options (+filling the registered variables)
        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string feats_rspecifier = po.GetArg(1);
        std::string wav_wspecifier = po.GetArg(2);

        Istft istft(stft_opts);

        SequentialBaseFloatMatrixReader reader(feats_rspecifier);
        RandomAccessBaseFloatReader dur_reader(wav_durations_rspecifier);

        TableWriter<WaveHolder> writer(wav_wspecifier);

        int32 samp_rate = stft_opts.frame_opts.samp_freq;

        int32 num_utts = 0, num_success = 0;
        for (; !reader.Done(); reader.Next()) {
            num_utts++;
            std::string utt = reader.Key();
            const Matrix<BaseFloat> &stftdata_matrix(reader.Value());

            Vector<BaseFloat> wave_vector; // no init here

            int32 wav_duration_in_samples;  // get wav durations
            if (wav_durations_rspecifier != "") {
                if (!dur_reader.HasKey(utt)) {
                    KALDI_WARN << "No duration entry for utterance-id "
                               << utt;
                    continue;
                }
                wav_duration_in_samples = samp_rate * dur_reader.Value(utt);
            } else {
                wav_duration_in_samples = -1; // do not specify duration and let istft figure it out from frames
            }

            istft.Compute(stftdata_matrix, &wave_vector, wav_duration_in_samples); //note, third arg optional

            // convert wave_vector to single channel WaveData
            WaveData wave(samp_rate, wave_vector);
            writer.Write(utt, wave); // write data in wave format.
            num_success++;
        }
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
