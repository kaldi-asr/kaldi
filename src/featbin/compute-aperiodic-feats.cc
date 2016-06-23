// featbin/compute-aperiodic-feats.cc

// Copyright 2016        CereProc Ltd (author: Blaise Potard)
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
#include "feat/feature-aperiodic.h"
#include "feat/wave-reader.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Apply Aperiodic Energy extractor, starting from wav input.\n"
        "Output is n-dimensional features consisting of log energy\n"
        "from a set of warped frequency bands. This correspond to the\n"
        "energy bands of the noise spectrum as a ratio over the original\n"
        "spectrum. The values on each band will always be negative or 0.\n"
        "Usage: compute-aperiodic-feats [options...] <wav-rspecifier> "
        "<f0-rspecifier> <feats-wspecifier>\n"
        "e.g. compute-aperiodic-feats scp:wav.scp scp:f0.scp ark:- \n"
        "\n"
        "See also: compute-kaldi-pitch-feats\n";

    bool f0_first = false;
    ParseOptions po(usage);
    AperiodicEnergyOptions aperiodic_opts;

    aperiodic_opts.Register(&po);
    po.Register("pitch-first", &f0_first, "Assume first column is pitch, "
                "second is POV, for compatibility with get_f0. ");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1),
        f0_rspecifier = po.GetArg(2),
        feat_wspecifier = po.GetArg(3);

    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    RandomAccessBaseFloatMatrixReader f0_reader(f0_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    int32 num_done = 0, num_err = 0;
    int f0_col = 1, pov_col = 0;
    // Reverse column order if f0_first
    if (f0_first) {
      f0_col = 0;
      pov_col = 1;
    }
    for (; !wav_reader.Done(); wav_reader.Next()) {
      // TODO(BP): check keys are matching
      std::string utt = wav_reader.Key();
      const WaveData &wave_data = wav_reader.Value();
      const Matrix<BaseFloat> &pitch_features = f0_reader.Value(utt);

      AperiodicEnergy ap_energy(aperiodic_opts);

      // Get first channel wave data
      SubVector<BaseFloat> waveform(wave_data.Data(), 0);
      int32 num_frames = NumFrames(waveform.Dim(), aperiodic_opts.frame_opts);
      int32 num_frames_f0 = pitch_features.NumRows();
      KALDI_LOG << "NF = " << num_frames << "; NR = " << num_frames_f0;
      /*KALDI_ASSERT(num_frames_f0 == num_frames);*/

      // F0 is read from second or first column
      Vector<BaseFloat> f0(num_frames_f0);
      f0.CopyColFromMat(pitch_features, f0_col);
      // Pov is read from the other column
      Vector<BaseFloat> pov(num_frames_f0);
      pov.CopyColFromMat(pitch_features, pov_col);
      if (num_frames_f0 != num_frames) {
        f0.Resize(num_frames, kCopyData);
        pov.Resize(num_frames, kCopyData);
      }

      // Output data
      Matrix<BaseFloat> features;
      try {
        ap_energy.Compute(waveform, pov, f0, &features, NULL);
      } catch(...) {
        KALDI_WARN << "Failed to compute bndap for utterance "
                   << utt;
        num_err++;
        continue;
      }

      feat_writer.Write(utt, features);
      if (num_done % 50 == 0 && num_done != 0)
        KALDI_VLOG(2) << "Processed " << num_done << " utterances";
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

