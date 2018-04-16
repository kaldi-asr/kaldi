// feat/compute-raw-frame-feats.cc

// Copyright 2015 Pegah Ghahremani

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/feature-functions.h"
#include "feat/feature-common.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
      "Creates raw feature files, starting from wave input and generate output as part of "
      "raw-wave with specific duration and no overlap.\n"
      "Some post-processing can be applied on output."
      "Usage: compute-raw-frame-feats.cc [options...] <wave-rspecifier> <feats-wspecifier>\n";

    ParseOptions po(usage);
    FrameExtractionOptions raw_opts;
    raw_opts.frame_shift_ms = 10.0;
    raw_opts.frame_length_ms = 10.0;
    raw_opts.window_type = "rectangular";
    raw_opts.round_to_power_of_two = false;
    raw_opts.remove_dc_offset = false;
    raw_opts.preemph_coeff = 0.0;
    raw_opts.dither = 0.0;
    // Register raw frame extraction options
    raw_opts.Register(&po);
    bool remove_dc = true,
      loudness_equalize = true;
    BaseFloat low_rms =0.2, high_rms = 0.2,
      scale_wav = 0.0;
    po.Register("remove-global-dc-offset", &remove_dc, "If true, subtract mean from waveform on each wave");
    po.Register("loudness-equalize", &loudness_equalize, "If true, variance-normalization "
                "is applied on output-wave");
    po.Register("low-rms", &low_rms, "The lowest variance of output-wave, where the variance"
                "is randomly set between [low-rms, high-rms], and "
                " the loudness of wave is equal to this randomly chosen rms.");
    po.Register("high-rms", &high_rms, "The highest variance of output-wave, where the variance"
                "is randomly set between [low-rms, high-rms], and "
                " the loudness of wave is equal to this randomly chosen rms.");
    po.Register("scale-wav", &scale_wav, "If non-zero, the raw waveform scaled using that.");
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string wav_rspecifier = po.GetArg(1);
    std::string output_wspecifier = po.GetArg(2);

    SequentialTableReader<WaveHolder> wave_reader(wav_rspecifier);
    BaseFloatMatrixWriter feat_writer(output_wspecifier);


    int32 num_done = 0, num_err = 0;
    for (; !wave_reader.Done(); wave_reader.Next()) {
      std::string utt = wave_reader.Key();
      const WaveData &wave_data = wave_reader.Value();

      // The channel is not configurable and it is better to extract it
      // on command line using sox or...
      int32 num_chan = wave_data.Data().NumRows();
      if (num_chan != 1)
        KALDI_WARN << "You have data with "
                   << num_chan  << " channels; defaulting to zero";

      SubVector<BaseFloat> waveform(wave_data.Data(), 0);
      Vector<BaseFloat> input(waveform.Dim());
      input.CopyRowFromMat(wave_data.Data(), 0);
      BaseFloat mean = waveform.Sum() / waveform.Dim();
      // compute variance
      input.Add(-mean);
      BaseFloat variance = std::pow(VecVec(input, input) / waveform.Dim(), 0.5);

      // remove DC offset
      if (remove_dc)
        waveform.Add(-1.0 * mean);

      // apply variance normalization
      BaseFloat target_rms =  low_rms + RandUniform() * (high_rms - low_rms);
      if (loudness_equalize && variance != 0)
        waveform.Scale(target_rms * 1.0 / variance);

      if (scale_wav != 0.0)
        waveform.Scale(scale_wav);

      Matrix<BaseFloat> raw_mat;
      try {
        FeatureWindowFunction window_function(raw_opts);
        int32 rows_out = NumFrames(waveform.Dim(), raw_opts),
          cols_out = raw_opts.WindowSize();
        raw_mat.Resize(rows_out, cols_out);
        for (int32 frame = 0; frame < rows_out; frame++) {
          Vector<BaseFloat> raw_feat(cols_out);
          ExtractWindow(0, waveform, frame, raw_opts,
                        window_function, &raw_feat);
          raw_mat.CopyRowFromVec(raw_feat, frame);

        }
        //ComputeAndProcessRawSignal(raw_opts, waveform, &raw_feats);
      } catch (...) {
        KALDI_WARN << "Failed to extract raw-feats for utterance "
                   << utt;
        num_err++;
        continue;
      }

      feat_writer.Write(utt, raw_mat);
      if (num_done % 50 == 0 && num_done != 0)
        KALDI_VLOG(2) << "Processed " << num_done << " utterances";
      num_done++;
    }
    KALDI_LOG << " Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
