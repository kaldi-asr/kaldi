#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fvector/fvector-perturb.h"
#include "feat/wave-reader.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Usage: fvector-wav-to-vector [options...] <wav-rspecifier> <wave-wspecifier>\n";

    ParseOptions po(usage);
    BaseFloat sample_freq=16000; 
    po.Register("sample-frequency",&sample_freq, "sample-frequency of the wave.");
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1),
                output_wspecifier = po.GetArg(2);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatVectorWriter kaldi_writer(output_wspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !reader.Done(); reader.Next(), num_read++) {
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      Vector<BaseFloat> waveform(SubVector<BaseFloat>(wave_data.Data(), 0));
      kaldi_writer.Write(utt, waveform);
    }
    KALDI_LOG << " Done " << num_written << " out of " << num_read
              << " utterances.";
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
