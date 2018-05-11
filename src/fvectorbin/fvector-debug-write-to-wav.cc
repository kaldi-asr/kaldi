#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fvector/fvector-perturb.h"
#include "feat/wave-reader.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Usage:  fvector-write-to-wav [options...] <chunk-rspecifier> <wave-path>\n";

    ParseOptions po(usage);
    BaseFloat sample_freq=16000; 
    po.Register("sample-frequency",&sample_freq, "sample-frequency of the wave.");
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string chunk_rspecifier = po.GetArg(1),
                wave_path = po.GetArg(2);

    SequentialBaseFloatMatrixReader chunk_reader(chunk_rspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !chunk_reader.Done(); chunk_reader.Next(), num_read++) {
      std::string key = chunk_reader.Key();
      // input_chunk has 3 lines.
      const Matrix<BaseFloat> &input_chunk = chunk_reader.Value();
      num_read++;
      for(int i=0; i<input_chunk.NumRows(); i++) {
        std::stringstream utt_id_new;
        utt_id_new << wave_path << '/' << key << '_' << i << ".wav";
        Output os(utt_id_new.str(), false);
        
        Matrix<BaseFloat> temp(1, input_chunk.NumCols());
        temp.CopyRowFromVec(input_chunk.Row(i),0);
        WaveData wave(sample_freq, temp);
        
        wave.Write(os.Stream());
        num_written++;
      }
    }
    KALDI_LOG << " Done " << num_written << " out of " << num_read
              << " utterances.";
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
