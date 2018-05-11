#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fvector/fvector-perturb.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Perturb the chunk data. We read in one source chunk and two noise chunks separately\n"
        "According to the setup, use (0-4) kinds of perturbation opertation, and then each output chunk \n"
        "is a 2 consecutive rows of output matrix.\n"
        "The two rows come from the same source wavform signal, but now they are different.\n"
        "Usage:  fvector-add-noise [options...] <source-chunk-rspecifier> <noise-chunk-respecifier> <perturbed-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    FvectorPerturbOptions perturb_opts;
    perturb_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string chunk_rspecifier = po.GetArg(1),
      noise_chunk_rspecifier = po.GetArg(2),
      perturbed_chunk_rspecifier = po.GetArg(3);

    SequentialBaseFloatVectorReader chunk_reader(chunk_rspecifier);
    SequentialBaseFloatVectorReader noise_chunk_reader(noise_chunk_rspecifier);
    BaseFloatMatrixWriter perturbed_chunk_writer(perturbed_chunk_rspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !chunk_reader.Done(); chunk_reader.Next(), num_read++) {
      // Read 2 noise chunks
      if (noise_chunk_reader.Done()) {
        KALDI_ERR << "Noise chunk is too short to enough";
      }
      const Vector<BaseFloat> noise1_chunk(noise_chunk_reader.Value());
      noise_chunk_reader.Next();
      const Vector<BaseFloat> noise2_chunk(noise_chunk_reader.Value());
      noise_chunk_reader.Next();

      std::string key = chunk_reader.Key();
      // input_chunk has 3 lines.
      const Vector<BaseFloat> &input_chunk = chunk_reader.Value();
      // whole_chunk has 4 lines, it copies the first line and will be operate.
      Matrix<BaseFloat> whole_chunk(4, input_chunk.Dim());
      // For here, we copy the first line. So in the "whole_chunk" the first
      // two lines come from the same source wavform signal. And the third/forth
      // line is the random noise.
      whole_chunk.CopyRowFromVec(input_chunk, 0);
      whole_chunk.CopyRowFromVec(input_chunk, 1);
      whole_chunk.CopyRowFromVec(noise1_chunk, 2);
      whole_chunk.CopyRowFromVec(noise2_chunk, 3);
      Matrix<BaseFloat> perturbed_chunk;

      // the class FvectorPerturb conduct the different perturb operation.
      FvectorPerturb perturb_fvector(perturb_opts);
      perturb_fvector.ApplyPerturbation(whole_chunk, &perturbed_chunk);
      perturbed_chunk_writer.Write(key, perturbed_chunk);
      num_written++;
    }
    KALDI_LOG << " Done " << num_written << " out of " << num_read
              << " utterances.";
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
