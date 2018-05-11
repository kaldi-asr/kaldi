#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fvector/fvector-perturb.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Perturb the chunk data. Each input chunk is a four consecutive rows matrix(S1, S2, N1, N2).\n"
        "According to the setup, use (0-4) kinds of perturbation opertation, and then each output chunk \n"
        "is a 2 consecutive rows of output matrix.\n"
        "The two rows come from the same source wavform signal, but now they are different.\n"
        "Usage:  fvector-add-noise [options...] <chunk-rspecifier> <perturbed-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    FvectorPerturbOptions perturb_opts;
    perturb_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string chunk_rspecifier = po.GetArg(1),
      perturbed_chunk_rspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader chunk_reader(chunk_rspecifier);
    BaseFloatMatrixWriter perturbed_chunk_writer(perturbed_chunk_rspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !chunk_reader.Done(); chunk_reader.Next(), num_read++) {
      std::string key = chunk_reader.Key();
      // input_chunk has 3 lines.
      const Matrix<BaseFloat> &input_chunk = chunk_reader.Value();
      // whole_chunk has 4 lines, it copies the first line and will be operate.
      Matrix<BaseFloat> whole_chunk(4, input_chunk.NumCols());
      // For here, we copy the first line. So in the "whole_chunk" the first
      // two lines come from the same source wavform signal. And the third/forth
      // line is the random noise.
      MatrixIndexT indices[4] = {0, 0, 1, 2};
      whole_chunk.CopyRows(input_chunk, indices);
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
