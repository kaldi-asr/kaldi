#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fvector/fvector-perturb.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Perturb the chunk data. Each time the input is a source chunk block and\n"
        "two noise chunk block. The two noise blocks are added to source block separately,\n"
        "and then we maybe do volume perturbate, speed perturb or time shift.\n"
        "At last, the output is a matrix. Each two consecutive rows of the matrix\n"
        "come from same source wave, but were used different perturbation method.\n"
        "Usage: fvector-add-noise-block [options...] <source-chunk-rspecifier> "
        "<noise1-chunk-rspecifier> <noise2-chunk-respecifer> <perturbed-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    FvectorPerturbOptions perturb_opts;
    perturb_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string source_chunk_rspecifier = po.GetArg(1),
                noise1_chunk_rspecifier = po.GetArg(2),
                noise2_chunk_rspecifier = po.GetArg(3),
                perturbed_chunk_rspecifier = po.GetArg(4);

    SequentialBaseFloatMatrixReader source_chunk_reader(source_chunk_rspecifier);
    RandomAccessBaseFloatMatrixReader noise1_chunk_reader(noise1_chunk_rspecifier);
    RandomAccessBaseFloatMatrixReader noise2_chunk_reader(noise2_chunk_rspecifier);
    BaseFloatMatrixWriter perturbed_chunk_writer(perturbed_chunk_rspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !source_chunk_reader.Done(); source_chunk_reader.Next(), num_read++) {
      std::string key = source_chunk_reader.Key();
      // get source and 2 noise matrices.
      const Matrix<BaseFloat> &source_input = source_chunk_reader.Value();
      const Matrix<BaseFloat> &noise1_input = noise1_chunk_reader.Value(key);
      const Matrix<BaseFloat> &noise2_input = noise2_chunk_reader.Value(key);

      // the class FvectorPerturbBlock conduct the different perturb operation.
      FvectorPerturbBlock perturb_fvector_block(perturb_opts, source_input,
                                                noise1_input, noise2_input);
      Matrix<BaseFloat> perturbed_chunk;
      perturb_fvector_block.ApplyPerturbationBlock(&perturbed_chunk);
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
