#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "rnnlm/rnnlm-diagnostics.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints to in logging messages the average log-prob per frame of\n"
        "the given data with an nnet3 neural net.  The input of this is the output of\n"
        "e.g. nnet3-get-egs | nnet3-merge-egs.\n"
        "\n"
        "Usage:  nnet3-compute-prob [options] <raw-model-in> <training-examples-in>\n"
        "e.g.: nnet3-compute-prob 0.raw ark:valid.egs\n";

    
    // This program doesn't support using a GPU, because these probabilities are
    // used for diagnostics, and you can just compute them with a small enough
    // amount of data that a CPU can do it within reasonable time.

    LmNnetComputeProbOptions opts;
    
    ParseOptions po(usage);

    opts.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string raw_nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2);

    LmNnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);

    LmNnetComputeProb prob_computer(opts, nnet);
    
    SequentialNnetExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next()) {
      prob_computer.Compute(example_reader.Value());
    }

    bool ok = prob_computer.PrintTotalStats();
    
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


