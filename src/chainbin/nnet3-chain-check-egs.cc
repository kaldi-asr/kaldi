// chainbin/nnet3-chain-check-egs.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-chain-example.h"
#include "chain/chain-num-graph.h"
#include "chain/chain-numerator.h"
#include "chain/chain-cu-numerator.h"
#include "chain/chain-training.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace fst;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Usage:  nnet3-chain-check-egs [options] <egs-rspecifier>\n";

//    bool compress = false;
//    int32 minibatch_size = 64;
    std::string use_gpu = "no";

    ParseOptions po(usage);
//    po.Register("minibatch-size", &minibatch_size, "Target size of minibatches "
//                "when merging (see also --measure-output-frames)");
//    po.Register("compress", &compress, "If true, compress the output examples "
//                "(not recommended unless you are writing to disk");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      const NnetChainExample &eg = example_reader.Value();
      std::string key = example_reader.Key();
      if (num_read > 0)
        break;
      std::cout << "Example " << key << ":\n\n"
                << "n-inputs: " << eg.inputs.size() << "\n"
                << "n-outputs: " << eg.outputs.size() << "\n"
                << "input[0].name: " << eg.inputs[0].name << "\n"
                << "input[0].feats-size: " << eg.inputs[0].features.NumRows() << ", " << eg.inputs[0].features.NumCols() << "\n"
                << "out[0].name: " << eg.outputs[0].name << "\n"
                << "out[0].supervision.num_sequences: " << eg.outputs[0].supervision.num_sequences << "\n"
                << "out[0].supervision.frames_per_sequence: " << eg.outputs[0].supervision.frames_per_sequence << "\n"
                << "out[0].supervision.label_dim: " << eg.outputs[0].supervision.label_dim << "\n"
                << "out[0].supervision.fst.NumStates: " << eg.outputs[0].supervision.fst.NumStates() << "\n"
                << "out[0].supervision.fsts.size: " << eg.outputs[0].supervision.fsts.size() << "\n"                
                ;
      for (int i = 0; i < eg.outputs[0].supervision.fsts.size(); i++)
        std::cout << "fsts[i].NumStates: " << eg.outputs[0].supervision.fsts[i].NumStates() << "\n";
      NumeratorGraph ng(eg.outputs[0].supervision);
      //ng.PrintInfo();
      int32 T = eg.outputs[0].supervision.frames_per_sequence,
            B = eg.outputs[0].supervision.num_sequences,
            N = eg.outputs[0].supervision.label_dim; //num pdfs
      CuMatrix<BaseFloat> random_nnet_output(T*B, N),
                          nnet_output_deriv1(T*B, N),
                          nnet_output_deriv2(T*B, N);
      random_nnet_output.SetRandn();
      random_nnet_output.ApplyLogSoftMaxPerRow(random_nnet_output);
      NumeratorComputation numerator(eg.outputs[0].supervision, random_nnet_output);
      BaseFloat num_logprob_weighted = numerator.Forward();
      std::cout << "num logprob weighted: " << num_logprob_weighted << "\n";
      numerator.Backward(&nnet_output_deriv1);
      //nnet_output_deriv.Write(std::cout, false);
      
      ChainTrainingOptions opts;
      CuNumeratorComputation cunum(opts, ng, random_nnet_output);
      BaseFloat cu_num_logprob_weighted = cunum.Forward();
      std::cout << "cu num logprob weighted: " << cu_num_logprob_weighted << "\n";
      bool ok = true;
      ok = cunum.Backward(eg.outputs[0].supervision.weight, &nnet_output_deriv2);
      std::cout << "ok: " << ok << "\n";
      
      AssertEqual(nnet_output_deriv1, nnet_output_deriv2, 0.001);
      //nnet_output_deriv.Write(std::cout, false);
    }
    
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    
    KALDI_LOG << "Checked " << num_read << " egs.";
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


