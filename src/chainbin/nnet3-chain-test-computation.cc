// chainbin/nnet3-chain-test-computation.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-chain-example.h"
#include "chain/chain-num-graph.h"
#include "chain/chain-numerator.h"
#include "chain/chain-cu-leakynum.h"
#include "chain/chain-training.h"
#include "chainbin/profiler2.h"
#include "nnet3/nnet-chain-diagnostics.h"
#include "chain/chain-denominator.h"

using namespace kaldi;
using namespace kaldi::nnet3;
using namespace fst;
using namespace kaldi::chain;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

void Compute(const NnetComputeProbOptions &nnet_config,
                       const chain::ChainTrainingOptions &chain_config,
                       const fst::StdVectorFst &den_fst,
                       const Nnet &nnet,
                       const NnetChainExample &chain_eg,
                       CuMatrix<BaseFloat>* nnet_output) {
  Profiler pf;
  CachingOptimizingCompiler compiler(nnet, nnet_config.optimize_config);
  Nnet *deriv_nnet;
  if (nnet_config.compute_deriv) {
    deriv_nnet = new Nnet(nnet);
    bool is_gradient = true;
    //SetZero(is_gradient, deriv_nnet);
  }

  bool need_model_derivative = nnet_config.compute_deriv,
       store_component_stats = false;
  ComputationRequest request;
  bool use_xent_regularization = (chain_config.xent_regularize != 0.0),
       use_xent_derivative = false;
  pf.tic("Get request");
  GetChainComputationRequest(nnet, chain_eg, need_model_derivative,
                             store_component_stats, use_xent_regularization,
                             use_xent_derivative, &request);
  pf.tic("compile");
  const NnetComputation *computation = compiler.Compile(request);
  pf.tic("computer");
  NnetComputer computer(nnet_config.compute_config, *computation,
                        nnet, deriv_nnet);
  pf.tic("accept inputs");
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet, chain_eg.inputs);
  pf.tic("forward");
  computer.Forward();

  pf.tic("rest");
  CuMatrix<BaseFloat> tmp(computer.GetOutput("output"));
  nnet_output->Resize(tmp.NumRows(), tmp.NumCols());
  nnet_output->CopyFromMat(tmp);
  pf.tac();
  std::cout << "Compute Profiling results:\n" << pf.toString() << "\n";
  
//  if (nnet_config.compute_deriv)
//    computer.Backward();
//  return nnet_output;
}


void SaveMatrixMatlab(const CuMatrixBase<BaseFloat> &cumat, std::string fname, std::string varname) {
  std::ofstream os(fname.c_str());
  Matrix<BaseFloat> temp(cumat.NumRows(), cumat.NumCols(), kUndefined);
  cumat.CopyToMat(&temp);
  if (temp.NumCols() == 0) {
    os << varname << " = [ ];\n";
  } else {
    os << varname << " = [";
    for (MatrixIndexT i = 0; i < temp.NumRows(); i++) {
      os << "\n  ";
      for (MatrixIndexT j = 0; j < temp.NumCols(); j++)
        os << temp(i, j) << " ";
    }
    os << "];\n";
  }
}


void SaveMatrixSparselyMatlab(const CuMatrixBase<BaseFloat> &cumat, std::string fname, std::string varname) {
  std::ofstream os(fname.c_str());
  Matrix<BaseFloat> temp(cumat.NumRows(), cumat.NumCols(), kUndefined);
  cumat.CopyToMat(&temp);
  std::vector<int32> row_idxs, col_idxs;
  std::vector<BaseFloat> vals;

  for (MatrixIndexT i = 0; i < temp.NumRows(); i++) {
    for (MatrixIndexT j = 0; j < temp.NumCols(); j++) {
      if (temp(i, j) != 0.0) {
        row_idxs.push_back(i + 1);
        col_idxs.push_back(j + 1);
        vals.push_back(temp(i, j));
      }
    }
  }
  if (temp(temp.NumRows() - 1, temp.NumCols() - 1) == 0) {
    row_idxs.push_back(temp.NumRows());
    col_idxs.push_back(temp.NumCols());
    vals.push_back(0.0);
  }

  os << "sp_i = [ "; for (int i = 0; i < row_idxs.size(); i++) os << row_idxs[i] << " "; os << "];\n";
  os << "sp_j = [ "; for (int i = 0; i < col_idxs.size(); i++) os << col_idxs[i] << " "; os << "];\n";
  os << "sp_v = [ "; for (int i = 0; i < vals.size(); i++) os << vals[i] << " "; os << "];\n";
  os << varname << " = sparse(sp_i, sp_j, sp_v);\n";
  os << "# num-non-zero-elems: " << row_idxs.size() << "\n";
}

void SaveVectorSparselyMatlab(const Vector<BaseFloat> vec, std::string fname, std::string varname) {
  std::ofstream os(fname.c_str());
  os << varname << " = [";
  for (MatrixIndexT i = 0; i < vec.Dim(); i++) {
    os << vec(i) << " ";
  }
  os << "];\n";
}

std::vector<int32> SelectAPath(const fst::StdVectorFst &fst, BaseFloat *pathlogprob) {
  std::vector<int32> path;
  int32 state = 0;
  *pathlogprob = 0;
  while (true) {
    bool any_arc = false;
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      path.push_back(arc.ilabel);
      state = arc.nextstate;
      *pathlogprob += -arc.weight.Value();
      any_arc = true;
      break;
    }
    if (!any_arc)
      break;
  }
  KALDI_LOG << "path-length ====== " << path.size();
  return path;
}

BaseFloat FindPath(const fst::StdVectorFst &fst, const std::vector<int32>& path) {
  int32 num_states = fst.NumStates();
  BaseFloat pathlogprob = 0;
  for (int32 startstate = 0; startstate < num_states; startstate++) {
    int32 state = startstate;
    pathlogprob = 0;
    bool pathfullytraversed = true;
    for (int32 i = 0; i < path.size(); i++) {
      int32 pdfid = path[i];
      std::cout << pdfid << " --> ";
      bool found = false;
      for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state);
           !aiter.Done(); aiter.Next()) {
        const fst::StdArc &arc = aiter.Value();
        if (arc.ilabel == pdfid) {
          found = true;
          state = arc.nextstate;
          pathlogprob += -arc.weight.Value();
          if (i == path.size() - 1)
            pathlogprob += -fst.Final(arc.nextstate).Value();
          break;
        }
      }
      if (!found) {
        pathfullytraversed = false;
        break;
      }
    }
    std::cout << "\n";
    if (pathfullytraversed)
      break;
  }
  return pathlogprob;
}

void FixFinals(fst::StdVectorFst *fst) {
  int32 num_states = fst->NumStates(), finalstate = 0;
  for (int32 state = 0; state < num_states; state++)
    if (fst->Final(state) != fst::TropicalWeight::Zero()) {
      finalstate = state;
      break;
    }

  for (int32 state = 0; state < num_states; state++) {
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      if (arc.nextstate > finalstate) {
        KALDI_ASSERT(fst->Final(arc.nextstate) == fst::TropicalWeight::One());
        fst::StdArc arc2(arc);
        arc2.nextstate = finalstate;
        aiter.SetValue(arc2);
      }
    }
  }
  std::vector<int32> del; 
  for (int32 state = finalstate + 1; state < num_states; state++) {
    del.push_back(state);
  }
  fst->DeleteStates(del);
}


void NormalizeFst(fst::StdVectorFst *fst) {
  int32 num_states = fst->NumStates();
  for (int32 state = 0; state < num_states - 1; state++) {
    double outgoing_prob_sum = 0.0;
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      outgoing_prob_sum += Exp(-arc.weight.Value());
    }
//    KALDI_LOG << "sum for state " << state << " is " << outgoing_prob_sum;
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
//      KALDI_LOG << "for state " << state << " weight before: " << arc.weight.Value();
      fst::StdArc arc2(arc);
      arc2.weight = StdArc::Weight(arc.weight.Value() + Log(outgoing_prob_sum));
//      KALDI_LOG << "for state " << state << " weight after: " << arc2.weight.Value();
      aiter.SetValue(arc2);
    }
  }
}

int main(int argc, char *argv[]) {
  try {

{
    StdVectorFst num3;
    ReadFstKaldi("num3.fst", &num3);
    StdVectorFst den;
    ReadFstKaldi("den.fst", &den);
    Supervision sup;
    sup.frames_per_sequence = 3;
    sup.label_dim = 10;
    sup.fst = num3;
    sup.fsts.push_back(num3);
    NumeratorGraph numg(sup, false);
    //num3g.PrintInfo(true);
    CuMatrix<BaseFloat>
            deriv1(3, 10, kSetZero),
            deriv2(3, 10, kSetZero);
    std::cout << "fst:\n";
//    sup.Write(std::cout, false); std::cout << "\n";
    
    DenominatorGraph deng(den, sup.label_dim);
    CuMatrix<BaseFloat> obs_mat(sup.frames_per_sequence, sup.label_dim);
    for (int t = 0; t < obs_mat.NumRows(); t++)
      for (int j = 0; j < obs_mat.NumCols(); j++) {
        int pdfid = j + 1;
        obs_mat(t, j) = Log((float)((t+1)*(pdfid+1) % 4 + 1));
      }
//    obs_mat.Write(std::cout, false);
    NumeratorComputation numc(sup, obs_mat);
    BaseFloat cpu_num_logprob = numc.Forward();
    numc.Backward(&deriv1);
    std::cout << "cpu num log prob: " << cpu_num_logprob << "\n";
//    deriv1.Write(std::cout, false);
{
    NormalizeFst(&sup.fst);
    std::cout << "Normed fst:\n";
//    sup.Write(std::cout, false);  std::cout << "\n";
    NumeratorComputation numc(sup, obs_mat);
    BaseFloat cpu_num_logprob = numc.Forward();
    std::cout << "cpu num log prob normed: " << cpu_num_logprob << "\n";
    numc.Backward(&deriv2);
//    deriv2.Write(std::cout, false);
}
    ChainTrainingOptions copts;
    copts.leakynum_leak_prob = 0.05;
    copts.leakynum_use_priors = 0.0;
    CuLeakyNumeratorComputation leakyc(copts, numg, deng, obs_mat);
    BaseFloat leaky_num_logprob = leakyc.Forward();
    std::cout << "leaky num log prob: " << leaky_num_logprob << "\n";
}

//    return 0;

/* The result should be
cpu num log prob: -2.16679
leaky num log prob: -0.779313

This has been calculated and verified by 
(1) Actually creating the full leaked num graph (including all leak and unleak transitions) and doing FB on it
(2) Actually creating the full leaked num graph (as above) and enumerating all the paths from start to end
    and summing the likelihoods of these paths!!!
*/

    const char *usage =
        "Usage:  nnet3-chain-test-computation [options] <den-fst> <egs-rspecifier>\n";

//    bool compress = false;
//    int32 minibatch_size = 64;
    std::string use_gpu = "no", nnet_rxfilename="";
    ChainTrainingOptions opts;
    int32 numegs = 1;
    bool save = false;

    ParseOptions po(usage);
//    po.Register("minibatch-size", &minibatch_size, "Target size of minibatches "
//                "when merging (see also --measure-output-frames)");
//    po.Register("compress", &compress, "If true, compress the output examples "
//                "(not recommended unless you are writing to disk");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("num-egs", &numegs, "");
    po.Register("nnet", &nnet_rxfilename,
                "'nnet3-am-copy --raw=true <model> - |'");
    po.Register("save", &save, "");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string examples_rspecifier = po.GetArg(2),
        den_fst_rxfilename = po.GetArg(1);

    Nnet nnet;
    if (nnet_rxfilename != "") {
      ReadKaldiObject(nnet_rxfilename, &nnet);
    } else {
      KALDI_LOG << "WAAAAAAAAAAAAAARNING: nnet not provided!";
    }

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);

    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_rxfilename, &den_fst);
    
    DenominatorGraph den_graph(den_fst, nnet.OutputDim("output"));
    KALDI_LOG << "initital probs sum: " << den_graph.InitialProbs().Sum();
    KALDI_LOG << "initital probs dim: " << den_graph.InitialProbs().Dim();
    //return 0;

    Vector<BaseFloat> diff_per_time_base(50, kSetZero); /// Assuming framespereg = 50
    Vector<BaseFloat> diff_per_time_leaky(50, kSetZero); /// Assuming framespereg = 50
    Vector<BaseFloat> diff_per_time_rel(50, kSetZero); /// Assuming framespereg = 50
    
    BaseFloat base_avg_logprob = 0.0, leaky_avg_logprob = 0.0, objf_sum = 0.0;
    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      NnetChainExample eg(example_reader.Value());
      std::string key = example_reader.Key();
      //if (num_read < 1)
      //  continue;
      if (num_read >= numegs)
        break;
      std::cout << "Example " << key << ":\n"
                //<< "n-inputs: " << eg.inputs.size() << "\n"
                //<< "n-outputs: " << eg.outputs.size() << "\n"
                //<< "input[0].name: " << eg.inputs[0].name << "\n"
                //<< "input[0].feats-size: " << eg.inputs[0].features.NumRows() << ", " << eg.inputs[0].features.NumCols() << "\n"
                //<< "out[0].name: " << eg.outputs[0].name << "\n"
                //<< "out[0].supervision.num_sequences: " << eg.outputs[0].supervision.num_sequences << "\n"
                //<< "out[0].supervision.frames_per_sequence: " << eg.outputs[0].supervision.frames_per_sequence << "\n"
                //<< "out[0].supervision.label_dim: " << eg.outputs[0].supervision.label_dim << "\n"
                << "out[0].supervision.fst.NumStates: " << eg.outputs[0].supervision.fst.NumStates() << "\n"
                //<< "out[0].supervision.fsts.size: " << eg.outputs[0].supervision.fsts.size() << "\n"                
                ;
      Profiler pf;
      //for (int i = 0; i < eg.outputs[0].supervision.fsts.size(); i++)
      //  std::cout << "fsts[i].NumStates: " << eg.outputs[0].supervision.fsts[i].NumStates() << "\n";


      //for (int32 i = 0; i < eg.outputs[0].supervision.fsts.size(); i++)
      //  NormalizeFst(&eg.outputs[0].supervision.fsts[i]);
      //NormalizeFst(&eg.outputs[0].supervision.fst);
//      std::ofstream os("eg-unnormed.txt");
//      eg.Write(os, false);

      pf.tic("normalizing");
      for (int32 i = 0; i < eg.outputs[0].supervision.fsts.size(); i++)
        PushInLog<REWEIGHT_TO_INITIAL>(&eg.outputs[0].supervision.fsts[i], kPushLabels|kPushWeights);
      PushInLog<REWEIGHT_TO_INITIAL>(&eg.outputs[0].supervision.fst, kPushLabels|kPushWeights);
      //std::ofstream os("eg-pushed.txt");
      //eg.Write(os, false);
      pf.tac();
      /*BaseFloat numlogprob;
      std::vector<int32> path = SelectAPath(eg.outputs[0].supervision.fst, &numlogprob);
      std::cout << "the path is: ";
      for (int32 i = 0; i < path.size(); i++)
        std::cout << path[i] << " -- ";
      std::cout << "num logprob = " << numlogprob << "\n";
      BaseFloat denlogprob = FindPath(den_fst, path);
      std::cout << "den logprob = " << denlogprob << "\n";
      return 0;*/
      


      
      
      
      BaseFloat weight = eg.outputs[0].supervision.weight * eg.outputs[0].supervision.num_sequences *
      eg.outputs[0].supervision.frames_per_sequence;

      
//KALDI_LOG << "First trans: " << den_graph.Transitions()[0].transition_prob;
//DenominatorGraph dg_copy(den_graph);
//pf.tic("iterateOverAllDenTransitions");
  //den_graph.ScaleTransitions(0.9);
  //DenominatorGraph dg_copy(den_graph);
//pf.tac();
//KALDI_LOG << "First trans modified: " << dg_copy.Transitions()[0].transition_prob;
//KALDI_LOG << "First trans again: " << den_graph.Transitions()[0].transition_prob;
//std::cout << "Profiling results:\n" << pf.toString() << "\n";
//break;
      
      pf.tic("matPrep");
      int32 T = eg.outputs[0].supervision.frames_per_sequence,
            B = eg.outputs[0].supervision.num_sequences,
            N = eg.outputs[0].supervision.label_dim; //num pdfs
      CuMatrix<BaseFloat> //random_nnet_output(T*B, N),
                          nnet_output_deriv1(T*B, N, kSetZero),
                          nnet_output_deriv2(T*B, N, kSetZero),
                          den_derivs(T*B, N, kSetZero);
      /*random_nnet_output.SetRandUniform();
      random_nnet_output.ApplyLogSoftMaxPerRow(random_nnet_output);*/
      NnetComputeProbOptions nnet_config;
      nnet_config.optimize_config.optimize = false;  // setting this false disallow all optimization.
      nnet_config.optimize_config.consolidate_model_update = false;
      nnet_config.optimize_config.propagate_in_place = false;
      nnet_config.optimize_config.backprop_in_place = false;
      nnet_config.optimize_config.convert_addition = false;
      nnet_config.optimize_config.remove_assignments = false;
      nnet_config.optimize_config.allow_left_merge = false;
      nnet_config.optimize_config.allow_right_merge = false;
      nnet_config.optimize_config.initialize_undefined = false;
      nnet_config.optimize_config.move_sizing_commands = false;
      nnet_config.optimize_config.allocate_from_other = false;
      nnet_config.compute_deriv = false;
      CuMatrix<BaseFloat> random_nnet_output;// = 
      Compute(nnet_config, opts, den_fst, nnet, eg, &random_nnet_output);
      pf.tac();




      /*BaseFloat tot_objf = 0.0, tot_l2_term = 0.0, tot_weight = 0.0;
      ComputeChainObjfAndDeriv(opts, den_graph,
                             eg.outputs[0].supervision, random_nnet_output,
                             &tot_objf, &tot_l2_term, &tot_weight,
                             &nnet_output_deriv1,
                             NULL);
      std::cout << "#############################\n";
      std::cout << "#############################\n";
      std::cout << "tot_objf:" << tot_objf << ", tot_l2_term: " << tot_l2_term << ", tot_weight: " << tot_weight << "\n";
      std::cout << "#############################\n";
      std::cout << "#############################\n";
      objf_sum += tot_objf/tot_weight;
      continue;*/

      
      // 
      pf.tic("on-CPU");
      NumeratorComputation numerator(eg.outputs[0].supervision, random_nnet_output);
      BaseFloat num_logprob_weighted = numerator.Forward();
      std::cout << "original num logprob weighted: " << num_logprob_weighted << "\n";
      std::cout << "original num logprob per frame: " << num_logprob_weighted/weight << "\n";
      base_avg_logprob += num_logprob_weighted/weight;
      numerator.Backward(&nnet_output_deriv1);
      pf.tac();
      // 

      pf.tic("numGraph");
      NumeratorGraph ng(eg.outputs[0].supervision, opts.leakynum_scale_first_transitions);
      //ng.PrintInfo(true);
      pf.tac();

      /*
      pf.tic("on-GPU-my");
      ChainTrainingOptions opts;
      CuNumeratorComputation cunum(opts, ng, random_nnet_output);
      BaseFloat cu_num_logprob_weighted = cunum.Forward();
      std::cout << "cu num logprob weighted: " << cu_num_logprob_weighted << "\n";
      bool ok = true;
      ok = cunum.Backward(eg.outputs[0].supervision.weight, &nnet_output_deriv2);
      std::cout << "ok: " << ok << "\n";
      pf.tac();
      */


      pf.tic("numleaky");
      CuLeakyNumeratorComputation culeakynum(opts, ng, den_graph, random_nnet_output);
      BaseFloat cu_leakynum_logprob_weighted = culeakynum.Forward();
      std::cout << "cu leaky num logprob weighted: " << cu_leakynum_logprob_weighted << "\n";
      std::cout << "cu leaky num logprob per frame: " << cu_leakynum_logprob_weighted/weight << "\n";
      leaky_avg_logprob += cu_leakynum_logprob_weighted/weight;
      bool ok = true;
      ok = culeakynum.Backward(&nnet_output_deriv2);
      std::cout << "ok: " << ok << "\n";
      pf.tac();


      if (save) {
        SaveMatrixSparselyMatlab(nnet_output_deriv1, "f_occup_base.m", "occup_base");
        SaveMatrixSparselyMatlab(nnet_output_deriv2, "f_occup_leaky.m", "occup_leaky");
      }

      pf.tic("Den");
      DenominatorComputation denominator(opts, den_graph,
                                    eg.outputs[0].supervision.num_sequences,
                                    random_nnet_output);
      BaseFloat den_logprob = denominator.Forward();
      denominator.Backward(1.0,
                              &den_derivs);
      nnet_output_deriv1.AddMat(-eg.outputs[0].supervision.weight, den_derivs);
      nnet_output_deriv2.AddMat(-eg.outputs[0].supervision.weight, den_derivs);

      std::cout << "denlogprob: " << den_logprob << "\n";
      std::cout << "cu leaky logprob: " << cu_leakynum_logprob_weighted - eg.outputs[0].supervision.weight * den_logprob << "\n";
      std::cout << "logprob: " << num_logprob_weighted - eg.outputs[0].supervision.weight * den_logprob << "\n";
      std::cout << "fixed cu leaky logprob: " << cu_leakynum_logprob_weighted - 2 * eg.outputs[0].supervision.weight * den_logprob << "\n";
      
      if (save) {
        std::cout << "net_output_deriv1. sum: " << nnet_output_deriv1.Sum() << ", fnorm: " << nnet_output_deriv1.FrobeniusNorm() << "\n";
        std::cout << "net_output_deriv2. sum: " << nnet_output_deriv2.Sum() << ", fnorm: " << nnet_output_deriv2.FrobeniusNorm() << "\n";      
        
        SaveMatrixSparselyMatlab(den_derivs, "f_occup_den.m", "occup_den");
        SaveMatrixSparselyMatlab(nnet_output_deriv1, "f_deriv_base.m", "deriv_base");
        SaveMatrixSparselyMatlab(nnet_output_deriv2, "f_deriv_leaky.m", "deriv_leaky");
      }
      
      std::cout << "Profiling results:\n" << pf.toString() << "\n";

      //WriteKaldiObject(nnet_output_deriv1, "deriv1.txt", false);
      //WriteKaldiObject(nnet_output_deriv2, "deriv2.txt", false);

      for (int32 time = 0; time < T; time++) {
        CuSubMatrix<BaseFloat> this_time_deriv1(nnet_output_deriv1, time*B, B, 0, N);
        diff_per_time_base(time) += this_time_deriv1.Sum();
        CuSubMatrix<BaseFloat> this_time_deriv2(nnet_output_deriv2, time*B, B, 0, N);
        diff_per_time_leaky(time) += this_time_deriv2.Sum();
      }
      nnet_output_deriv2.AddMat(-1.0, nnet_output_deriv1);
      for (int32 time = 0; time < T; time++) {
        CuSubMatrix<BaseFloat> this_time_deriv2(nnet_output_deriv2, time*B, B, 0, N);
        diff_per_time_rel(time) += this_time_deriv2.Sum();
      }
      
      
    }
    //std::cout << "avg leaky objf per frame: " << objf_sum/numegs << "\n";
    //return 0;
    
    std::cout << "cu leaky num logprob per frame: " << leaky_avg_logprob/numegs << "\n";
    std::cout << "base logprob per frame: " << base_avg_logprob/numegs << "\n";
    std::cout << "net_output_deriv_base. sum: " << diff_per_time_base.Sum() << "\n";
    std::cout << "net_output_deriv_leaky. sum: " << diff_per_time_leaky.Sum() << "\n";      
    if (save) {
      SaveVectorSparselyMatlab(diff_per_time_base, "fdiff_base.m", "diff_base");
      SaveVectorSparselyMatlab(diff_per_time_leaky, "fdiff_leaky.m", "diff_leaky");
      SaveVectorSparselyMatlab(diff_per_time_rel, "fdiff_rel.m", "diff_rel");
    }

//#if HAVE_CUDA==1
    //CuDevice::Instantiate().PrintProfile();
//#endif
    
    KALDI_LOG << "Checked " << num_read << " egs.";
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


