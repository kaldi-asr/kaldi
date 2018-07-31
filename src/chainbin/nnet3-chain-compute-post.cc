// nnet3bin/nnet3-chain-compute-post.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2015   Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
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
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "chain/chain-denominator.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Compute posteriors from 'denominator FST' of chain model and optionally "
        "map them to phones.\n"
        "\n"
        "Usage: nnet3-chain-compute-post [options] <nnet-in> <den-fst> <features-rspecifier> <matrix-wspecifier>\n"
        " e.g.: nnet3-chain-compute-post --transform-mat=transform.mat final.raw den.fst scp:feats.scp ark:nnet_prediction.ark\n"
        "See also: nnet3-compute\n"
        "See steps/nnet3/chain/get_phone_post.sh for example of usage.\n"
        "Note: this program makes *extremely inefficient* use of the GPU.\n"
        "You are advised to run this on CPU until it's improved.\n";

    ParseOptions po(usage);
    Timer timer;

    BaseFloat leaky_hmm_coefficient = 0.1;
    NnetSimpleComputationOptions opts;
    opts.acoustic_scale = 1.0; // by default do no acoustic scaling.

    std::string use_gpu = "yes";

    std::string transform_mat_rxfilename;
    std::string ivector_rspecifier,
                online_ivector_rspecifier,
                utt2spk_rspecifier;
    int32 online_ivector_period = 0;
    opts.Register(&po);

    po.Register("ivectors", &ivector_rspecifier, "Rspecifier for "
                "iVectors as vectors (i.e. not estimated online); per utterance "
                "by default, or per speaker if you provide the --utt2spk option.");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for "
                "utt2spk option used to get ivectors per speaker");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier for "
                "iVectors estimated online, as matrices.  If you supply this,"
                " you must set the --online-ivector-period option.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of frames "
                "between iVectors in matrices supplied to the --online-ivectors "
                "option");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("leaky-hmm-coefficient", &leaky_hmm_coefficient, "'Leaky HMM' "
                "coefficient: smaller values will tend to lead to more "
                "confident posteriors.  0.1 is what we normally use in "
                "training.");
    po.Register("transform-mat", &transform_mat_rxfilename, "Location to read "
                "the matrix to transform posteriors to phones.  Matrix is "
                "of dimension num-phones by num-pdfs.");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        den_fst_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        matrix_wspecifier = po.GetArg(4);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped ivector_reader(
        ivector_rspecifier, utt2spk_rspecifier);

    CachingOptimizingCompiler compiler(nnet, opts.optimize_config);

    chain::ChainTrainingOptions chain_opts;
    // the only option that actually gets used here is
    // opts_.leaky_hmm_coefficient, and that's the only one we expose on the
    // command line.
    chain_opts.leaky_hmm_coefficient = leaky_hmm_coefficient;

    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_rxfilename, &den_fst);
    int32 num_pdfs = nnet.OutputDim("output");
    if (num_pdfs < 0) {
      KALDI_ERR << "Neural net '" << nnet_rxfilename
                << "' has no output named 'output'";
    }
    chain::DenominatorGraph den_graph(den_fst, num_pdfs);


    CuSparseMatrix<BaseFloat> transform_sparse_mat;
    if (!transform_mat_rxfilename.empty()) {
      Matrix<BaseFloat> transform_mat;
      ReadKaldiObject(transform_mat_rxfilename, &transform_mat);
      if (transform_mat.NumCols() != num_pdfs)
        KALDI_ERR << "transform-mat from " << transform_mat_rxfilename
                  << " has " << transform_mat.NumCols() << " cols, expected "
                  << num_pdfs;
      SparseMatrix<BaseFloat> temp_sparse_mat(transform_mat);
      // the following is just a shallow swap if we're on CPU.  This program
      // actually won't actually work very fast on GPU, but doing it this way
      // will make it easier to modify it later if we really want efficient
      // operation on GPU.
      transform_sparse_mat.Swap(&temp_sparse_mat);
    }

    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);

    int32 num_success = 0, num_fail = 0;
    int64 tot_input_frames = 0, tot_output_frames = 0;
    double tot_forward_prob = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &features (feature_reader.Value());
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }
      const Matrix<BaseFloat> *online_ivectors = NULL;
      const Vector<BaseFloat> *ivector = NULL;
      if (!ivector_rspecifier.empty()) {
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector available for utterance " << utt;
          num_fail++;
          continue;
        } else {
          ivector = &ivector_reader.Value(utt);
        }
      }
      if (!online_ivector_rspecifier.empty()) {
        if (!online_ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No online iVector available for utterance " << utt;
          num_fail++;
          continue;
        } else {
          online_ivectors = &online_ivector_reader.Value(utt);
        }
      }

      Vector<BaseFloat> priors;  // empty vector, we don't need priors here.
      DecodableNnetSimple nnet_computer(
          opts, nnet, priors,
          features, &compiler,
          ivector, online_ivectors,
          online_ivector_period);

      Matrix<BaseFloat> matrix(nnet_computer.NumFrames(),
                               nnet_computer.OutputDim());
      for (int32 t = 0; t < nnet_computer.NumFrames(); t++) {
        SubVector<BaseFloat> row(matrix, t);
        nnet_computer.GetOutputForFrame(t, &row);
      }

      // Of course it makes no sense to copy to GPU and then back again.
      // But anyway this program woudn't work very well if we actually ran
      // with --use-gpu=yes.  In the CPU case the following is just a shallow
      // swap.
      CuMatrix<BaseFloat> gpu_nnet_output;
      gpu_nnet_output.Swap(&matrix);


      chain::DenominatorComputation den_computation(
          chain_opts, den_graph,
          1, // num_sequences,
          gpu_nnet_output);


      int32 num_frames = gpu_nnet_output.NumRows();
      BaseFloat forward_prob = den_computation.Forward();

      CuMatrix<BaseFloat> posteriors(num_frames, num_pdfs);
      BaseFloat scale = 1.0;
      bool ok = den_computation.Backward(scale, &posteriors);

      KALDI_VLOG(1) << "For utterance " << utt << ", log-prob per frame was "
                    << (forward_prob / num_frames) << " over "
                    << num_frames << " frames.";

      if (!ok || !(forward_prob - forward_prob == 0)) {  // if or NaN
        KALDI_WARN << "Something went wrong for utterance " << utt
                   << "; forward-prob = " << forward_prob
                   << ", num-frames = " << num_frames;
        num_fail++;
        continue;
      }

      num_success++;
      tot_input_frames += features.NumRows();
      tot_output_frames += num_frames;
      tot_forward_prob += forward_prob;

      // Write out the posteriors.
      if (transform_mat_rxfilename.empty()) {
        // write out posteriors over pdfs.
        Matrix<BaseFloat> posteriors_cpu;
        posteriors.Swap(&posteriors_cpu);
        matrix_writer.Write(utt, posteriors_cpu);
      } else {
        // write out posteriors over (most likely) phones.
        int32 num_phones = transform_sparse_mat.NumRows();
        CuMatrix<BaseFloat> phone_post(num_frames, num_phones);
        phone_post.AddMatSmat(1.0, posteriors,
                              transform_sparse_mat, kTrans, 0.0);
        Matrix<BaseFloat> phone_post_cpu;
        phone_post.Swap(&phone_post_cpu);
        // write out posteriors over phones.
        matrix_writer.Write(utt, phone_post_cpu);

        if (GetVerboseLevel() >= 1 || RandInt(0,99)==0) {
          BaseFloat sum = posteriors.Sum();
          if (((sum / num_frames) - 1.0) > 0.01) {
            KALDI_WARN << "Expected sum of posteriors " << sum
                       << " to be close to num-frames " << num_frames;
          }
        }
      }
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 input frames/sec is "
              << (elapsed*100.0/tot_input_frames);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    KALDI_LOG << "Overall log-prob per (output) frame was "
              << (tot_forward_prob / tot_output_frames)
              << " over " << tot_output_frames << " frames.";

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
