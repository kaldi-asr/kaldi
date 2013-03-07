// nnetbin/nnet-mpe.cc

// Copyright 2011-2013  Karel Vesely;  Arnab Ghoshal

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include "nnet/nnet-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-nnet.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"


namespace kaldi {

void LatticeAcousticRescore(const Matrix<BaseFloat> &log_like,
                            const TransitionModel &trans_model,
                            const std::vector<int32> state_times,
                            Lattice *lat) {
  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  KALDI_ASSERT(!state_times.empty());
  std::vector<std::vector<int32> > time_to_state(log_like.NumRows());
  for (size_t i = 0; i < state_times.size(); i++) {
    KALDI_ASSERT(state_times[i] >= 0);
    if (state_times[i] < log_like.NumRows())  // end state may be past this..
      time_to_state[state_times[i]].push_back(i);
    else
      KALDI_ASSERT(state_times[i] == log_like.NumRows()
                   && "There appears to be lattice/feature mismatch.");
  }

  for (int32 t = 0; t < log_like.NumRows(); t++) {
    for (size_t i = 0; i < time_to_state[t].size(); i++) {
      int32 state = time_to_state[t][i];
      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
           aiter.Next()) {
        LatticeArc arc = aiter.Value();
        int32 trans_id = arc.ilabel;
        if (trans_id != 0) {  // Non-epsilon input label on arc
          int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
          arc.weight.SetValue2(-log_like(t, pdf_id) + arc.weight.Value2());
          aiter.SetValue(arc);
        }
      }
    }
  }
}

}  // namespace kaldi


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Perform iteration of Neural Network MPE/sMBR training by stochastic "
        "gradient descent.\n"
        "Usage:  nnet-mpe [options] <model-in> <transition-model-in> "
        "<feature-rspecifier> <den-lat-rspecifier> <ali-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-mpe nnet.init trans.mdl scp:train.scp scp:denlats.scp ark:train.ali "
        "nnet.iter1\n";

    ParseOptions po(usage);
    bool binary = false,
        crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate,
                "Perform cross-validation (don't backpropagate)");

    BaseFloat learn_rate = 0.00001,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    std::string feature_transform, class_frame_counts, silence_phones_str;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");
    po.Register("class-frame-counts", &class_frame_counts,
                "Class frame counts to compute the class priors");
    po.Register("silence-phones", &silence_phones_str, "Colon-separated list "
                "of integer id's of silence phones, e.g. 46:47");

    BaseFloat acoustic_scale = 1.0,
        lm_scale = 1.0,
        old_acoustic_scale = 0.0;
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    po.Register("old-acoustic-scale", &old_acoustic_scale,
                "Add in the scores in the input lattices with this scale, rather "
                "than discarding them.");

    bool act_as_llik = true,
        do_smbr = false;
    po.Register("act-as-llik", &act_as_llik, "Use activation as log-likelihood.");
    po.Register("do-smbr", &do_smbr, "Use state-level accuracies instead of "
                "phone accuracies.");

#if HAVE_CUDA == 1
    kaldi::int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID "
                "(-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#endif

    po.Read(argc, argv);

    if (po.NumArgs() != 6-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        transition_model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        den_lat_rspecifier = po.GetArg(4),
        ref_ali_rspecifier = po.GetArg(5);

    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(6);
    }

    std::vector<int32> silence_phones;
    if (!kaldi::SplitStringToIntegers(silence_phones_str, ":", false,
                                      &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    kaldi::SortAndUniq(&silence_phones);
    if (silence_phones.empty())
      KALDI_LOG << "No silence phones specified.";

    // Select the GPU
#if HAVE_CUDA == 1
    if (use_gpu_id > -2)
      CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    if (act_as_llik) {  // using activations directly: remove softmax, if present
      KALDI_LOG << "Using activations as log-likelihood.";
      if (nnet.Layer(nnet.LayerCount()-1)->GetType() == Component::kSoftmax) {
        KALDI_WARN << "Found softmax at output: removing it.";
        int32 num_layers = nnet.LayerCount();
        nnet.RemoveLayer(num_layers-1);
      }
    } else {  // add a softmax layer, if none present
      if (nnet.Layer(nnet.LayerCount()-1)->GetType() != Component::kSoftmax) {
        KALDI_WARN << "No softmax layer found at output when expecting one.";
        KALDI_LOG << "Adding a softmax layer.";
        int32 out_dim = nnet.Layer(nnet.LayerCount()-1)->OutputDim();
        Component *p_comp = new Softmax(out_dim, out_dim, &nnet);
        nnet.AppendLayer(p_comp);
      }
    }

    nnet.SetLearnRate(learn_rate, NULL);
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);

    TransitionModel trans_model;
    ReadKaldiObject(transition_model_filename, &trans_model);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessLatticeReader den_lat_reader(den_lat_rspecifier);
    RandomAccessInt32VectorReader ref_ali_reader(ref_ali_rspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, nnet_diff;
    Matrix<BaseFloat> nnet_out_h, nnet_loglik_h, nnet_diff_h;

    // Read the class-counts, compute priors
    Vector<BaseFloat> log_priors;  // was CuVector - now moved to CPU
    if (class_frame_counts != "") {
      Input in;
      in.OpenTextMode(class_frame_counts);
      log_priors.Read(in.Stream(), false);
      in.Close();

      // create inv. priors, or log inv priors
      BaseFloat sum = log_priors.Sum();
      log_priors.Scale(1.0 / sum);
      log_priors.ApplyLog();
    }


    Timer time;
    double time_next = 0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_ref_ali = 0, num_no_den_lat = 0,
        num_other_error = 0;

    double total_frame_acc = 0.0, lat_frame_acc;
    double total_mpe_obj = 0.0,
        mpe_obj;  // per-utterance objective function

    // do per-utterance processing
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!den_lat_reader.HasKey(utt)) {
        num_no_den_lat++;
        continue;
      }
      if (!ref_ali_reader.HasKey(utt)) {
        num_no_ref_ali++;
        continue;
      }

      // 1) get the features, numerator alignment
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const std::vector<int32> &ref_ali = ref_ali_reader.Value(utt);
      // check for temporal length of numerator alignments
      if (static_cast<MatrixIndexT>(ref_ali.size()) != mat.NumRows()) {
        KALDI_WARN << "Numerator alignment has wrong length "
                   << ref_ali.size() << " vs. "<< mat.NumRows();
        num_other_error++;
        continue;
      }

      // 2) get the denominator lattice, preprocess
      Lattice den_lat = den_lat_reader.Value(utt);
      if (old_acoustic_scale != 1.0) {
        fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale),
                          &den_lat);
      }
      // sort it topologically if not already so
      kaldi::uint64 props = den_lat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&den_lat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      }
      // get the lattice length and times of states
      vector<int32> state_times;
      int32 max_time = kaldi::LatticeStateTimes(den_lat, &state_times);
      // check for temporal length of denominator lattices
      if (max_time != mat.NumRows()) {
        KALDI_WARN << "Denominator lattice has wrong length " << max_time
                   << " vs. " << mat.NumRows();
        num_other_error++;
        continue;
      }

      // 3) propagate the feature to get the log-posteriors (nnet w/o sofrmax)
      // push features to GPU
      feats = mat;
      // possibly apply transform
      nnet_transf.Feedforward(feats, &feats_transf);
      // propagate through the nnet (assuming w/o softmax)
      nnet.Propagate(feats_transf, &nnet_out);
      // transfer it back to the host
      int32 num_frames = nnet_out.NumRows(),
          num_pdfs = nnet_out.NumCols();
      nnet_out_h.Resize(num_frames, num_pdfs, kUndefined);
      nnet_out.CopyToMat(&nnet_out_h);
      nnet_loglik_h.Resize(num_frames, num_pdfs, kUndefined);
      nnet_loglik_h.CopyFromMat(nnet_out_h);
      if (!act_as_llik) {  // with softmax, convert output to log-posteriors
        nnet_loglik_h.ApplyLog();
      }
      // subtract the log_priors
      if (log_priors.Dim() > 0) {
        nnet_loglik_h.AddVecToRows(-1.0, log_priors);
      }

      // 4) rescore the latice
      LatticeAcousticRescore(nnet_loglik_h, trans_model, state_times, &den_lat);
      if (acoustic_scale != 1.0 || lm_scale != 1.0)
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

      // 5) get the posteriors
      vector< std::map<int32, char> > arc_accs;
      arc_accs.resize(ref_ali.size());
      kaldi::Posterior post;

      if (do_smbr) {  // use state-level accuracies, i.e. sMBR estimation
        for (size_t i = 0; i < ref_ali.size(); i++) {
          int32 pdf = trans_model.TransitionIdToPdf(ref_ali[i]);
          arc_accs[i][pdf] = 1;
        }
        lat_frame_acc = LatticeForwardBackwardSmbr(den_lat, trans_model,
                                                   arc_accs, silence_phones,
                                                   &post);
      } else {  // use phone-level accuracies, i.e. regular MPE
        for (size_t i = 0; i < ref_ali.size(); i++) {
          int32 phone = trans_model.TransitionIdToPhone(ref_ali[i]);
          arc_accs[i][phone] = 1;
        }
        lat_frame_acc = kaldi::LatticeForwardBackwardMpe(den_lat, trans_model,
                                                         arc_accs, &post,
                                                         silence_phones);
      }

      // 6) convert the Posterior to a matrix
      nnet_diff_h.Resize(num_frames, num_pdfs, kSetZero);
      for (int32 t = 0; t < post.size(); t++) {
        for (int32 arc = 0; arc < post[t].size(); arc++) {
          int32 pdf = trans_model.TransitionIdToPdf(post[t][arc].first);
          nnet_diff_h(t, pdf) += post[t][arc].second;
          if (!act_as_llik)
            nnet_diff_h(t, pdf) *= (1 - nnet_out_h(t, pdf));
        }
      }

      // 7) Calculate the MPE-objective function
      mpe_obj = 0.0;
      for (int32 t = 0; t < nnet_diff_h.NumRows(); t++) {
        int32 pdf = trans_model.TransitionIdToPdf(ref_ali[t]);
        double posterior = nnet_diff_h(t, pdf);
        if (posterior < 1e-20)
          posterior = 1e-20;
        mpe_obj += log(posterior);
      }

      // report
      std::stringstream ss;
      ss << "Utt " << num_done + 1 << " : " << utt << " ("
          << feats_transf.NumRows() << "frm)" << " lat_frame_acc/frm: "
          << lat_frame_acc / feats_transf.NumRows() << "\t" << " mpe_obj/frm: "
          << mpe_obj / feats_transf.NumRows();
      KALDI_LOG << ss.str();
      // accumulate
      total_frame_acc += lat_frame_acc;
      total_mpe_obj += mpe_obj;

      // 8) subtract the pdf-Viterbi-path
      for (int32 t = 0; t < nnet_diff_h.NumRows(); t++) {
        int32 pdf = trans_model.TransitionIdToPdf(ref_ali[t]);
        /// Make sure the sum in vector is as close as possible
        /// to zero, reduce round-off errors:
        nnet_diff_h(t, pdf) += -nnet_diff_h.Row(t).Sum();
      }

      // 9) backpropagate through the nnet
      if (!crossvalidate) {
        nnet_diff = nnet_diff_h;
        nnet.Backpropagate(nnet_diff, NULL);
      }

      // increase time counter
      tot_t += feats_transf.NumRows();
      num_done++;
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED "
              << time.Elapsed() << "s, fps" << tot_t/time.Elapsed()
              << ", feature wait " << time_next << "s";

    KALDI_LOG << "Done " << num_done << " files, "
              << num_no_ref_ali << " with no reference alignments, "
              << num_no_den_lat << " with no lattices, "
              << num_other_error << " with other errors.";

    KALDI_LOG << "Overall MPE-objective/frame is "
              << (total_mpe_obj/tot_t) << " over " << tot_t << " frames. ";

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
