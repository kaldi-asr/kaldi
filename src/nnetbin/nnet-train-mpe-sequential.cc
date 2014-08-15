// nnetbin/nnet-train-mpe-sequential.cc

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely);  Arnab Ghoshal

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-pdf-prior.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"


namespace kaldi {
namespace nnet1 {

void LatticeAcousticRescore(const Matrix<BaseFloat> &log_like,
                            const TransitionModel &trans_model,
                            const std::vector<int32> &state_times,
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

}  // namespace nnet1
}  // namespace kaldi


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Perform iteration of Neural Network MPE/sMBR training by stochastic "
        "gradient descent.\n"
        "The network weights are updated on each utterance.\n"
        "Usage:  nnet-train-mpe-sequential [options] <model-in> <transition-model-in> "
        "<feature-rspecifier> <den-lat-rspecifier> <ali-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-mpe-sequential nnet.init trans.mdl scp:train.scp scp:denlats.scp ark:train.ali "
        "nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts; trn_opts.learn_rate=0.00001;
    trn_opts.Register(&po);

    bool binary = true; 
    po.Register("binary", &binary, "Write output in binary mode");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, 
                "Feature transform in Nnet format");
    std::string silence_phones_str;
    po.Register("silence-phones", &silence_phones_str, "Colon-separated list "
                "of integer id's of silence phones, e.g. 46:47");

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

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
    kaldi::int32 max_frames = 6000; // Allow segments maximum of one minute by default
    po.Register("max-frames",&max_frames, "Maximum number of frames a segment can have to be processed");
    bool do_smbr = false;
    po.Register("do-smbr", &do_smbr, "Use state-level accuracies instead of "
                "phone accuracies.");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
     
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        transition_model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        den_lat_rspecifier = po.GetArg(4),
        ref_ali_rspecifier = po.GetArg(5);

    std::string target_model_filename;
    target_model_filename = po.GetArg(6);

    std::vector<int32> silence_phones;
    if (!kaldi::SplitStringToIntegers(silence_phones_str, ":", false,
                                      &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    kaldi::SortAndUniq(&silence_phones);
    if (silence_phones.empty())
      KALDI_LOG << "No silence phones specified.";

    // Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    // using activations directly: remove softmax, if present
    if (nnet.GetComponent(nnet.NumComponents()-1).GetType() == Component::kSoftmax) {
      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
      nnet.RemoveComponent(nnet.NumComponents()-1);
    } else {
      KALDI_LOG << "The nnet was without softmax " << model_filename;
    }
    nnet.SetTrainOptions(trn_opts);

    // Read the class-frame-counts, compute priors
    PdfPrior log_prior(prior_opts);

    // Read transition model
    TransitionModel trans_model;
    ReadKaldiObject(transition_model_filename, &trans_model);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessLatticeReader den_lat_reader(den_lat_rspecifier);
    RandomAccessInt32VectorReader ref_ali_reader(ref_ali_rspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, nnet_diff;
    Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

    Timer time;
    double time_now = 0;
    KALDI_LOG << "TRAINING STARTED";

    int32 num_done = 0, num_no_ref_ali = 0, num_no_den_lat = 0,
      num_other_error = 0;

    kaldi::int64 total_frames = 0;
    double total_frame_acc = 0.0, utt_frame_acc;

    // do per-utterance processing
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!den_lat_reader.HasKey(utt)) {
        KALDI_WARN << "Utterance " << utt << ": found no lattice.";
        num_no_den_lat++;
        continue;
      }
      if (!ref_ali_reader.HasKey(utt)) {
        KALDI_WARN << "Utterance " << utt << ": found no reference alignment.";
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
      if (mat.NumRows() > max_frames) {
	KALDI_WARN << "Utterance " << utt << ": Skipped because it has " << mat.NumRows() << 
	  " frames, which is more than " << max_frames << ".";
	num_other_error++;
	continue;
      }
      // 2) get the denominator lattice, preprocess
      Lattice den_lat = den_lat_reader.Value(utt);
      if (den_lat.Start() == -1) {
        KALDI_WARN << "Empty lattice for utt " << utt;
        num_other_error++;
        continue;
      }
      if (old_acoustic_scale != 1.0) {
        fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale),
                          &den_lat);
      }
      // optional sort it topologically
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
        KALDI_WARN << "Denominator lattice has wrong length "
                   << max_time << " vs. " << mat.NumRows();
        num_other_error++;
        continue;
      }

      // get actual dims for this utt and nnet
      int32 num_frames = mat.NumRows(),
          num_fea = mat.NumCols(),
          num_pdfs = nnet.OutputDim();

      // 3) propagate the feature to get the log-posteriors (nnet w/o sofrmax)
      // push features to GPU
      feats.Resize(num_frames, num_fea, kUndefined);
      feats.CopyFromMat(mat);
      // possibly apply transform
      nnet_transf.Feedforward(feats, &feats_transf);
      // propagate through the nnet (assuming w/o softmax)
      nnet.Propagate(feats_transf, &nnet_out);
      // subtract the log_prior
      if (prior_opts.class_frame_counts != "") {
        log_prior.SubtractOnLogpost(&nnet_out);
      }
      // transfer it back to the host
      nnet_out_h.Resize(num_frames, num_pdfs, kUndefined);
      nnet_out.CopyToMat(&nnet_out_h);
      // release the buffers we don't need anymore
      feats.Resize(0,0);
      feats_transf.Resize(0,0);
      nnet_out.Resize(0,0);

      // 4) rescore the latice
      LatticeAcousticRescore(nnet_out_h, trans_model, state_times, &den_lat);
      if (acoustic_scale != 1.0 || lm_scale != 1.0)
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

      kaldi::Posterior post;

      if (do_smbr) {  // use state-level accuracies, i.e. sMBR estimation
        utt_frame_acc = LatticeForwardBackwardMpeVariants(
            trans_model, silence_phones, den_lat, ref_ali, "smbr", &post);
      } else {  // use phone-level accuracies, i.e. MPFE (minimum phone frame error)
        utt_frame_acc = LatticeForwardBackwardMpeVariants(
            trans_model, silence_phones, den_lat, ref_ali, "mpfe", &post);
      }

      // 6) convert the Posterior to a matrix
      nnet_diff_h.Resize(num_frames, num_pdfs, kSetZero);
      for (int32 t = 0; t < post.size(); t++) {
        for (int32 arc = 0; arc < post[t].size(); arc++) {
          int32 pdf = trans_model.TransitionIdToPdf(post[t][arc].first);
          nnet_diff_h(t, pdf) -= post[t][arc].second;
        }
      }

      KALDI_VLOG(1) << "Processed lattice for utterance " << num_done + 1
                    << " (" << utt << "): found " << den_lat.NumStates()
                    << " states and " << fst::NumArcs(den_lat) << " arcs.";

      KALDI_VLOG(1) << "Utterance " << utt << ": Average frame accuracy = "
                    << (utt_frame_acc/num_frames) << " over " << num_frames
                    << " frames.";

      // 7) backpropagate through the nnet
      nnet_diff.Resize(num_frames, num_pdfs, kUndefined);
      nnet_diff.CopyFromMat(nnet_diff_h);
      nnet.Backpropagate(nnet_diff, NULL);
      // relase the buffer, we don't need anymore
      nnet_diff.Resize(0,0);

      // increase time counter
      total_frame_acc += utt_frame_acc;
      total_frames += num_frames;
      num_done++;

      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << total_frames/time_now
                      << " frames per second.";
#if HAVE_CUDA==1        
        // check the GPU is not overheated
        CuDevice::Instantiate().CheckGpuHealth();
#endif
      }
    }

    // add the softmax layer back before writing
    KALDI_LOG << "Appending the softmax " << target_model_filename;
    nnet.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
    //store the nnet
    nnet.Write(target_model_filename, binary);

    time_now = time.Elapsed();
    KALDI_LOG << "TRAINING FINISHED; "
              << "Time taken = " << time_now/60 << " min; processed "
              << (total_frames/time_now) << " frames per second.";

    KALDI_LOG << "Done " << num_done << " files, "
              << num_no_ref_ali << " with no reference alignments, "
              << num_no_den_lat << " with no lattices, "
              << num_other_error << " with other errors.";

    KALDI_LOG << "Overall average frame-accuracy is "
              << (total_frame_acc/total_frames) << " over " << total_frames
              << " frames.";

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
