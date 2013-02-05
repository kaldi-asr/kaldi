// nnetbin/nnet-train-xent-hardlab-perutt.cc

// Copyright 2011  Karel Vesely

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
    if (state_times[i] < log_like.NumRows()) // end state may be past this..
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
          arc.weight.SetValue2(-log_like(t,pdf_id) + arc.weight.Value2());
          aiter.SetValue(arc);
        }
      }
    }
  }
}

} // namespace kaldi





int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform iteration of Neural Network MMI training by stochastic gradient descent.\n"
        "Usage:  nnet-train-mmi-sequential [options] <model-in> <transition-model-in> <feature-rspecifier> <den-lat-rspecifier> <ali-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-mmi-sequential nnet.init trans.mdl scp:train.scp scp:denlats.scp ark:train.ali nnet.iter1\n";

    ParseOptions po(usage);
    bool binary = false, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    BaseFloat learn_rate = 0.00001,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    std::string feature_transform,class_frame_counts;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");
    po.Register("class-frame-counts", &class_frame_counts, "Class frame counts to compute the class priors");

    kaldi::BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;
    kaldi::BaseFloat old_acoustic_scale = 0.0;
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    po.Register("old-acoustic-scale", &old_acoustic_scale,
                "Add the current acoustic scores with some scale.");

    po.Read(argc, argv);

    if (po.NumArgs() != 6-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        transition_model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        den_lat_rspecifier = po.GetArg(4),
        num_ali_rspecifier = po.GetArg(5);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(6);
    }

     
    using namespace kaldi;
    typedef kaldi::int32 int32;


    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    if(nnet.Layer(nnet.LayerCount()-1)->GetType() == Component::kSoftmax) {
      KALDI_ERR << "The MLP has to be without softmax...";
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
    RandomAccessInt32VectorReader num_ali_reader(num_ali_rspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, nnet_diff;
    Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

    // Read the class-counts, compute priors
    CuVector<BaseFloat> log_priors;
    if(class_frame_counts != "") {
      Vector<BaseFloat> tmp_priors;
      //read the values
      Input in;
      in.OpenTextMode(class_frame_counts);
      tmp_priors.Read(in.Stream(), false);
      in.Close();
     
      //create inv. priors, or log inv priors 
      BaseFloat sum = tmp_priors.Sum();
      tmp_priors.Scale(1.0/sum);
      tmp_priors.ApplyLog();
      
      // push priors to GPU
      log_priors.Resize(tmp_priors.Dim());
      log_priors.CopyFromVec(tmp_priors);
    }




    Timer tim;
    double time_next=0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_num_ali = 0, num_no_den_lat = 0, num_other_error = 0, num_frm_drop = 0;

    double total_like = 0.0, lat_like;
    double total_lat_ac_like = 0.0, lat_ac_like; // acoustic likelihood weighted by posterior.
    double total_mmi_obj = 0.0, mmi_obj, mmi_obj_notdrop; // mmi objective function

    //do per-utterance processing,,,
    for( ; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      bool skip_utt = false;
      if (!den_lat_reader.HasKey(key)) { num_no_den_lat++; skip_utt = true; }
      if (!num_ali_reader.HasKey(key)) { num_no_num_ali++; skip_utt = true; }
      if (!skip_utt) {
        //1) get the features, numerator alignment
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &num_ali = num_ali_reader.Value(key);
        // check for temporal length of numerator alignments
        if ((int32)num_ali.size() != mat.NumRows()) {
          KALDI_WARN << "Numerator alignment has wrong length "<< num_ali.size() << " vs. "<< mat.NumRows();
          num_other_error++;
          continue;
        }
       
        
        //2) get the denominator lattice, preprocess
        Lattice den_lat = den_lat_reader.Value(key);
        if (old_acoustic_scale != 1.0) {
          fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale), &den_lat);
        }
        // optionaly sort it topologically
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
          KALDI_WARN << "Denominator lattice has wrong length "<< max_time << " vs. "<< mat.NumRows();
          num_other_error++;
          continue;
        }
        
        //3) propagate the feature to get the log-posteriors (nnet w/o sofrmax)
        // push features to GPU
        feats = mat;
        // possibly apply transform
        nnet_transf.Feedforward(feats, &feats_transf);
        // propagate through the nnet (assuming w/o softmax)
        nnet.Propagate(feats_transf, &nnet_out);
        // subtract the log_priors
        if(log_priors.Dim() > 0) {
          nnet_out.AddVecToRows(-1.0,log_priors);
        }
        // transfer it back to the host
        nnet_out_h.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
        nnet_out.CopyToMat(&nnet_out_h);

        //4) rescore the latice
        LatticeAcousticRescore(nnet_out_h, trans_model, state_times, &den_lat);
        if (acoustic_scale != 1.0 || lm_scale != 1.0)
          fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

        //5) get the posteriors
        kaldi::Posterior post;
        lat_like = kaldi::LatticeForwardBackward(den_lat, &post, &lat_ac_like);

        //6) convert the Posterior to a matrix
        nnet_diff_h.Resize(nnet_out_h.NumRows(), nnet_out_h.NumCols());
        for(int32 t=0; t<post.size(); t++) {
          for(int32 arc=0; arc<post[t].size(); arc++) {
            int32 pdf = trans_model.TransitionIdToPdf(post[t][arc].first);
            nnet_diff_h(t, pdf) += post[t][arc].second;
          }
        }

        //7) Calculate the MMI-objective function
        // Calculate it twice, 
        // - once including the frames with zero posterior under the alignment
        // - once leaving these ``suspicious'' frames out (mmi_obj_notdrop)
        mmi_obj = mmi_obj_notdrop = 0.0;
        int32 frm_drop = 0; 
        for(int32 t=0; t<nnet_diff_h.NumRows(); t++) {
          int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
          double posterior = nnet_diff_h(t, pdf);
          if(posterior < 1e-20) {
            posterior = 1e-20;
            frm_drop++;
          } else {
            mmi_obj_notdrop += log(posterior);
          }
          //here we sum even the dropped frames
          mmi_obj += log(posterior);
        }

        // report
        std::stringstream ss;
        ss << "Utt " << num_done+1 << " : " << key
                  << " (" << feats_transf.NumRows() << "frm)"
                  << " lat_like/frm: " << lat_like / feats_transf.NumRows()
                  << "\t"
                  << " mmi_obj/frm: " << mmi_obj/feats_transf.NumRows();
        if (frm_drop > 0) {
          ss << " (mmi_obj_notdrop/frm: "
             << mmi_obj_notdrop/(nnet_diff_h.NumRows()-frm_drop)
             << ")";
        }
        KALDI_LOG << ss.str();
        // accumulate
        total_like += lat_like;
        total_lat_ac_like += lat_ac_like;
        total_mmi_obj += mmi_obj;
        
        //7a) Check there is non-zero FW-BW probability 
        //at the alignment position (ie. the correct path 
        //exist within the lattice, and MMI has a chance to correct it)
        frm_drop = 0;
        for(int32 t=0; t<nnet_diff_h.NumRows(); t++) {
          int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
          if(nnet_diff_h(t, pdf) < 1e-20) {
            frm_drop++;
            nnet_diff_h.Row(t).Set(0.0);
          }
        }
        if (frm_drop > 0) {
          KALDI_WARN << " The lattice is likely to miss part of the correct path..."
                       << " Dropped: " << frm_drop << "/" << nnet_diff_h.NumRows() << " frames";
        }
        num_frm_drop += frm_drop;

        

        //8) subtract the pdf-Viterbi-path
        for(int32 t=0; t<nnet_diff_h.NumRows(); t++) {
          int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
          ///
          /// Make sure the sum in vector is as close as possible 
          /// to zero, reduce round-off errors:
          ///
          nnet_diff_h(t, pdf) += -nnet_diff_h.Row(t).Sum(); 
          ///
        }

        //9) backpropagate through the nnet
        if (!crossvalidate) {
          nnet_diff = nnet_diff_h;
          nnet.Backpropagate(nnet_diff, NULL);
        }

        //increase time counter
        tot_t += feats_transf.NumRows();
        num_done++;
      }
    }
       
    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }
    
    std::cout << "\n" << std::flush;

    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED " 
              << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " 
              << num_no_num_ali << " with no numerator alignments, " 
              << num_no_den_lat << " with no denumerator lattices, " 
              << num_other_error << " with other errors.";

    KALDI_LOG << "Overall MMI-objective/frame is "
              << (total_mmi_obj/tot_t) << " over " << tot_t << " frames. "
              << " Dropped " << num_frm_drop << " frames."
              << "\n";




#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif


    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
