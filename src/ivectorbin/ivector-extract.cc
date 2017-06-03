// ivectorbin/ivector-extract.cc

// Copyright 2013  Daniel Povey

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
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "util/kaldi-thread.h"

namespace kaldi {

// This class will be used to parallelize over multiple threads the job
// that this program does.  The work happens in the operator (), the
// output happens in the destructor.
class IvectorExtractTask {
 public:
  IvectorExtractTask(const IvectorExtractor &extractor,
                     std::string utt,
                     const Matrix<BaseFloat> &feats,
                     const Posterior &posterior,
                     BaseFloatVectorWriter *writer,
                     double *tot_auxf_change):
      extractor_(extractor), utt_(utt), feats_(feats), posterior_(posterior),
      writer_(writer), tot_auxf_change_(tot_auxf_change) { }

  void operator () () {
    bool need_2nd_order_stats = false;

    IvectorExtractorUtteranceStats utt_stats(extractor_.NumGauss(),
                                             extractor_.FeatDim(),
                                             need_2nd_order_stats);

    utt_stats.AccStats(feats_, posterior_);

    ivector_.Resize(extractor_.IvectorDim());
    ivector_(0) = extractor_.PriorOffset();

    if (tot_auxf_change_ != NULL) {
      double old_auxf = extractor_.GetAuxf(utt_stats, ivector_);
      extractor_.GetIvectorDistribution(utt_stats, &ivector_, NULL);
      double new_auxf = extractor_.GetAuxf(utt_stats, ivector_);
      auxf_change_ = new_auxf - old_auxf;
    } else {
      extractor_.GetIvectorDistribution(utt_stats, &ivector_, NULL);
    }
  }
  ~IvectorExtractTask() {
    if (tot_auxf_change_ != NULL) {
      double T = TotalPosterior(posterior_);
      *tot_auxf_change_ += auxf_change_;
      KALDI_VLOG(2) << "Auxf change for utterance " << utt_ << " was "
                    << (auxf_change_ / T) << " per frame over " << T
                    << " frames (weighted)";
    }
    // We actually write out the offset of the iVectors from the mean of the
    // prior distribution; this is the form we'll need it in for scoring.  (most
    // formulations of iVectors have zero-mean priors so this is not normally an
    // issue).
    ivector_(0) -= extractor_.PriorOffset();
    KALDI_VLOG(2) << "Ivector norm for utterance " << utt_
                  << " was " << ivector_.Norm(2.0);
    writer_->Write(utt_, Vector<BaseFloat>(ivector_));
  }
 private:
  const IvectorExtractor &extractor_;
  std::string utt_;
  Matrix<BaseFloat> feats_;
  Posterior posterior_;
  BaseFloatVectorWriter *writer_;
  double *tot_auxf_change_; // if non-NULL we need the auxf change.
  Vector<double> ivector_;
  double auxf_change_;
};

int32 RunPerSpeaker(const std::string &ivector_extractor_rxfilename,
                   const IvectorEstimationOptions &opts,
                   bool compute_objf_change,
                   const std::string &spk2utt_rspecifier,
                   const std::string &feature_rspecifier,
                   const std::string &posterior_rspecifier,
                   const std::string &ivector_wspecifier) {
  IvectorExtractor extractor;
  ReadKaldiObject(ivector_extractor_rxfilename, &extractor);
  SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
  RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
  RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
  BaseFloatVectorWriter ivector_writer(ivector_wspecifier);

  double tot_auxf_change = 0.0, tot_post = 0.0, tot_norm = 0.0;
  int32 num_utt_done = 0, num_utt_err = 0,
      num_spk_done = 0, num_spk_err = 0;

  for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
    std::string spk = spk2utt_reader.Key();
    const std::vector<std::string> &utts = spk2utt_reader.Value();

    bool need_2nd_order_stats = false;

    IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(),
                                             extractor.FeatDim(),
                                             need_2nd_order_stats);

    for (size_t i = 0; i < utts.size(); i++) {
      const std::string &utt = utts[i];
      if (!feature_reader.HasKey(utt)) {
        KALDI_WARN << "No features present for utterance " << utt;
        num_utt_err++;
        continue;
      }
      const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
      if (!posterior_reader.HasKey(utt)) {
        KALDI_WARN << "No posteriors present for utterance " << utt;
        num_utt_err++;
        continue;
      }
      Posterior posterior = posterior_reader.Value(utt);
      if (feats.NumRows() != posterior.size()) {
        KALDI_WARN << "Posterior has wrong size " << posterior.size()
                   << " vs. feats " << feats.NumRows() << " for "
                   << utt;
        num_utt_err++;
        continue;
      }
      ScalePosterior(opts.acoustic_weight, &posterior);
      num_utt_done++;
      utt_stats.AccStats(feats, posterior);
    }

    if (utt_stats.NumFrames() == 0.0) {
      KALDI_WARN << "No stats accumulated for speaker " << spk;
      num_spk_err++;
      continue;
    } else {
      if (opts.max_count > 0 && utt_stats.NumFrames() > opts.max_count) {
        double scale = opts.max_count / utt_stats.NumFrames();
        utt_stats.Scale(scale);
        KALDI_LOG << "Scaling stats for speaker " << spk << " by scale "
                  << scale << " due to --max-count=" << opts.max_count;
      }

      Vector<double> ivector(extractor.IvectorDim());
      ivector(0) = extractor.PriorOffset();

      if (compute_objf_change) {
        double old_auxf = extractor.GetAuxf(utt_stats, ivector);
        extractor.GetIvectorDistribution(utt_stats, &ivector, NULL);
        double new_auxf = extractor.GetAuxf(utt_stats, ivector);
        double auxf_change = new_auxf - old_auxf;

        KALDI_LOG << "Auxf change for speaker " << spk << " was "
                  << (auxf_change / utt_stats.NumFrames()) << " per frame, over "
                  << utt_stats.NumFrames() << " frames (weighted).";
        tot_auxf_change += auxf_change;
      } else {
        extractor.GetIvectorDistribution(utt_stats, &ivector, NULL);
      }
      // We actually write out the offset of the iVectors from the mean of the
      // prior distribution; this is the form we'll need it in for scoring and
      // as a feature for neural nets.  (most formulations of iVectors have
      // zero-mean priors so this is not normally an issue).
      ivector(0) -= extractor.PriorOffset();
      KALDI_LOG << "Ivector norm for speaker " << spk
                << " was " << ivector.Norm(2.0);

      tot_norm += ivector.Norm(2.0) * utt_stats.NumFrames();
      tot_post += utt_stats.NumFrames();
      num_spk_done++;
      Vector<BaseFloat> ivector_flt(ivector);
      ivector_writer.Write(spk, ivector_flt);
    }
  }

  KALDI_LOG << "Done " << num_spk_done << " speakers; " << num_spk_err
            << " with errors.  " << num_utt_done << " utterances "
            << "were processed, " << num_utt_err << " with errors.";
  if (tot_post != 0.0) {
    if (compute_objf_change) {
      KALDI_LOG << "Overall weighted-average objective function improvement was "
                << (tot_auxf_change / tot_post) << " over " << tot_post
                << " frames (weighted)";
    }
    KALDI_LOG << "Average iVector norm (weighted by frames) was "
              << (tot_norm / tot_post) << " over " << tot_post
              << " frames (weighted)";
  }
  return (num_spk_done != 0 ? 0 : 1);
}

}



int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Extract iVectors for utterances, using a trained iVector extractor,\n"
        "and features and Gaussian-level posteriors\n"
        "Usage:  ivector-extract [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <ivector-wspecifier>\n"
        "e.g.: \n"
        " fgmm-global-gselect-to-post 1.ubm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
        "  ivector-extract final.ie '$feats' ark,s,cs:- ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    bool compute_objf_change = true;
    IvectorEstimationOptions opts;
    std::string spk2utt_rspecifier;
    TaskSequencerConfig sequencer_config;
    po.Register("compute-objf-change", &compute_objf_change,
                "If true, compute the change in objective function from using "
                "nonzero iVector (a potentially useful diagnostic).  Combine "
                "with --verbose=2 for per-utterance information");
    po.Register("spk2utt", &spk2utt_rspecifier, "Supply this option if you "
                "want iVectors to be output at the per-speaker level, estimated "
                "using stats accumulated from multiple utterances.  Note: this "
                "is not the normal way iVectors are obtained for speaker-id. "
                "This option will cause the program to ignore the --num-threads "
                "option.");

    opts.Register(&po);
    sequencer_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posterior_rspecifier = po.GetArg(3),
        ivectors_wspecifier = po.GetArg(4);


    if (spk2utt_rspecifier.empty()) {
      // g_num_threads affects how ComputeDerivedVars is called when we read the
      // extractor.
      g_num_threads = sequencer_config.num_threads;
      IvectorExtractor extractor;
      ReadKaldiObject(ivector_extractor_rxfilename, &extractor);

      double tot_auxf_change = 0.0, tot_t = 0.0;
      int32 num_done = 0, num_err = 0;

      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
      BaseFloatVectorWriter ivector_writer(ivectors_wspecifier);

      {
        TaskSequencer<IvectorExtractTask> sequencer(sequencer_config);
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          if (!posterior_reader.HasKey(utt)) {
            KALDI_WARN << "No posteriors for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          Posterior posterior = posterior_reader.Value(utt);

          if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
            KALDI_WARN << "Size mismatch between posterior " << posterior.size()
                       << " and features " << mat.NumRows() << " for utterance "
                       << utt;
            num_err++;
            continue;
          }

          double *auxf_ptr = (compute_objf_change ? &tot_auxf_change : NULL );

          double this_t = opts.acoustic_weight * TotalPosterior(posterior),
              max_count_scale = 1.0;
          if (opts.max_count > 0 && this_t > opts.max_count) {
            max_count_scale = opts.max_count / this_t;
            KALDI_LOG << "Scaling stats for utterance " << utt << " by scale "
                      << max_count_scale << " due to --max-count="
                      << opts.max_count;
            this_t = opts.max_count;
          }
          ScalePosterior(opts.acoustic_weight * max_count_scale,
                         &posterior);
          // note: now, this_t == sum of posteriors.

          sequencer.Run(new IvectorExtractTask(extractor, utt, mat, posterior,
                                               &ivector_writer, auxf_ptr));

          tot_t += this_t;
          num_done++;
        }
        // Destructor of "sequencer" will wait for any remaining tasks.
      }

      KALDI_LOG << "Done " << num_done << " files, " << num_err
                << " with errors.  Total (weighted) frames " << tot_t;
      if (compute_objf_change)
        KALDI_LOG << "Overall average objective-function change from estimating "
                  << "ivector was " << (tot_auxf_change / tot_t) << " per frame "
                  << " over " << tot_t << " (weighted) frames.";

      return (num_done != 0 ? 0 : 1);
    } else {
      KALDI_ASSERT(sequencer_config.num_threads == 1 &&
                   "--spk2utt option is incompatible with --num-threads option");
      return RunPerSpeaker(ivector_extractor_rxfilename,
                           opts,
                           compute_objf_change,
                           spk2utt_rspecifier,
                           feature_rspecifier,
                           posterior_rspecifier,
                           ivectors_wspecifier);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
