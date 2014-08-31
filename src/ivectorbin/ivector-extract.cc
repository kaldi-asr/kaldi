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
#include "thread/kaldi-task-sequence.h"

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
      int32 T = posterior_.size();
      *tot_auxf_change_ += auxf_change_;
      KALDI_VLOG(2) << "Auxf change for utterance " << utt_ << " was "
                    << (auxf_change_ / T) << " per frame over " << T
                    << " frames.";
    }
    // We actually write out the offset of the iVector's from the mean of the
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



}


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Extract iVectors for utterances, using a trained iVector extractor,\n"
        "and features and Gaussian-level posteriors\n"
        "Usage:  ivector-extract [options] <model-in> <feature-rspecifier>"
        "<posteriors-rspecifier> <ivector-wspecifier>\n"
        "e.g.: \n"
        " fgmm-global-gselect-to-post 1.ubm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
        "  ivector-extract final.ie '$feats' ark,s,cs:- ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    bool compute_objf_change = true;
    IvectorExtractorStatsOptions stats_opts;
    TaskSequencerConfig sequencer_config;
    po.Register("compute-objf-change", &compute_objf_change,
                "If true, compute the change in objective function from using "
                "nonzero iVector (a potentially useful diagnostic).  Combine "
                "with --verbose=2 for per-utterance information");
    stats_opts.Register(&po);
    sequencer_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        ivectors_wspecifier = po.GetArg(4);

    // g_num_threads affects how ComputeDerivedVars is called when we read the
    // extractor.
    g_num_threads = sequencer_config.num_threads; 
    IvectorExtractor extractor;
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);

    double tot_auxf_change = 0.0;
    int64 tot_t = 0;
    int32 num_done = 0, num_err = 0;
    
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    BaseFloatVectorWriter ivector_writer(ivectors_wspecifier);

    {
      TaskSequencer<IvectorExtractTask> sequencer(sequencer_config);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string key = feature_reader.Key();
        if (!posteriors_reader.HasKey(key)) {
          KALDI_WARN << "No posteriors for utterance " << key;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);

        if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
          KALDI_WARN << "Size mismatch between posterior " << posterior.size()
                     << " and features " << mat.NumRows() << " for utterance "
                     << key;
          num_err++;
          continue;
        }

        double *auxf_ptr = (compute_objf_change ? &tot_auxf_change : NULL );

        sequencer.Run(new IvectorExtractTask(extractor, key, mat, posterior,
                                             &ivector_writer, auxf_ptr));
                      
        tot_t += posterior.size();
        num_done++;
      }
      // Destructor of "sequencer" will wait for any remaining tasks.
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.  Total frames " << tot_t;

    if (compute_objf_change)
      KALDI_LOG << "Overall average objective-function change from estimating "
                << "ivector was " << (tot_auxf_change / tot_t) << " per frame "
                << " over " << tot_t << " frames.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
