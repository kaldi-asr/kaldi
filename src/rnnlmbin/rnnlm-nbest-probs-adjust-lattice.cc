// rnnlmbin/rnnlm-nbest-probs-adjust-lattice.cc

// Copyright 2015-2017  Johns Hopkins University (author: Daniel Povey)

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
#include "rnnlm/rnnlm-training.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "rnnlm/rnnlm-core-compute.h"
#include "rnnlm/rnnlm-compute-state.h"
#include "nnet3/nnet-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/compose-lattice-pruned.h"
#include <fstream>
#include <sstream>

using std::ifstream;
using std::map;
using std::unordered_map;

namespace kaldi {

// This class computes and outputs
// the information about arc posteriors.

class ArcPosteriorComputer {
 public:
  // Note: 'clat' must be topologically sorted.
  ArcPosteriorComputer(const CompactLattice &clat,
                       BaseFloat min_post,
                       const TransitionModel *trans_model = NULL):
      clat_(clat), min_post_(min_post) {}
      

  // returns the number of arc posteriors that it output.
  void OutputPosteriors(
                        std::map<int, double> *word_to_count,
                        std::map<int, double> *word_to_count2,
                        double* sum,
                        double* sum2) {
//    *sum = 0;
    int32 num_post = 0;
    if (!ComputeCompactLatticeAlphas(clat_, &alpha_))
      return;
    if (!ComputeCompactLatticeBetas(clat_, &beta_))
      return;

    CompactLatticeStateTimes(clat_, &state_times_);
    if (clat_.Start() < 0)
      return;
    double tot_like = beta_[clat_.Start()];

    int32 num_states = clat_.NumStates();
    for (int32 state = 0; state < num_states; state++) {
      for (fst::ArcIterator<CompactLattice> aiter(clat_, state);
           !aiter.Done(); aiter.Next()) {
        const CompactLatticeArc &arc = aiter.Value();
        double arc_loglike = -ConvertToCost(arc.weight) +
            alpha_[state] + beta_[arc.nextstate] - tot_like;
        KALDI_ASSERT(arc_loglike < 0.1 &&
                     "Bad arc posterior in forward-backward computation");
        if (arc_loglike > 0.0) arc_loglike = 0.0;

        int32 word = arc.ilabel;
        BaseFloat arc_post = exp(arc_loglike);
        if (arc_post <= min_post_) continue;

        (*word_to_count)[word] += arc_post;
        (*word_to_count2)[word] += arc_post;
        *sum += arc_post;
        *sum2 += arc_post;
        num_post++;
      }
    }
  }
 private:
  const CompactLattice &clat_;
  std::vector<double> alpha_;
  std::vector<double> beta_;
  std::vector<int32> state_times_;

  BaseFloat min_post_;
};
} // namespace kaldi

// read the file and genereate a map from [utt-id] to [convo-id], stored in *m
void ReadUttToConvo(string filename, map<string, string> &m) {
  KALDI_ASSERT(m.size() == 0);
  ifstream ifile(filename.c_str());
  string utt, convo;
  while (ifile >> utt >> convo) {
    m[utt] = convo;
  }
}

// read a unigram count file and generate unigram mapping from [word-id] to
// its unigram prob
void ReadUnigram(string filename, std::vector<double> *unigram) {
  std::vector<double> &m = *unigram;
  ifstream ifile(filename.c_str());
  int32 word;
  double count;
  double sum = 0.0;
  while (ifile >> word >> count) {
    m[word] = count;
    sum += count;
  }

  for (int32 i = 0; i < m.size(); i++) {
    m[i] /= sum;
  }
}

// first set *key to the first field, and then break the remaining line into a vector of integers
void GetNumbersFromLine(std::string line, std::string *key, std::vector<int32> *v) {
  std::stringstream ss(line);
  ss >> *key;
  int32 i;
  while (ss >> i) {
    v->push_back(i);
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program computes the probability per word of the provided training\n"
        "data in 'egs' format as prepared by rnnlm-get-egs.  The interface is similar\n"
        "to rnnlm-train, except that it doesn't train, and doesn't write the model;\n"
        "it just prints the average probability to the standard output (in addition\n"
        "to printing various diagnostics to the standard error).\n"
        "\n"
        "Usage:\n"
        " rnnlm-compute-prob [options] <rnnlm> <word-embedding-matrix> <egs-rspecifier>\n"
        "e.g.:\n"
        " rnnlm-get-egs ... ark:- | \\\n"
        " rnnlm-compute-prob 0.raw 0.word_embedding ark:-\n"
        "(note: use rnnlm-get-word-embedding to get the word embedding matrix if\n"
        "you are using sparse word features.)\n";

    BaseFloat lm_scale = 0.5;
    BaseFloat acoustic_scale = 0.1;
    BaseFloat min_post = 0.0001;
    std::string use_gpu = "no";
    bool batchnorm_test_mode = true, dropout_test_mode = true;
    bool two_speaker_mode = true, one_best_mode = false;
    double correction_weight = 1.0;

    ParseOptions po(usage);
    rnnlm::RnnlmComputeStateComputationOptions opts;
    po.Register("lm-scale", &lm_scale, "Scaling factor for <lm-to-add>; its negative "
                "will be applied to <lm-to-subtract>.");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic "
                "probabilities (e.g. 0.1 for non-chain systems); important because "
                "of its effect on pruning.");
    po.Register("min-post", &min_post,
                "Arc posteriors below this value will be pruned away");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");
    po.Register("correction_weight", &correction_weight, "The weight on the "
                "correction term of the RNNLM scores.");
    po.Register("two_speaker_mode", &two_speaker_mode, "If true, use two "
                "speaker's utterances to estimate cache models.");
    po.Register("one_best_mode", &one_best_mode, "If true, use 1 best decoding "
                "results instead of lattice posteriors to estimate cache models.");

    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string rnnlm_rxfilename = po.GetArg(1),
                word_embedding_rxfilename = po.GetArg(2),
                text_filename = po.GetArg(3),
                utt_to_convo_file = po.GetArg(4),
                unigram_file = po.GetArg(5),
                lats_rspecifier = po.GetArg(6);

    map<string, string> utt2convo;
    ReadUttToConvo(utt_to_convo_file, utt2convo);

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().AllowMultithreading();
#endif

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);
    KALDI_ASSERT(IsSimpleNnet(rnnlm));

    if (!IsSimpleNnet(rnnlm))
      KALDI_ERR << "Input RNNLM in " << rnnlm_rxfilename
                << " is not the type of neural net we were looking for; "
          "failed IsSimpleNnet().";
    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &rnnlm);
    if (dropout_test_mode)
      SetDropoutTestMode(true, &rnnlm);

    CuMatrix<BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);

    std::vector<double> original_unigram(word_embedding_mat.NumRows(), 0.0);  // number of words
    ReadUnigram(unigram_file, &original_unigram);

    const rnnlm::RnnlmComputeStateInfo info(opts, rnnlm, word_embedding_mat);
    
    // Reads as compact lattice.
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);

    std::map<string, map<int, double> > per_convo_counts;
    std::map<string, map<int, double> > per_utt_counts;
    std::map<string, double> per_convo_sums;
    std::map<string, double> per_utt_sums;

    {
      SequentialCompactLatticeReader clat_reader(lats_rspecifier);

      for (; !clat_reader.Done(); clat_reader.Next()) {
        std::string utt_id = clat_reader.Key();
        kaldi::CompactLattice &clat = clat_reader.Value();
        
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
        kaldi::TopSortCompactLatticeIfNeeded(&clat);

        string convo_id = utt2convo[utt_id];
        if (two_speaker_mode) {
          std::string convo_id_2spk = std::string(convo_id.begin(), convo_id.end() - 2);
          convo_id = convo_id_2spk;
        }

        // Estimate cache models from 1-best hypothese instead of 
        // word-posteriors from first-pass decoded lattices
        if (one_best_mode) {
          kaldi::CompactLattice best_path;
          kaldi::CompactLatticeShortestPath(clat, &best_path);
          clat = best_path;
        }

        kaldi::ArcPosteriorComputer computer(clat, min_post);

        computer.OutputPosteriors(
                                  &(per_convo_counts[convo_id]),
                                  &(per_utt_counts[utt_id]),
                                  &(per_convo_sums[convo_id]),
                                  &(per_utt_sums[utt_id]));

      }
      clat_reader.Close();
    }
    
    std::map<string, std::vector<int> > text;
    {
      std::ifstream ifile(text_filename.c_str());
      std::string line;
      std::string utt_id;
      while (getline(ifile, line)) {
        std::vector<int> v;
        GetNumbersFromLine(line, &utt_id, &v);
        text[utt_id] = v;
      }
    }

    double total_probs = 0.0;
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      // compute a local cache model for adjusting RNNLM scores.
      std::string key = compact_lattice_reader.Key();
      std::string convo_id = utt2convo[key];
      if (two_speaker_mode) {
        std::string convo_id_2spk = std::string(convo_id.begin(), convo_id.end() - 2);
        convo_id = convo_id_2spk;
      }
      KALDI_ASSERT(convo_id != "");
      // collect counts of nearby sentences of the current utterance.
      map<int, double> unigram = per_convo_counts[convo_id];
      for (map<int, double>::iterator iter = per_utt_counts[key].begin();
                                      iter != per_utt_counts[key].end(); ++iter) {
        unigram[iter->first] = unigram[iter->first] - iter->second;
      }
      double sum = per_convo_sums[convo_id] - per_utt_sums[key];

      double debug_sum = 0.0;
      for (map<int, double>::iterator iter = unigram.begin();
                                      iter != unigram.end(); ++iter) {
        iter->second /= sum;
        debug_sum += iter->second;
      }
      KALDI_ASSERT(ApproxEqual(debug_sum, 1.0));

      // adjust the RNNLM weights and compute perplexity
      std::map<string, std::vector<int> >::iterator iter = text.find(key);
      KALDI_ASSERT(iter != text.end());
      RnnlmComputeState rnnlm_compute_state(info, opts.bos_index);
      std::vector<int> v = iter->second;
      for (int32 i = 1; i < v.size(); i++) {
        int32 word_id = v[i];
        CuMatrix<BaseFloat> word_logprobs(1, word_embedding_mat.NumRows());
        rnnlm_compute_state.GetLogProbOfWords(&word_logprobs);
        word_logprobs.ApplyLogSoftMaxPerRow(word_logprobs);
        // now every element is a legit probability
        if (correction_weight > 0) {
          for (map<int, double>::iterator iter = unigram.begin();
                                        iter != unigram.end(); iter++) {
              double u = iter->second;  // already unigram probs

              const double C = 0.00000001;
              double correction = (u + C) / (original_unigram[iter->first] + C);
              // smoothing by a background unigram distribution original_unigram
              correction = 0.5 * correction + 0.5;
              if (correction != 0) {
                  word_logprobs.Row(0).Range(iter->first, 1).Add(Log(correction) 
                      * correction_weight);
              }
          }
          word_logprobs.ApplyLogSoftMaxPerRow(word_logprobs);
        }
        rnnlm_compute_state.AddWord(word_id);
        total_probs += word_logprobs(0, word_id);
      }
      int32 word_id = opts.eos_index;
      total_probs += rnnlm_compute_state.LogProbOfWord(word_id);
    }
    KALDI_LOG << "Log probs " << total_probs;

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
