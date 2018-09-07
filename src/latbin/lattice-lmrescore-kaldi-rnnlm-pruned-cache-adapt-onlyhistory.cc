// latbin/lattice-lmrescore-kaldi-rnnlm-pruned-cache-adapt-onlyhistory.cc

// Copyright 2017 Johns Hopkins University (author: Daniel Povey)
//           2017 Hainan Xu
//           2018 Ke Li

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
#include "fstext/fstext-lib.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "lm/const-arpa-lm.h"
#include "util/common-utils.h"
#include "nnet3/nnet-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/compose-lattice-pruned.h"
#include <fstream>

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
}  // namespace kaldi

void ReadUttToConvo(string filename, map<string, string> &m) {
  KALDI_ASSERT(m.size() == 0);
  ifstream ifile(filename.c_str());
  string utt, convo;
  while (ifile >> utt >> convo) {
    m[utt] = convo;
  }
}

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;
    using fst::ReadFstKaldi;

    const char *usage =
        "Rescores lattice with kaldi-rnnlm. This script is called from \n"
        "scripts/rnnlm/lmrescore_pruned.sh. An example for rescoring \n"
        "lattices is at egs/swbd/s5c/local/rnnlm/run_lstm.sh \n"
        "\n"
        "Usage: lattice-lmrescore-kaldi-rnnlm-pruned-cache-adapt-onlyhistory [options] \\\n"
        "             <old-lm-rxfilename> <embedding-file> \\\n"
        "             <raw-rnnlm-rxfilename> \\\n"
        "             <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-lmrescore-kaldi-rnnlm-pruned-cache-adapt-onlyhistory --lm-scale=-1.0 fst_words.txt \\\n"
        "              --bos-symbol=1 --eos-symbol=2 \\\n"
        "              data/lang_test/G.fst word_embedding.mat \\\n"
        "              final.raw ark:in.lats ark:out.lats\n\n";

    ParseOptions po(usage);
    rnnlm::RnnlmComputeStateComputationOptions opts;
    ComposeLatticePrunedOptions compose_opts;

    BaseFloat lm_scale = 0.5;
    BaseFloat acoustic_scale = 0.1;
    BaseFloat correction_weight = 0.8;
    int32 max_ngram_order = 3;
    BaseFloat min_post = 0.0001;
    bool use_carpa = false;
    bool two_speaker_mode = false, one_best_mode = false;

    po.Register("lm-scale", &lm_scale, "Scaling factor for <lm-to-add>; its "
                "negative will be applied to <lm-to-subtract>.");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for "
                "acoustic probabilities (e.g. 0.1 for non-chain systems); "
                "important because of its effect on pruning.");
    po.Register("use-const-arpa", &use_carpa, "If true, read the old-LM file "
                "as a const-arpa file as opposed to an FST file");
    po.Register("correction_weight", &correction_weight, "The weight on the "
                "correction term of the RNNLM scores.");
    po.Register("max-ngram-order", &max_ngram_order,
                "If positive, allow RNNLM histories longer than this to be "
                "identified with each other for rescoring purposes (an "
                "approximation that saves time and reduces output lattice "
                "size).");
    po.Register("min-post", &min_post,
                "Arc posteriors below this value will be pruned away");
    po.Register("two_speaker_mode", &two_speaker_mode, "If true, use two "
                "speaker's utterances to estimate cache models or "
                "as the input of DNN models.");
    po.Register("one_best_mode", &one_best_mode, "If true, use 1 best decoding "
                "results instead of lattice posteriors to estimate cache "
                "models or as the input of DNN models.");

    opts.Register(&po);
    compose_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    if (opts.bos_index == -1 || opts.eos_index == -1) {
      KALDI_ERR << "must set --bos-symbol and --eos-symbol options";
    }

    std::string lm_to_subtract_rxfilename = po.GetArg(1),
                word_embedding_rxfilename = po.GetArg(2),
                rnnlm_rxfilename = po.GetArg(3),
                lats_rspecifier = po.GetArg(4),
                lats_wspecifier = po.GetArg(5),
                utt_to_convo_file = po.GetArg(6),
                unigram_file = po.GetArg(7);

    // for G.fst
    fst::ScaleDeterministicOnDemandFst *lm_to_subtract_det_scale = NULL;
    fst::BackoffDeterministicOnDemandFst<StdArc>
      *lm_to_subtract_det_backoff = NULL;
    VectorFst<StdArc> *lm_to_subtract_fst = NULL;

    // for G.carpa
    ConstArpaLm* const_arpa = NULL;
    fst::DeterministicOnDemandFst<StdArc> *carpa_lm_to_subtract_fst = NULL;

    KALDI_LOG << "Reading old LMs...";
    if (use_carpa) {
      const_arpa = new ConstArpaLm();
      ReadKaldiObject(lm_to_subtract_rxfilename, const_arpa);
      carpa_lm_to_subtract_fst = new ConstArpaLmDeterministicFst(*const_arpa);
      lm_to_subtract_det_scale
        = new fst::ScaleDeterministicOnDemandFst(-lm_scale,
                                                 carpa_lm_to_subtract_fst);
    } else {
      lm_to_subtract_fst = fst::ReadAndPrepareLmFst(
          lm_to_subtract_rxfilename);
      lm_to_subtract_det_backoff =
        new fst::BackoffDeterministicOnDemandFst<StdArc>(*lm_to_subtract_fst);
      lm_to_subtract_det_scale =
           new fst::ScaleDeterministicOnDemandFst(-lm_scale,
                                                  lm_to_subtract_det_backoff);
    }

    map<string, string> utt2convo;
    ReadUttToConvo(utt_to_convo_file, utt2convo);

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);
    KALDI_ASSERT(IsSimpleNnet(rnnlm));

    CuMatrix<BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);

    // number of words
    std::vector<double> original_unigram(word_embedding_mat.NumRows(), 0.0);
    ReadUnigram(unigram_file, &original_unigram);

    const rnnlm::RnnlmComputeStateInfo info(opts, rnnlm, word_embedding_mat);

    // Reads and writes as compact lattice.
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 num_done = 0, num_err = 0;

    std::map<string, map<int, double> > per_convo_counts;
    std::map<string, map<int, double> > per_utt_counts;
    std::map<string, double> per_convo_sums;
    std::map<string, double> per_utt_sums;

    std::vector<string> utt_ids;
    {
      SequentialCompactLatticeReader clat_reader(lats_rspecifier);

      for (; !clat_reader.Done(); clat_reader.Next()) {
        std::string utt_id = clat_reader.Key();
        utt_ids.push_back(utt_id);
        kaldi::CompactLattice &clat = clat_reader.Value();

        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
        kaldi::TopSortCompactLatticeIfNeeded(&clat);

        string convo_id = utt2convo[utt_id];
        if (two_speaker_mode) {
          std::string convo_id_2spk = std::string(convo_id.begin(),
                                                  convo_id.end() - 2);
          convo_id = convo_id_2spk;
        }

        // Estimate cache models from 1-best hypotheses instead of
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

    std::map<string, map<int, double> > per_utt_hists;
    // std::map<string, double> per_utt_hists_sums;
    std::vector<string>::iterator it = utt_ids.begin();
    std::string init_utt_id = *it;
    per_utt_hists[init_utt_id] = per_utt_counts[init_utt_id];
    // KALDI_LOG << "init utt id " << init_utt_id;
    for (int32 i = 1; i < utt_ids.size(); i++) {
        std::string utt_id = *(it + i);
        std::string utt_id_prev = *(it + i - 1);
        // copy the previous speaker's utts to the current one
        per_utt_hists[utt_id] = per_utt_hists[utt_id_prev];
        // add the current utt to the counts for the current speaker
        for (std::map<int, double>::iterator cur_utt =
             per_utt_counts[utt_id].begin();
             cur_utt != per_utt_counts[utt_id].end(); ++cur_utt) {
            int32 word = cur_utt->first;
            BaseFloat count = cur_utt->second;
            per_utt_hists[utt_id][word] += count;
        }
    }

    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      // compute a unigram cache model for adjusting RNNLM scores.
      std::string key = compact_lattice_reader.Key();
      std::string convo_id = utt2convo[key];
      if (two_speaker_mode) {
        std::string convo_id_2spks = std::string(convo_id.begin(),
                                                 convo_id.end() - 2);
        convo_id = convo_id_2spks;
      }
      KALDI_ASSERT(convo_id != "");

      map<int, double> unigram = per_utt_hists[key];
      for (map<int, double>::iterator iter = per_utt_counts[key].begin();
                                      iter != per_utt_counts[key].end();
                                      ++iter) {
        unigram[iter->first] = (unigram[iter->first] - iter->second);
      }
      double sum = 0.0;
      for (map<int, double>::iterator iter = unigram.begin();
                                      iter != unigram.end(); ++iter) {
        sum += iter->second;
      }
      double debug_sum = 0.0;
      for (map<int, double>::iterator iter = unigram.begin();
                                      iter != unigram.end(); ++iter) {
        if (sum > 0) {
            iter->second /= sum;
        }
        debug_sum += iter->second;
      }
      KALDI_ASSERT(ApproxEqual(debug_sum, 1.0));

      // Rescoring and pruning happens below.
      rnnlm::KaldiRnnlmDeterministicFst* lm_to_add_orig =
           new rnnlm::KaldiRnnlmDeterministicFst(max_ngram_order, info,
                                                 correction_weight,
                                                 unigram, original_unigram);
      fst::DeterministicOnDemandFst<StdArc> *lm_to_add =
         new fst::ScaleDeterministicOnDemandFst(lm_scale, lm_to_add_orig);

      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();

      // Before composing with the LM FST, we scale the lattice weights
      // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
      // We do it this way so we can determinize and it will give the
      // right effect (taking the "best path" through the LM) regardless
      // of the sign of lm_scale.
      if (acoustic_scale != 1.0) {
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
      }
      TopSortCompactLatticeIfNeeded(&clat);

      fst::ComposeDeterministicOnDemandFst<StdArc> combined_lms(
          lm_to_subtract_det_scale, lm_to_add);

      // Composes lattice with language model.
      CompactLattice composed_clat;
      ComposeCompactLatticePruned(compose_opts, clat,
                                  &combined_lms, &composed_clat);

      if (composed_clat.NumStates() == 0) {
        // Something went wrong.  A warning will already have been printed.
        num_err++;
      } else {
        if (acoustic_scale != 1.0) {
          if (acoustic_scale == 0.0)
            KALDI_ERR << "Acoustic scale cannot be zero.";
          fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                            &composed_clat);
        }
        compact_lattice_writer.Write(key, composed_clat);
        num_done++;
      }
      delete lm_to_add_orig;
      delete lm_to_add;
    }

    delete lm_to_subtract_fst;
    delete lm_to_subtract_det_backoff;
    delete lm_to_subtract_det_scale;

    delete const_arpa;
    delete carpa_lm_to_subtract_fst;

    KALDI_LOG << "Overall, succeeded for " << num_done
              << " lattices, failed for " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
