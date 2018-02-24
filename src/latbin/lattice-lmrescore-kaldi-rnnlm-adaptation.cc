// latbin/lattice-lmrescore-kaldi-rnnlm-adaptation.cc

// Copyright 2017 Johns Hopkins University (author: Daniel Povey)
//           2017 Hainan Xu
//           2017 Yiming Wang

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
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "util/common-utils.h"
#include "nnet3/nnet-utils.h"
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

// void ReadUnigramCountByConversation(string filename,
//                                unordered_map<string, map<int, double> > *counts,
//                                unordered_map<string, double> *total_counts) {
// 
//   unordered_map<string, map<int, double> > &m = *counts;
//   KALDI_ASSERT(m.size() == 0);
//   ifstream ifile(filename.c_str());
//   string convo_id;
//   int32 word_id;
//   double soft_count;
// 
//   unordered_map<string, double> &convo_to_sum = *total_counts;
//   KALDI_ASSERT(convo_to_sum.size() == 0);
// 
//   while (ifile >> convo_id >> word_id >> soft_count) {
//     m[convo_id][word_id] = soft_count;
//     convo_to_sum[convo_id] += soft_count;
//   }
// 
//   // now normalize everything; should add smoothing later
// //  for (unordered_map<string, map<int, double> >::iterator i = m.begin();
// //                                                          i != m.end(); i++) {
// //    map<int, double> &this_map = i->second;
// //    double sum = convo_to_sum[i->first];
// //    for (map<int, double>::iterator j = this_map.begin();
// //                                      j!= this_map.end(); j++) {
// //      j->second /= sum;
// //    }
// //  }
// 
//   ifile.close();
// }
// 

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

    const char *usage =
        "Rescores lattice with kaldi-rnnlm. This script is called from \n"
        "scripts/rnnlm/lmrescore_rnnlm_lat.sh. An example for rescoring \n"
        "lattices is at egs/swbd/s5/local/rnnlm/run_rescoring.sh \n"
        "\n"
        "Usage: lattice-lmrescore-kaldi-rnnlm [options] \\\n"
        "             <embedding-file> <raw-rnnlm-rxfilename> \\\n"
        "             <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-lmrescore-kaldi-rnnlm --lm-scale=-1.0 \\\n"
        "              word_embedding.mat \\\n"
        "              --bos-symbol=1 --eos-symbol=2 \\\n"
        "              final.raw ark:in.lats ark:out.lats\n";

    ParseOptions po(usage);
    rnnlm::RnnlmComputeStateComputationOptions opts;

    kaldi::BaseFloat min_post = 0.0001;
    int32 max_ngram_order = 3;
    kaldi::BaseFloat acoustic_scale = 0.1, lm_scale = 1.0;

    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "costs");
    po.Register("max-ngram-order", &max_ngram_order,
        "If positive, allow RNNLM histories longer than this to be identified "
        "with each other for rescoring purposes (an approximation that "
        "saves time and reduces output lattice size).");
    po.Register("min-post", &min_post,
                "Arc posteriors below this value will be pruned away");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    if (opts.bos_index == -1 || opts.eos_index == -1) {
      KALDI_ERR << "You must set --bos-symbol and --eos-symbol options";
    }

    std::string lats_rspecifier, word_embedding_rxfilename,
        word_symbols_rxfilename, rnnlm_rxfilename, lats_wspecifier;
    std::string utt_to_convo_file, unigram_file;

    word_embedding_rxfilename = po.GetArg(1);
    rnnlm_rxfilename = po.GetArg(2);
    lats_rspecifier = po.GetArg(3);
    lats_wspecifier = po.GetArg(4);
    utt_to_convo_file = po.GetArg(5);
    unigram_file = po.GetArg(6);

    map<string, string> utt2convo;
    ReadUttToConvo(utt_to_convo_file, utt2convo);

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

    KALDI_ASSERT(IsSimpleNnet(rnnlm));

    CuMatrix<BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);

    std::vector<double> original_unigram(word_embedding_mat.NumRows(), 0.0);  // number of words
    ReadUnigram(unigram_file, &original_unigram);

    const rnnlm::RnnlmComputeStateInfo info(opts, rnnlm, word_embedding_mat);

    // Reads and writes as compact lattice.
    int32 n_done = 0, n_fail = 0;

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

        kaldi::ArcPosteriorComputer computer(clat, min_post);

        computer.OutputPosteriors(
                                  &(per_convo_counts[convo_id]),
                                  &(per_utt_counts[utt_id]),
                                  &(per_convo_sums[convo_id]),
                                  &(per_utt_sums[utt_id]));
      }

      clat_reader.Close();
    }

    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {

      std::string key = compact_lattice_reader.Key();
      std::string convo_id = utt2convo[key];
      KALDI_ASSERT(convo_id != "");

      map<int, double> unigram = per_convo_counts[convo_id];
      for (map<int, double>::iterator iter = per_utt_counts[key].begin();
                                      iter != per_utt_counts[key].end(); iter++) {
        unigram[iter->first] =
                 (unigram[iter->first] - iter->second); // / per_utt_sums[key];
//        debug_sum += unigram[iter->first];
      }

      double sum = per_convo_sums[convo_id] - per_utt_sums[key];
      double debug_sum = 0.0;
      for (map<int, double>::iterator iter = unigram.begin();
                                      iter != unigram.end(); iter++) {
        iter->second /= sum;
        debug_sum += iter->second;
      }

      KALDI_ASSERT(ApproxEqual(debug_sum, 1.0));

      rnnlm::KaldiRnnlmDeterministicFst
               rnnlm_fst(max_ngram_order, info, unigram, original_unigram);
      CompactLattice &clat = compact_lattice_reader.Value();

      if (lm_scale != 0.0) {
        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        fst::ScaleLattice(fst::GraphLatticeScale(1.0 / lm_scale), &clat);
        ArcSort(&clat, fst::OLabelCompare<CompactLatticeArc>());

        // Wraps the rnnlm into FST. We re-create it for each lattice to prevent
        // memory usage increasing with time.

        // Composes lattice with language model.
        CompactLattice composed_clat;
        ComposeCompactLatticeDeterministic(clat, &rnnlm_fst, &composed_clat);

        // Determinizes the composed lattice.
        Lattice composed_lat;
        ConvertLattice(composed_clat, &composed_lat);
        Invert(&composed_lat);
        CompactLattice determinized_clat;
        DeterminizeLattice(composed_lat, &determinized_clat);
        fst::ScaleLattice(fst::GraphLatticeScale(lm_scale), &determinized_clat);
        if (determinized_clat.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty lattice for utterance " << key
              << " (incompatible LM?)";
          n_fail++;
        } else {
          compact_lattice_writer.Write(key, determinized_clat);
          n_done++;
        }
      } else {
        // Zero scale so nothing to do.
        n_done++;
        compact_lattice_writer.Write(key, clat);
      }
    }

    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
