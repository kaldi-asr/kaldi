// gmmbin/gmm-align-compiled-plusphones.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/training-graph-compiler.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "decoder/decodable-sum.h"
#include "decoder/decodable-mapped.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

namespace kaldi {
// This creates a model indexed by (phone-index - 1).
// Note: the object DecodableAmDiagGmmUnmapped subtracts
// one from the index it's given, this is where the -1
// will happen in test time.

void CreatePhoneModel(const TransitionModel &trans_model,
                      const AmDiagGmm &am_gmm,
                      const Vector<BaseFloat> &transition_accs,
                      int32 max_num_gauss, // max #gauss for each phone.
                      AmDiagGmm *phone_am) {
  KALDI_LOG << "Creating phone-level model by clustering GMMs merged from context-dependent states";
  BaseFloat min_weight = 1.0e-05; // We assign this weight to transition-ids with no observations;
  // this ensures that we get a model for unseen phones.
  
  // The vector phone_weights is a list, indexed by phone-id,
  // of pairs of (index into am_gmm, weight).  We'll use this to
  // construct the GMMs for each phone.
  std::vector<std::map<int32, BaseFloat> > phone_weights;
  KALDI_ASSERT(transition_accs.Dim() == trans_model.NumTransitionIds()+1);
  // +1 because transition_accs[0] is empty; transition-ids are one-based.
  for (int32 tid = 1; tid < trans_model.NumTransitionIds(); tid++) {
    int32 phone = trans_model.TransitionIdToPhone(tid),
        pdf_id = trans_model.TransitionIdToPdf(tid);
    if (phone_weights.size() <= phone) phone_weights.resize(phone+1);
    if (phone_weights[phone].count(pdf_id) == 0)
      phone_weights[phone][pdf_id] = 0.0;
    BaseFloat max_weight =  std::max(min_weight, transition_accs(tid));
    phone_weights[phone][pdf_id] += max_weight;

  }
  int32 num_phones = trans_model.GetTopo().GetPhones().back(); // #phones, assuming
  // they start from 1.
  int32 dim = am_gmm.Dim();
  DiagGmm gmm(1, dim);
  { // give it valid values..  note: should never be accessed, but nice to avoid NaNs...
    Matrix<BaseFloat> inv_covars(1, dim);
    inv_covars.Set(1.0);
    gmm.SetInvVars(inv_covars);
    Vector<BaseFloat> weights(1);
    weights(0) = 1.0;
    gmm.SetWeights(weights);
  }
  phone_am->Init(gmm, num_phones);
  for (int32 phone = 1; phone < static_cast<int32>(phone_weights.size()); phone++) {
    if (phone_weights[phone].empty()) continue; // No GMM for this phone.  Presumably
    // not a valid phone.
    std::vector<std::pair<BaseFloat, const DiagGmm*> > gmm_vec;
    BaseFloat tot_weight = 0.0;
    for (std::map<int32, BaseFloat>::const_iterator iter = phone_weights[phone].begin();
         iter != phone_weights[phone].end();
         ++iter) {
      int32 pdf_id = iter->first;
      BaseFloat weight = iter->second;
      std::pair<BaseFloat, const DiagGmm*> pr(weight, &(am_gmm.GetPdf(pdf_id)));
      gmm_vec.push_back(pr);
      tot_weight += weight;
    }
    for (size_t i = 0; i < gmm_vec.size(); i++)
      gmm_vec[i].first *= (1.0 / tot_weight);
    DiagGmm gmm(gmm_vec); // Initializer creates merged GMM.
    if (gmm.NumGauss() > max_num_gauss) {
      ClusterKMeansOptions cfg;
      cfg.verbose = false;
      gmm.MergeKmeans(max_num_gauss, cfg);
    }
    phone_am->GetPdf(phone-1).CopyFromDiagGmm(gmm); // Set this phone's GMM to the specified value.
  }
  KALDI_LOG << "Done.";
}
                      
void CreatePhoneMap(const TransitionModel &trans_model,
                    std::vector<int32> *phone_map) {
  // Set up map from transition-id to phone.
  phone_map->resize(trans_model.NumTransitionIds() + 1);
  // transition-ids are one based: there's nothing in index zero.
  (*phone_map)[0] = 0;
  for (int32 i = 1; i <= trans_model.NumTransitionIds(); i++)
    (*phone_map)[i] = trans_model.TransitionIdToPhone(i);
}

}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given [GMM-based] models, but adds in likelihoods of simple per-phone GMMs\n"
        "with alpha*per-phone-like + (1-alpha)*model-like.  This gives more consistent alignments.\n"
        "Per-phone models are obtained by K-means on weighted model states, using the transition-accs\n"
        "to get weights. (e.g. use the first line of text format of normal accs).\n"
        "Note: this program actually isn't that useful.  We keep it mainly as an example\n"
        "of how to write a decoder with interpolated likelihoods.\n"

        "Usage:   gmm-align-compiled-plusphones [options] transition-accs-in model-in graphs-rspecifier feature-rspecifier alignments-wspecifier\n"
        "e.g.: \n"
        " gmm-align-compiled-plusphones --alpha=0.2 --acoustic-scale=0.1 \\\n"
        "    1.acc 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali\n"
        "or:\n"
        " compile-train-graphs tree 1.mdl lex.fst ark:train.tra b, ark:- | \\\n"
        "   gmm-align-compiled-plusphones 1.acc 1.mdl ark:- scp:train.scp t, ark:1.ali\n";

    ParseOptions po(usage);
    bool binary = true;
    BaseFloat alpha = 0.2;
    BaseFloat beam = 200.0;
    BaseFloat retry_beam = 0.0;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    int32 max_gauss = 10;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("alpha", &alpha, "Weight on simple phone model (rest of weight goes to normal model)");
    po.Register("max-gauss", &max_gauss, "Maximum number of Gaussians in any of the simple phone models.");
    po.Register("beam", &beam, "Decoding beam");
    po.Register("retry-beam", &retry_beam, "Decoding beam for second try at alignment");
    po.Register("transition-scale", &transition_scale, "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("self-loop-scale", &self_loop_scale, "Scale of self-loop versus non-self-loop log probs [relative to acoustics]");
    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }
    if (retry_beam != 0 && retry_beam <= beam)
      KALDI_WARN << "Beams do not make sense: beam " << beam
                 << ", retry-beam " << retry_beam;
    
    FasterDecoderOptions decode_opts;
    decode_opts.beam = beam;  // Don't set the other options.

    std::string trans_accs_in_filename = po.GetArg(1),
        model_in_filename = po.GetArg(2),
        fst_rspecifier = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        alignment_wspecifier = po.GetArg(5),
        scores_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Vector<BaseFloat> trans_accs; // Transition accs.
    {
      bool binary;
      Input ki(trans_accs_in_filename, &binary);
      trans_accs.Read(ki.Stream(), binary);
      KALDI_ASSERT(trans_accs.Dim() == trans_model.NumTransitionIds() + 1)
    }

    AmDiagGmm phone_am;
    CreatePhoneModel(trans_model, am_gmm, trans_accs, max_gauss, &phone_am);
    std::vector<int32> tid_to_phone_map;
    CreatePhoneMap(trans_model, &tid_to_phone_map);

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);
    BaseFloatWriter scores_writer(scores_wspecifier);

    int num_success = 0, num_no_feat = 0, num_other_error = 0;
    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      if (!feature_reader.HasKey(key)) {
        num_no_feat++;
        KALDI_WARN << "No features for utterance " << key;
      } else {
        const Matrix<BaseFloat> &features = feature_reader.Value(key);
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << key;
          num_other_error++;
          continue;
        }
        if (decode_fst.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty decoding graph for " << key;
          num_other_error++;
          continue;
        }

        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &decode_fst);
        }

        // SimpleDecoder decoder(decode_fst, beam);
        FasterDecoder decoder(decode_fst, decode_opts);
        // makes it a bit faster: 37 sec -> 26 sec on 1000 RM utterances @ beam 200.

        DecodableAmDiagGmm gmm_decodable(am_gmm, trans_model, features);

        BaseFloat log_sum_exp_prune = 0.0;
        DecodableAmDiagGmmUnmapped phone_decodable(phone_am, features, log_sum_exp_prune);

        DecodableMapped phone_decodable_mapped(tid_to_phone_map,
                                               &phone_decodable);
        // indexed by transition-ids.
        DecodableSum sum_decodable(&gmm_decodable, acoustic_scale * (1.0-alpha),
                                   &phone_decodable_mapped, acoustic_scale * alpha);
        
        decoder.Decode(&sum_decodable);

        VectorFst<LatticeArc> decoded;  // linear FST.
        bool ans = decoder.ReachedFinal() // consider only final states.
            && decoder.GetBestPath(&decoded);  
        if (!ans && retry_beam != 0.0) {
          KALDI_WARN << "Retrying utterance " << key << " with beam " << retry_beam;
          decode_opts.beam = retry_beam;
          decoder.SetOptions(decode_opts);
          decoder.Decode(&sum_decodable);
          ans = decoder.ReachedFinal() // consider only final states.
              && decoder.GetBestPath(&decoded);  
          decode_opts.beam = beam;
          decoder.SetOptions(decode_opts);
        }
        if (ans) {
          std::vector<int32> alignment;
          std::vector<int32> words;
          LatticeWeight weight;
          frame_count += features.NumRows();

          GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
          BaseFloat like = -(weight.Value1()+weight.Value2()) / acoustic_scale;
          tot_like += like;
          if (scores_writer.IsOpen())
            scores_writer.Write(key, -(weight.Value1()+weight.Value2()));
          alignment_writer.Write(key, alignment);
          num_success ++;
          if (num_success % 50  == 0) {
            KALDI_LOG << "Processed " << num_success << " utterances, "
                      << "log-like per frame for " << key << " is "
                      << (like / features.NumRows()) << " over "
                      << features.NumRows() << " frames.";
          }
        } else {
          KALDI_WARN << "Did not successfully decode file " << key << ", len = "
                     << (features.NumRows());
          num_other_error++;
        }
      }
    }
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count<< " frames.";
    KALDI_LOG << "Done " << num_success << ", could not find features for "
              << num_no_feat << ", other errors on " << num_other_error;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


