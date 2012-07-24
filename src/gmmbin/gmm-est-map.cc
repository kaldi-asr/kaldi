// gmmbin/gmm-est-map.cc

// Copyright 2012  Cisco Systems (author: Neha Agrawal)

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

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/map-diag-gmm-accs.h"

int main(int argc, char *argv[]) {
    try {
        typedef kaldi::int32 int32;
        using namespace kaldi;
        const char *usage =
            "Compute MAP estimates per-utterance (default) or per-speaker for "
            "the supplied set of speakers (spk2utt option).  \n"
            "Usage: gmm-est-map  [options] <model-in> <feature-rspecifier> "
            "<posteriors-rspecifier> <map-am-wspecifier>\n";

        ParseOptions po(usage);
        string spk2utt_rspecifier;
        BaseFloat tau = 0.0;
        bool binary = true;
        po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                    "utterance-list map");
        po.Register("binary", &binary, "Write output in binary mode");
        po.Register("tau",&tau,"MAP tau-value");


        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }

        string model_filename = po.GetArg(1),
               feature_rspecifier = po.GetArg(2),
               posteriors_rspecifier = po.GetArg(3),
               map_am_wspecifier = po.GetArg(4);

        RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
        MapAmDiagGmmWriter map_am_writer(map_am_wspecifier);

        AmDiagGmm am_gmm;
        TransitionModel trans_model;
        {
            bool binary;
            Input is(model_filename, &binary);
            trans_model.Read(is.Stream(), binary);
            am_gmm.Read(is.Stream(), binary);
        }

        MapDiagGmmAccs map_accs;
        map_accs.Init(am_gmm.NumPdfs());

        double tot_like = 0.0, tot_t = 0;
        int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;

        if (spk2utt_rspecifier != "") {  // per-speaker adaptation
            SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
            RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
            for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
                string spk = spk2utt_reader.Key();
                map_accs.SetZero();
                const vector<string> &uttlist = spk2utt_reader.Value();

                // foreach speaker estimate MAP means
                for ( vector<string>::const_iterator utt_itr = uttlist.begin(),
                        itr_end = uttlist.end(); utt_itr != itr_end; ++utt_itr ) { 
                    if ( !feature_reader.HasKey(*utt_itr) ) {
                        KALDI_WARN << "Did not find features for utterance " 
                            << *utt_itr;
                        continue;
                    }
                    if ( !posteriors_reader.HasKey(*utt_itr) ) {
                        KALDI_WARN << "Did not find posteriors for utterance "
                            << *utt_itr;
                        num_no_posterior++;
                        continue;
                    }
                    const Matrix<BaseFloat> &feats = feature_reader.Value(*utt_itr);
                    const Posterior &posterior = posteriors_reader.Value(*utt_itr);
                    if ( posterior.size() != feats.NumRows() ) {
                        KALDI_WARN << "Posteriors has wrong size " << (posterior.size())
                            << " vs. " << (feats.NumRows());
                        num_other_error++;
                        continue;
                    }

                    BaseFloat file_like = 0.0, file_t = 0.0;
                    for ( size_t i = 0; i < posterior.size(); i++ ) {
                        for ( size_t j = 0; j < posterior[i].size(); j++ ) {
                            //Get the pdf_id for the frame
                            int32 pdf_id = 
                                trans_model.TransitionIdToPdf(posterior[i][j].first); 
                            //Get the weight for the frame
                            BaseFloat prob = posterior[i][j].second; 
                            //Accumulate the frame
                            file_like += map_accs.AccumulateForGmm(pdf_id, 
                                                                   feats.Row(i),
                                                                   am_gmm, prob);
                            file_t += prob;
                        }
                    }

                    KALDI_VLOG(2) << "Average like for this file is " 
                        << (file_like/file_t) << " over " << file_t << " frames.";

                    tot_like += file_like;
                    tot_t += file_t;
                    num_done++;

                    if (num_done % 10 == 0)
                        KALDI_VLOG(1) << "Avg like per frame so far is "
                            << (tot_like / tot_t);

                }  // end looping over all utterances of the current speaker

                AmDiagGmm map_am_gmm;
                map_accs.Update(am_gmm,tau,map_am_gmm); // Calculating MAP adapted AM
                
                // Writing AM for each speaker in a table
                map_am_writer.Write(spk,map_am_gmm);

            }  // end looping over speakers
        } else {  // per-utterance adaptation
            SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
            for ( ; !feature_reader.Done(); feature_reader.Next() ) {
                string key = feature_reader.Key();
                map_accs.SetZero();
                if ( !posteriors_reader.HasKey(key) ) {
                    KALDI_WARN << "Did not find aligned transcription for utterance "
                        << key;
                    num_no_posterior++;
                    continue;
                }
                const Matrix<BaseFloat> &feats = feature_reader.Value();
                const Posterior &posterior = posteriors_reader.Value(key);

                if ( posterior.size() != feats.NumRows() ) {
                    KALDI_WARN << "Posteriors has wrong size " << (posterior.size())
                        << " vs. " << (feats.NumRows());
                    num_other_error++;
                    continue;
                }

                num_done++;
                BaseFloat file_like = 0.0, file_t = 0.0;
                for ( size_t i = 0; i < posterior.size(); i++ ) {
                    for ( size_t j = 0; j < posterior[i].size(); j++ ) {
                        int32 pdf_id = 
                            trans_model.TransitionIdToPdf(posterior[i][j].first);
                        BaseFloat prob = posterior[i][j].second;
                        file_like += 
                            map_accs.AccumulateForGmm(pdf_id, feats.Row(i),
                                                      am_gmm, prob);
                        file_t += prob;
                    }
                }
                KALDI_VLOG(2) << "Average like for this file is " 
                    << (file_like/file_t) << " over " << file_t << " frames.";
                tot_like += file_like;
                tot_t += file_t;
                if ( num_done % 10 == 0 )
                    KALDI_VLOG(1) << "Avg like per frame so far is " 
                        << (tot_like / tot_t);

                AmDiagGmm map_am_gmm;
                map_accs.Update(am_gmm,tau,map_am_gmm);
                map_am_writer.Write(feature_reader.Key(),map_am_gmm);
            }
        }

        KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";
        KALDI_LOG << "Overall acoustic likelihood was " << (tot_like/tot_t)
              << " over " << tot_t << " frames.";
        return 0;

    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
}


