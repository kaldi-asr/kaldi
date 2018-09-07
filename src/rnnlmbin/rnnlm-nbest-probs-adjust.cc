// rnnlmbin/rnnlm-nbest-probs-adjust.cc

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
#include <fstream>
#include <sstream>

using std::ifstream;
using std::map;
using std::unordered_map;

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

// first set *key to the first field, and then break the remaining line into a
// vector of integers
void GetNumbersFromLine(std::string line, std::string *key,
                        std::vector<int32> *v) {
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

    std::string use_gpu = "no";
    bool batchnorm_test_mode = true, dropout_test_mode = true;
    bool two_speaker_mode = true;
    double correction_weight = 1.0;

    ParseOptions po(usage);
    rnnlm::RnnlmComputeStateComputationOptions opts;
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
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string rnnlm_rxfilename = po.GetArg(1),
                word_embedding_rxfilename = po.GetArg(2),
                text_filename = po.GetArg(3),
                utt_to_convo_file = po.GetArg(4),
                unigram_file = po.GetArg(5);

    map<string, string> utt2convo;
    ReadUttToConvo(utt_to_convo_file, utt2convo);

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().AllowMultithreading();
#endif

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

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

    // number of words
    std::vector<double> original_unigram(word_embedding_mat.NumRows(), 0.0);
    ReadUnigram(unigram_file, &original_unigram);

    const rnnlm::RnnlmComputeStateInfo info(opts, rnnlm, word_embedding_mat);

    std::map<string, map<int, double> > per_convo_counts;
    std::map<string, map<int, double> > per_utt_counts;
    std::map<string, double> per_convo_sums;
    std::map<string, double> per_utt_sums;

    int32 total_words = 0;
    {
      std::ifstream ifile(text_filename.c_str());
      std::string line;
      std::string utt_id;
      while (getline(ifile, line)) {
        std::vector<int> v;
        GetNumbersFromLine(line, &utt_id, &v);
        std::string convo_id = utt2convo[utt_id];
        if (two_speaker_mode) {
          std::string convo_id_2spk = std::string(convo_id.begin(),
                                                  convo_id.end() - 2);
          convo_id = convo_id_2spk;
        }

        per_convo_sums[convo_id] += v.size();
        per_utt_sums[utt_id] += v.size();
        total_words += v.size();
        for (int32 i = 0; i < v.size(); i++) {
          int32 word_id = v[i];
          per_convo_counts[convo_id][word_id]++;
          per_utt_counts[utt_id][word_id]++;
        }
      }
    }

    {
      std::ifstream ifile(text_filename.c_str());
      std::string line;
      std::string utt_id;
      double total_probs = 0.0;
      while (getline(ifile, line)) {
        std::vector<int> v;
        GetNumbersFromLine(line, &utt_id, &v);
        std::string convo_id = utt2convo[utt_id];
        // Use convs of both speakers
        if (two_speaker_mode) {
          std::string convo_id_2spk = std::string(convo_id.begin(),
                                                  convo_id.end() - 2);
          convo_id = convo_id_2spk;
        }
        map<int, double> unigram = per_convo_counts[convo_id];
        for (map<int, double>::iterator iter = per_utt_counts[utt_id].begin();
                                        iter != per_utt_counts[utt_id].end();
                                        iter++) {
          unigram[iter->first] = unigram[iter->first] - iter->second;
        }
        double sum = per_convo_sums[convo_id] - per_utt_sums[utt_id];

        double debug_sum = 0.0;
        for (map<int, double>::iterator iter = unigram.begin();
                                        iter != unigram.end(); iter++) {
          iter->second /= sum;
          debug_sum += iter->second;
        }
        KALDI_ASSERT(ApproxEqual(debug_sum, 1.0));

        RnnlmComputeState rnnlm_compute_state(info, opts.bos_index);
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

                const double C = 0.0000001;
                double correction = (u + C) /
                                    (original_unigram[iter->first] + C);
                // smoothing by a background unigram distribution
                // original_unigram
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
      double ppl = Exp(-total_probs/total_words);
      std::cout << "Perplexity: " << ppl << std::endl;
    }

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
