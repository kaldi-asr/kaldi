// rnnlmbin/rnnlm-eval.cc

// Copyright 2016  Hainan Xu

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

#include <fstream>
#include <sstream>

#include "rnnlm/rnnlm-utils.h"

using std::string;
using std::ifstream;
using std::ofstream;
using std::vector;

using namespace kaldi;
using namespace nnet3;

void RnnlmEval(NnetComputeProb &computer, // can't make this const it seems
               const unordered_map<string, int>& wlist_in,
               const unordered_map<string, int>& wlist_out,
               BaseFloat oos_cost, string in_file, string out_file) {
  ifstream ifile(in_file.c_str());
  ofstream ofile(out_file.c_str());
  string line;
//  int cur_line = 0;

  while (getline(ifile, line)) {
//    BaseFloat obj_to_add = 0.0;
    int num_oovs = 0;
    vector<string> words = SplitByWhiteSpace(line);

    vector<int> word_ids_in;
    vector<int> word_ids_out;
    int input_dim = wlist_in.size();
    int output_dim = wlist_out.size();

    for (int i = 0; i < words.size(); i++) {
      int id_in = OOS_ID;
      int id_out = OOS_ID;
     
      if (i != words.size() - 1) {
        unordered_map<string, int>::const_iterator iter = wlist_in.find(words[i]);
        if (iter != wlist_in.end()) {
          id_in = iter->second;
        }
        word_ids_in.push_back(id_in);
      }

      if (i != 0) {
        unordered_map<string, int>::const_iterator iter = wlist_out.find(words[i]);
        if (iter != wlist_out.end()) {
          id_out = iter->second;
        }
        word_ids_out.push_back(id_out);
        if (id_out == OOS_ID) {
  //        obj_to_add += oos_cost;
          num_oovs ++;
//          ofile << "# " << i << " " << words[i] << endl;
        }
      }

    }

    NnetExample egs = GetEgsFromSent(word_ids_in, input_dim,
                                     word_ids_out, output_dim);
    computer.Compute(egs);
    const SimpleObjectiveInfo *info = computer.GetObjective("output");
    ofile << info->tot_objective + oos_cost * num_oovs << endl;
    computer.Reset();
  }
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  const char *usage = "Get egs for rnnlm\n"
    "e.g. rnnlm-get-egs rnnlm-model wordlist.in wordlist.out text-file logprob-file\n";

  NnetComputeProbOptions opts;
  ParseOptions po(usage);
  int history = 20;
  int num_words = -1;
  po.Register("history", &history, "number of history words");
  po.Register("num-words", &num_words, "number of total words");

  opts.Register(&po);

  po.Read(argc, argv);

  if (po.NumArgs() != 5) {
    po.PrintUsage();
    exit(1);
  }

  int i = 1;
  string rnn_file = po.GetArg(i++);
  string in_wlist_file = po.GetArg(i++);
  string out_wlist_file = po.GetArg(i++);
  string text_file = po.GetArg(i++);
  string out_file = po.GetArg(i++);

  Nnet nnet;
  ReadKaldiObject(rnn_file, &nnet);

  NnetComputeProb prob_computer(opts, nnet);

  unordered_map<string, int> in_wlist = ReadWordlist(in_wlist_file);
  unordered_map<string, int> out_wlist = ReadWordlist(out_wlist_file);

  cout << "sizes are " << in_wlist.size() << " " << out_wlist.size() << endl;

  if (num_words != -1) {
    KALDI_ASSERT(num_words >= out_wlist.size());
  }

  BaseFloat oos_cost = log(1.0 / (num_words - out_wlist.size() + 1));
  if (num_words == -1) {
    oos_cost = 0;
  }

  RnnlmEval(prob_computer, in_wlist, out_wlist, oos_cost, text_file, out_file);

  return 0;
}
