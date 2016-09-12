// rnnlmbin/rnnlm-get-egs.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"

using std::string;
using std::ifstream;
using std::vector;

using namespace kaldi;
using namespace nnet3;

vector<string> SplitByWhiteSpace(const string &line) {
  std::stringstream ss(line);
  vector<string> ans;
  string word;
  while (ss >> word) {
    ans.push_back(word);
  }
  return ans;
}

unordered_map<string, int> ReadWordlist(string filename) {
  unordered_map<string, int> ans;
  ifstream ifile(filename.c_str());
  string word;
  int id;

  while (ifile >> word >> id) {
    ans[word] = id;
  }
  return ans;
}

void GenerateEgs(string text_file, const unordered_map<string, int>& wlist_in,
                 const unordered_map<string, int>& wlist_out, int history_size,
                 NnetExampleWriter* example_writer) {

  ifstream ifile(text_file.c_str());
  string line;
  int cur_line = 0;
  while (getline(ifile, line)) {
    if (cur_line % 1000 == 0) {
      cout << "processing line " << cur_line << endl;
    }
    vector<string> words = SplitByWhiteSpace(line);

    vector<int> word_ids_in;
    vector<int> word_ids_out;
    int input_dim = wlist_in.size();
    int output_dim = wlist_out.size();

    for (int i = 0; i < words.size(); i++) {
      int id_in = 1;
      int id_out = 1; // TODO(hxu) assuming OOS is always 1 in wordlists
      {
        unordered_map<string, int>::const_iterator iter = wlist_in.find(words[i]);
        if (iter != wlist_in.end()) {
          id_in = iter->second;
        }
      }
      word_ids_in.push_back(id_in);

      {
        unordered_map<string, int>::const_iterator iter = wlist_out.find(words[i]);
        if (iter != wlist_out.end()) {
          id_out = iter->second;
        }
      }
      word_ids_out.push_back(id_out);
    }

    for (int i = 1; i < word_ids_out.size(); i++) {
      // generate eg that predict word[i]
      vector<int> history(history_size, -1); // -1 means absence of word
      for (int j = 0; j < history_size; j++) {
        if (i - 1 - j < 0) {
          break;
        }
        history[history_size - 1 - j] = word_ids_in[i - 1 - j];
      }

      SparseMatrix<BaseFloat> input_frames(history_size, input_dim);
      for (int j = 0; j < history_size; j++) {
        vector<std::pair<MatrixIndexT, BaseFloat> > pairs;
        if (history[j] != -1) {
          pairs.push_back(std::make_pair(history[j], 1.0));
        }
        SparseVector<BaseFloat> v(input_dim, pairs);
        input_frames.SetRow(j, v);
      }

//      {
//        vector<std::pair<MatrixIndexT, BaseFloat> > pairs;
//        pairs.push_back(std::make_pair(word_ids_in[i], 1.0));
//        SparseVector<BaseFloat> v(input_dim, pairs);
//        input_frames.SetRow(history_size, v);
//      }

      NnetExample eg;
      eg.io.push_back(NnetIo("input", -history_size + 1, input_frames));
      Posterior posterior;
      vector<std::pair<int32, BaseFloat> > p;
      p.push_back(std::make_pair(word_ids_out[i], 1.0));
      posterior.push_back(p);
      eg.io.push_back(NnetIo("output", output_dim, 0, posterior));

      std::ostringstream os;
      os << "line-" << cur_line << "-" << i;

      std::string key = os.str(); // key is <line-num>-<word-num>
      example_writer->Write(key, eg);
    }
    cur_line++;
  }
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  const char *usage = "Get egs for rnnlm\n"
    "e.g. rnnlm-get-egs text-file wordlist.in wordlist.out ark,t:egs\n"
    ;
  ParseOptions po(usage);

  int history = 20;
  po.Register("history", &history, "number of history words");

  po.Read(argc, argv);

  if (po.NumArgs() != 4) {
    po.PrintUsage();
    exit(1);
  }

  int i = 1;
  string text_file = po.GetArg(i++);
  string in_wlist_file = po.GetArg(i++);
  string out_wlist_file = po.GetArg(i++);
  string examples_wspecifier = po.GetArg(i++);

  NnetExampleWriter example_writer(examples_wspecifier);
  unordered_map<string, int> in_wlist = ReadWordlist(in_wlist_file);
  unordered_map<string, int> out_wlist = ReadWordlist(out_wlist_file);

  GenerateEgs(text_file, in_wlist, out_wlist, history, &example_writer);

  return 0;
}
