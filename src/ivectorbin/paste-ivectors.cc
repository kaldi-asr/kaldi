// featbin/paste-ivectors.cc

// Copyright 2012 Korbinian Riedhammer
//           2013 Brno University of Technology (Author: Karel Vesely)
//           2013 Johns Hopkins University (Author: Daniel Povey)
//           2018 Ewald Enzinger

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
#include "matrix/kaldi-vector.h"

namespace kaldi {

// returns true if successfully appended.
bool AppendVectors(const std::vector<Vector<BaseFloat> > &in,
                 std::string utt,
                 Vector<BaseFloat> *out) {
  int32 tot_dim = in[0].Dim();
  for (int32 i = 1; i < in.size(); i++) {
    int32 dim = in[i].Dim();
    tot_dim += dim;
  }
  out->Resize(tot_dim, kUndefined);
  int32 dim_offset = 0;
  for (int32 i = 0; i < in.size(); i++) {
    int32 this_dim = in[i].Dim();
    out->Range(dim_offset, this_dim).CopyFromVec(
        in[i].Range(0, this_dim));
    dim_offset += this_dim;
  }
  return true;
}


}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Paste i-vector files; think of the unix command 'paste'.\n"
        "Usage: paste-ivectors <in-rspecifier1> <in-rspecifier2> [<in-rspecifier3> ...] <out-wspecifier>\n"
        " or: paste-ivectors <in-rxfilename1> <in-rxfilename2> [<in-rxfilename3> ...] <out-wxfilename>\n"
        " e.g. paste-ivectors feat1/ivector.1.ark:32 feat2/ivector.1.ark:32 - |\n"
        "See also: paste-feats, copy-vector\n";

    ParseOptions po(usage);

    bool binary = true;
    po.Register("binary", &binary, "If true, output files in binary "
                "(only relevant for single-file operation, i.e. no tables)");

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL)
        != kNoRspecifier) {
      // We're operating on tables, e.g. archives.

      // Last argument is output
      string wspecifier = po.GetArg(po.NumArgs());
      BaseFloatVectorWriter vector_writer(wspecifier);

      // First input is sequential
      string rspecifier1 = po.GetArg(1);
      SequentialBaseFloatVectorReader input1(rspecifier1);

      // Assemble vector of other input readers (with random-access)
      vector<RandomAccessBaseFloatVectorReader *> input;
      for (int32 i = 2; i < po.NumArgs(); i++) {
        string rspecifier = po.GetArg(i);
        RandomAccessBaseFloatVectorReader *rd = new RandomAccessBaseFloatVectorReader(rspecifier);
        input.push_back(rd);
      }

      int32 num_done = 0, num_err = 0;

      // Main loop
      for (; !input1.Done(); input1.Next()) {
        string utt = input1.Key();
        KALDI_VLOG(2) << "Merging i-vectors for utterance " << utt;

        // Collect i-vectors from streams to vector 'ivectors'
        vector<Vector<BaseFloat> > ivectors(po.NumArgs() - 1);
        ivectors[0] = input1.Value();
        int32 i;
        for (i = 0; i < static_cast<int32>(input.size()); i++) {
          if (input[i]->HasKey(utt)) {
            ivectors[i + 1] = input[i]->Value(utt);
          } else {
            KALDI_WARN << "Missing utt " << utt << " from input "
                       << po.GetArg(i+2);
            num_err++;
            break;
          }
        }
        if (i != static_cast<int32>(input.size()))
          continue;
        Vector<BaseFloat> output;
        if (!AppendVectors(ivectors, utt, &output)) {
          num_err++;
          continue; // it will have printed a warning.
        }
        vector_writer.Write(utt, output);
        num_done++;
      }

      for (int32 i=0; i < input.size(); i++)
        delete input[i];
      input.clear();

      KALDI_LOG << "Done " << num_done << " utts, errors on "
                << num_err;

      return (num_done == 0 ? -1 : 0);
    } else {
      // We're operating on rxfilenames|wxfilenames, most likely files.
      std::vector<Vector<BaseFloat> > ivectors(po.NumArgs() - 1);
      for (int32 i = 1; i < po.NumArgs(); i++)
        ReadKaldiObject(po.GetArg(i), &(ivectors[i-1]));
      Vector<BaseFloat> output;
      if (!AppendVectors(ivectors, "", &output))
        return 1; // it will have printed a warning.
      std::string output_wxfilename = po.GetArg(po.NumArgs());
      WriteKaldiObject(output, output_wxfilename, binary);
      KALDI_LOG << "Wrote appended i-vectors to " << output_wxfilename;
      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

/*
  Testing:

cat <<EOF >1.mat
[ 0 1 2 ]
EOF
cat <<EOF > 2.mat
 [ 3 4 5 ]
EOF
paste-ivectors --binary=false 1.mat 2.mat 3a.mat
cat <<EOF > 3b.mat
 [ 0 1 2 3 4 5 ]
EOF
cmp <(../bin/copy-vector 3b.mat -) <(../bin/copy-vector 3a.mat -) || echo 'Bad!'

paste-ivectors 'scp:echo foo 1.mat|' 'scp:echo foo 2.mat|' 'scp,t:echo foo 3a.mat|'
cmp <(../bin/copy-vector 3b.mat -) <(../bin/copy-vector 3a.mat -) || echo 'Bad!'

rm {1,2,3?}.mat
 */
