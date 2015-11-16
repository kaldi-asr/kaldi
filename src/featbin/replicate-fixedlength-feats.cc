// featbin/replicate-fixedlength-feats.cc

// Copyright 2015  Hakan Erdogan

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

#include <sstream>
#include <algorithm>
#include <iterator>
#include <utility>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace std;

        const char *usage =
            "Replicates fixed number of features per utterance as many times as given in the second argument per utterance. "
            "Second argument is an scp/ark file which has the number of repetitions (as int32) per utterance. "
            "Second argument can be obtained using feat-len command on a feats.scp/ark file."
            "\n"
            "Usage: replicate-fixedlength-feats <rspecifier> <featlen-rspecifier> <out-wspecifier>\n"
            "  e.g. replicate-fixedlength-feats ark:- ark:- ark:-\n";

        ParseOptions po(usage);

        int32 n = 1, offset = 0;

        po.Register("n", &n, "Take n features in the first file from the start or from the offset");
        po.Register("offset", &offset, "Start with the feature at this offset, "
                    "then take n features.");

        KALDI_ASSERT(n > 0);
        KALDI_ASSERT(offset >= 0);

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        string rspecifier = po.GetArg(1);
        string featlen_rspecifier = po.GetArg(2);
        string wspecifier = po.GetArg(3);

        SequentialBaseFloatMatrixReader feat_reader(rspecifier);
        SequentialInt32Reader featlen_reader(featlen_rspecifier);

        BaseFloatMatrixWriter feat_writer(wspecifier);

        int32 num_done = 0, num_err = 0;
        int64 frames_in = 0, frames_out = 0;

        // process all keys
        for (; !feat_reader.Done() && !featlen_reader.Done(); feat_reader.Next(), featlen_reader.Next()) {
            int32 num_frames = featlen_reader.Value();
            std::string utt = feat_reader.Key();
            std::string utt_len = featlen_reader.Key();
            const Matrix<BaseFloat> feats(feat_reader.Value());

            if ( utt != utt_len) {
                KALDI_WARN << "Mismatched utterance Id " << utt << ", " << utt_len
                           << "ignoring second one, using first one!!!";
            }
            if (feats.NumRows() < offset+n) {
                KALDI_WARN << "For utterance " << utt << ", output would have no rows, "
                           << "producing no output.";
                num_err++;
                continue;
            }

            Matrix<BaseFloat> output(num_frames, feats.NumCols());

            frames_in += feats.NumRows();
            frames_out += num_frames;
            int32 k=0;
            int32 j=0;

            while(k < num_frames) {
                if (j>=n) j=0;
                int32 i = offset + j;
                SubVector<BaseFloat> src(feats, i), dest(output, k);
                dest.CopyFromVec(src);
                j++;
                k++;
            }
            feat_writer.Write(utt, output);
            num_done++;
        }
        KALDI_LOG << "Processed " << num_done << " feature matrices; " << num_err
                  << " with errors.";
        KALDI_LOG << "Processed " << frames_in << " input frames and "
                  << frames_out << " output frames.";
        return (num_done != 0 ? 0 : 1);
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
