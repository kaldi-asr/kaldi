// featbin/append-ivector-to-feats.cc

// Copyright 2012 Korbinian Riedhammer
//           2013 Brno University of Technology (Author: Karel Vesely)
//           2013-2014 Johns Hopkins University (Author: Daniel Povey)
//           2018 Alibaba (Author: Shaofei Xue) 

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
#include "matrix/kaldi-matrix.h"

namespace kaldi {

	void AppendPeriodVectorToFeats(const Matrix<BaseFloat> &in,
		const Matrix<BaseFloat> &vec,
		int32 period,
		Matrix<BaseFloat> *out) {
		KALDI_ASSERT(in.NumRows() != 0);
		KALDI_ASSERT((in.NumRows() + period - 1) / period == vec.NumRows());
		out->Resize(in.NumRows(), in.NumCols() + vec.NumCols());
		out->Range(0, in.NumRows(),
			0, in.NumCols()).CopyFromMat(in);

		for (int i = 0; i < vec.NumRows()-1; i++)
		{
			out->Range(i*period, period,
			in.NumCols(), vec.NumCols()).CopyRowsFromVec(vec.Row(i));
	    }
		out->Range((vec.NumRows() - 1)*period, out->NumRows() - (vec.NumRows() - 1)*period,
			in.NumCols(), vec.NumCols()).CopyRowsFromVec(vec.Row(vec.NumRows() - 1));

}
void AppendVectorToFeats(const Matrix<BaseFloat> &in,
	const Vector<BaseFloat> &vec,
	Matrix<BaseFloat> *out) {
	KALDI_ASSERT(in.NumRows() != 0);
	out->Resize(in.NumRows(), in.NumCols() + vec.Dim());
    out->ColRange(0, in.NumCols()).CopyFromMat(in);
    out->ColRange(in.NumCols(), vec.Dim()).CopyRowsFromVec(vec);

}

}

int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		using namespace std;

		const char *usage =
			"Append i-vector to each row of input feature files\n"
			"\n"
			"Usage: append-ivector-to-feats <in-featspecifier> <in-ivectorspecifier> <out-wspecifier>\n"
			" or: append-vector-to-feats <in-featfilename1> <in-ivectorfilename> <out-wxfilename>\n"
			"See also: paste-feats, concat-feats\n";

		ParseOptions po(usage);

		bool binary = true;
		int online_ivector_period = 1;
		po.Register("binary", &binary, "If true, output files in binary "
			"(only relevant for single-file operation, i.e. no tables)");
		po.Register("online-ivector-period", &online_ivector_period, "Number of "
			"frames between iVectors in matrices supplied to the "
			"--online-ivectors option");

		po.Read(argc, argv);

		if (po.NumArgs() != 3) {
			po.PrintUsage();
			exit(1);
		}

		if (online_ivector_period == 1)
		{
			if (ClassifyRspecifier(po.GetArg(1), NULL, NULL)
				!= kNoRspecifier) {
				// We're operating on tables, e.g. archives.


				string feat_rspecifier = po.GetArg(1);
				SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);

				string vec_rspecifier = po.GetArg(2);
				RandomAccessBaseFloatVectorReader vec_reader(vec_rspecifier);

				string wspecifier = po.GetArg(3);
				BaseFloatMatrixWriter feat_writer(wspecifier);

				int32 num_done = 0, num_err = 0;
				// Main loop
				for (; !feat_reader.Done(); feat_reader.Next()) {
					string utt = feat_reader.Key();
					KALDI_VLOG(2) << "Processing utterance " << utt;

					const Matrix<BaseFloat> &feats(feat_reader.Value());

					if (!vec_reader.HasKey(utt)) {
						KALDI_WARN << "Could not read vector for utterance " << utt;
						num_err++;
						continue;
					}
					const Vector<BaseFloat> &vec(vec_reader.Value(utt));

					Matrix<BaseFloat> output;
					AppendVectorToFeats(feats, vec, &output);
					feat_writer.Write(utt, output);
					num_done++;
				}
				KALDI_LOG << "Done " << num_done << " utts, errors on "
					<< num_err;

				return (num_done == 0 ? -1 : 0);
			}
			else {
				// We're operating on rxfilenames|wxfilenames, most likely files.
				Matrix<BaseFloat> mat;
				ReadKaldiObject(po.GetArg(1), &mat);
				Vector<BaseFloat> vec;
				ReadKaldiObject(po.GetArg(2), &vec);
				Matrix<BaseFloat> output;
				AppendVectorToFeats(mat, vec, &output);
				std::string output_wxfilename = po.GetArg(3);
				WriteKaldiObject(output, output_wxfilename, binary);
				KALDI_LOG << "Wrote appended features to " << output_wxfilename;
				return 0;
			}
		}
		else{
			if (ClassifyRspecifier(po.GetArg(1), NULL, NULL)
				!= kNoRspecifier) {
				// We're operating on tables, e.g. archives.


				string feat_rspecifier = po.GetArg(1);
				SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);

				string vec_rspecifier = po.GetArg(2);
				RandomAccessBaseFloatMatrixReader vec_reader(vec_rspecifier);

				string wspecifier = po.GetArg(3);
				BaseFloatMatrixWriter feat_writer(wspecifier);

				int32 num_done = 0, num_err = 0;
				// Main loop
				for (; !feat_reader.Done(); feat_reader.Next()) {
					string utt = feat_reader.Key();
					KALDI_VLOG(2) << "Processing utterance " << utt;

					const Matrix<BaseFloat> &feats(feat_reader.Value());

					if (!vec_reader.HasKey(utt)) {
						KALDI_WARN << "Could not read vector for utterance " << utt;
						num_err++;
						continue;
					}
					const Matrix<BaseFloat> &vec(vec_reader.Value(utt));

					Matrix<BaseFloat> output;
					AppendPeriodVectorToFeats(feats, vec, online_ivector_period, &output);
					feat_writer.Write(utt, output);
					num_done++;
				}
				KALDI_LOG << "Done " << num_done << " utts, errors on "
					<< num_err;

				return (num_done == 0 ? -1 : 0);
			}
			else {
				// We're operating on rxfilenames|wxfilenames, most likely files.
				Matrix<BaseFloat> mat;
				ReadKaldiObject(po.GetArg(1), &mat);
				Matrix<BaseFloat> vec;
				ReadKaldiObject(po.GetArg(2), &vec);
				Matrix<BaseFloat> output;
				AppendPeriodVectorToFeats(mat, vec, online_ivector_period, &output);
				std::string output_wxfilename = po.GetArg(3);
				WriteKaldiObject(output, output_wxfilename, binary);
				KALDI_LOG << "Wrote appended features to " << output_wxfilename;
				return 0;
			}
		}
	


  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

/*
  Testing:

cat <<EOF >1.mat
[ 0 1 2
  3 4 5
  8 9 10 ]
EOF
cat <<EOF > 2.vec
 [ 0 1 ]
EOF
append-vector-to-feats --binary=false 1.mat 2.vec 3a.mat
cat <<EOF > 3b.mat
 [ 0 1 2 0 1
   3 4 5 0 1
   8 9 10 0 1 ]
EOF
cmp <(../bin/copy-matrix 3b.mat -) <(../bin/copy-matrix 3a.mat -) || echo 'Bad!'

append-vector-to-feats 'scp:echo foo 1.mat|' 'scp:echo foo 2.vec|' 'scp,t:echo foo 3a.mat|'
cmp <(../bin/copy-matrix 3b.mat -) <(../bin/copy-matrix 3a.mat -) || echo 'Bad!'

rm {1,3?}.mat 2.vec
 */
