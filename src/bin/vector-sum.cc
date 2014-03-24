// bin/vector-sum.cc

// Copyright 2014  Vimal Manohar
//           2014  Johns Hopkins University (author: Daniel Povey)

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

#include <vector>
#include <string>

using std::vector;
using std::string;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-vector.h"
#include "transform/transform-common.h"


namespace kaldi {

// sums a bunch of archives to produce one archive
int32 TypeOneUsage(const ParseOptions &po) {
  int32 num_args = po.NumArgs();
  std::string vector_in_fn1 = po.GetArg(1),
      vector_out_fn = po.GetArg(num_args);

  // Output vector
  BaseFloatVectorWriter vector_writer(vector_out_fn);

  // Input vectors
  SequentialBaseFloatVectorReader vector_reader1(vector_in_fn1);
  std::vector<RandomAccessBaseFloatVectorReader*> vector_readers(num_args-2, 
                                                                 static_cast<RandomAccessBaseFloatVectorReader*>(NULL));
  std::vector<std::string> vector_in_fns(num_args-2);
  for (int32 i = 2; i < num_args; ++i) {
    vector_readers[i-2] = new RandomAccessBaseFloatVectorReader(po.GetArg(i));
    vector_in_fns[i-2] = po.GetArg(i);
  }

  int32 n_utts = 0, n_total_vectors = 0, 
      n_success = 0, n_missing = 0, n_other_errors = 0;

  for (; !vector_reader1.Done(); vector_reader1.Next()) {
    std::string key = vector_reader1.Key();
    Vector<BaseFloat> vector1 = vector_reader1.Value();
    vector_reader1.FreeCurrent();
    n_utts++;
    n_total_vectors++;

    Vector<BaseFloat> vector_out(vector1);

    for (int32 i = 0; i < num_args-2; ++i) {
      if (vector_readers[i]->HasKey(key)) {
        Vector<BaseFloat> vector2 = vector_readers[i]->Value(key);
        n_total_vectors++;
        if (vector2.Dim() == vector_out.Dim()) {
          vector_out.AddVec(1.0, vector2);
        } else {
          KALDI_WARN << "Dimension mismatch for utterance " << key 
                     << " : " << vector2.Dim() << " for "
                     << "system " << (i + 2) << ", rspecifier: "
                     << vector_in_fns[i] << " vs " << vector_out.Dim() 
                     << " primary vector, rspecifier:" << vector_in_fn1;
          n_other_errors++;
        }
      } else {
        KALDI_WARN << "No vector found for utterance " << key << " for "
                   << "system " << (i + 2) << ", rspecifier: "
                   << vector_in_fns[i];
        n_missing++;
      }
    }

    vector_writer.Write(key, vector_out);
    n_success++;
  }

  KALDI_LOG << "Processed " << n_utts << " utterances: with a total of "
            << n_total_vectors << " vectors across " << (num_args-1)
            << " different systems";
  KALDI_LOG << "Produced output for " << n_success << " utterances; "
            << n_missing << " total missing vectors";
  
  DeletePointers(&vector_readers);
  
  return (n_success != 0 && n_missing < (n_success - n_missing)) ? 0 : 1;
}

int32 TypeTwoUsage(const ParseOptions &po,
                   bool binary) {
  KALDI_ASSERT(po.NumArgs() == 2);
  KALDI_ASSERT(ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier &&
               "vector-sum: first argument must be an rspecifier");
  // if next assert fails it would be bug in the code as otherwise we shouldn't
  // be called.
  KALDI_ASSERT(ClassifyRspecifier(po.GetArg(2), NULL, NULL) == kNoRspecifier);

  SequentialBaseFloatVectorReader vec_reader(po.GetArg(1));

  Vector<double> sum;
  
  int32 num_done = 0, num_err = 0;

  for (; !vec_reader.Done(); vec_reader.Next()) {
    const Vector<BaseFloat> &vec = vec_reader.Value();
    if (vec.Dim() == 0) {
      KALDI_WARN << "Zero vector input for key " << vec_reader.Key();
      num_err++;
    } else {
      if (sum.Dim() == 0) sum.Resize(vec.Dim());
      if (sum.Dim() != vec.Dim()) {
        KALDI_WARN << "Dimension mismatch for key " << vec_reader.Key()
                   << ": " << vec.Dim() << " vs. " << sum.Dim();
        num_err++;
      } else {
        sum.AddVec(1.0, vec);
        num_done++;
      }
    }
  }

  Vector<BaseFloat> sum_float(sum);
  WriteKaldiObject(sum_float, po.GetArg(2), binary);

  KALDI_LOG << "Summed " << num_done << " vectors, "
            << num_err << " with errors; wrote sum to "
            << PrintableWxfilename(po.GetArg(2));
  return (num_done > 0 && num_err < num_done) ? 0 : 1;
}

// sum a bunch of single files to produce a single file [including
// extended filenames, of course]
int32 TypeThreeUsage(const ParseOptions &po,
                     bool binary) {
  KALDI_ASSERT(po.NumArgs() >= 2);
  for (int32 i = 1; i <= po.NumArgs(); i++) {
    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      KALDI_ERR << "Wrong usage (type 3): if first and last arguments are not "
                << "tables, the intermediate arguments must not be tables.";
    }
  }

  bool add = true;
  Vector<BaseFloat> vec;
  for (int32 i = 1; i < po.NumArgs(); i++) {
    bool binary_in;
    Input ki(po.GetArg(i), &binary_in);
    // this Read function will throw if there is a size mismatch.
    vec.Read(ki.Stream(), binary_in, add);
  }
  WriteKaldiObject(vec, po.GetArg(po.NumArgs()), binary);
  KALDI_LOG << "Summed " << (po.NumArgs() - 1) << " vectors; "
            << "wrote sum to " << PrintableWxfilename(po.GetArg(po.NumArgs()));
  return 0;
}


} // namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Add vectors (e.g. weights, transition-accs; speaker vectors)\n"
        "If you need to scale the inputs, use vector-scale on the inputs\n"
        "\n"
        "Type one usage:\n"
        " vector-sum [options] <vector-in-rspecifier1> [<vector-in-rspecifier2>"
        " <vector-in-rspecifier3> ...] <vector-out-wspecifier>\n"
        "  e.g.: vector-sum ark:1.weights ark:2.weights ark:combine.weights\n"
        "Type two usage (sums a single table input to produce a single output):\n"
        " vector-sum [options] <vector-in-rspecifier> <vector-out-wxfilename>\n"
        " e.g.: vector-sum --binary=false vecs.ark sum.vec\n"
        "Type three usage (sums single-file inputs to produce a single output):\n"
        " vector-sum [options] <vector-in-rxfilename1> <vector-in-rxfilename2> ..."
        " <vector-out-wxfilename>\n"
        " e.g.: vector-sum --binary=false 1.vec 2.vec 3.vec sum.vec\n";
        
    bool binary;
    
    ParseOptions po(usage);

    po.Register("binary", &binary, "If true, write output as binary (only "
                "relevant for usage types two or three");
    
    po.Read(argc, argv);

    int32 N = po.NumArgs(), exit_status;

    if (po.NumArgs() >= 2 &&
        ClassifyRspecifier(po.GetArg(N), NULL, NULL) != kNoRspecifier) {
      // output to table.
      exit_status = TypeOneUsage(po);
    } else if (po.NumArgs() == 2 &&
               ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier &&
               ClassifyRspecifier(po.GetArg(N), NULL, NULL) ==
               kNoRspecifier) {
      // input from a single table, output not to table.
      exit_status = TypeTwoUsage(po, binary);
    } else if (po.NumArgs() >= 2 &&
               ClassifyRspecifier(po.GetArg(1), NULL, NULL) == kNoRspecifier &&
               ClassifyRspecifier(po.GetArg(N), NULL, NULL) == kNoRspecifier) {
      // summing flat files.
      exit_status = TypeThreeUsage(po, binary);
    } else {      
      po.PrintUsage();
      exit(1);
    }
    return exit_status;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
