// nnet3bin/nnet3-copy-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar

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
#include "hmm/transition-model.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {

// returns an integer randomly drawn with expected value "expected_count"
// (will be either floor(expected_count) or ceil(expected_count)).
int32 GetCount(double expected_count) {
  KALDI_ASSERT(expected_count >= 0.0);
  int32 ans = floor(expected_count);
  expected_count -= ans;
  if (WithProb(expected_count))
    ans++;
  return ans;
}

/** Returns true if the "eg" contains just a single example, meaning
    that all the "n" values in the indexes are zero, and the example
    has NnetIo members named both "input" and "output"
    
    Also computes the minimum and maximum "t" values in the "input" and
    "output" NnetIo members.
 */
bool ContainsSingleExample(const NnetExample &eg,
                           int32 *min_input_t,
                           int32 *max_input_t,
                           int32 *min_output_t,
                           int32 *max_output_t) {
  bool done_input = false, done_output = false;
  int32 num_indexes = eg.io.size();
  for (int32 i = 0; i < num_indexes; i++) {
    const NnetIo &io = eg.io[i];
    std::vector<Index>::const_iterator iter = io.indexes.begin(),
                                        end = io.indexes.end();
    // Should not have an empty input/output type.
    KALDI_ASSERT(!io.indexes.empty());
    if (io.name == "input" || io.name == "output") {
      int32 min_t = iter->t, max_t = iter->t;
      for (; iter != end; ++iter) {
        int32 this_t = iter->t;
        min_t = std::min(min_t, this_t);
        max_t = std::max(max_t, this_t);
        if (iter->n != 0) {
          KALDI_WARN << "Example does not contain just a single example; "
                     << "too late to do frame selection or reduce context.";
          return false;
        }
      }
      if (io.name == "input") {
        done_input = true;
        *min_input_t = min_t;
        *max_input_t = max_t;
      } else {
        KALDI_ASSERT(io.name == "output");
        done_output = true;
        *min_output_t = min_t;
        *max_output_t = max_t;
      }
    } else {
      for (; iter != end; ++iter) {
        if (iter->n != 0) {
          KALDI_WARN << "Example does not contain just a single example; "
                     << "too late to do frame selection or reduce context.";
          return false;
        }
      }
    }
  }
  if (!done_input) {
    KALDI_WARN << "Example does not have any input named 'input'";
    return false;
  }
  if (!done_output) {
    KALDI_WARN << "Example does not have any output named 'output'";
    return false;
  }
  return true;
}

struct QuantizationOptions {
  void Register(OptionsItf *opts) {
    opts->Register("bin-boundaries", &bin_boundaries_str, "Bin boundaries");
  }
  
  std::string bin_boundaries_str;
};

class ExampleSelector {
 public:
  bool SelectFromExample(const NnetExample &eg,
                         NnetExample *eg_out) const;
  
  void QuantizeExample(const NnetExample &eg,
                       NnetExample *eg_out) const;

  ExampleSelector(std::string frame_str, 
                  int32 left_context, int32 right_context,
                  const QuantizationOptions &quantization_opts,
                  bool quantize_input): frame_str_(frame_str), 
                  left_context_(left_context), right_context_(right_context),
                  quantize_input_(quantize_input), 
                  quantization_opts_(quantization_opts) { 
    if (!SplitStringToFloats(quantization_opts_.bin_boundaries_str, ":", false,
                               &bin_boundaries_)) {
      KALDI_ERR << "Bad value for --bin-boundaries option: "
                << quantization_opts_.bin_boundaries_str;
    }
    
    if (quantize_input && NumBins() <= 1) {
      KALDI_ERR << "Bad value for --bin-boundaries option: "
                << quantization_opts_.bin_boundaries_str;
    }
  }

 private:

  void FilterExample(const NnetExample &eg, 
                     int32 min_input_t,
                     int32 max_input_t,
                     int32 min_output_t,
                     int32 max_output_t,
                     NnetExample *eg_out) const;

  void FilterAndQuantizeGeneralMatrixRows(const GeneralMatrix &in,
                                          const std::vector<bool> &keep_rows,
                                          GeneralMatrix *out) const;

  void QuantizeGeneralMatrix(const GeneralMatrix &in,
                             GeneralMatrix *out) const;

  void QuantizeFeats(const MatrixBase<BaseFloat> &in, SparseMatrix<BaseFloat> *out) const;
  void QuantizeFeats(SparseMatrix<BaseFloat> *out) const;
  
  int32 NumBins() const { return bin_boundaries_.size() + 1; }

  std::string frame_str_;
  int32 left_context_;
  int32 right_context_;
  bool quantize_input_;

  const QuantizationOptions &quantization_opts_;
  
  std::vector<BaseFloat> bin_boundaries_;
};

void ExampleSelector::QuantizeFeats(const MatrixBase<BaseFloat> &in,
                                     SparseMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), in.NumCols() * NumBins());
  for (size_t t = 0; t < in.NumRows(); t++) {
    std::vector<std::pair<int32, BaseFloat> > bins(in.NumCols());
    for (size_t j = 0; j < in.NumCols(); j++) {
      auto bin = std::lower_bound(bin_boundaries_.begin(), 
                                  bin_boundaries_.end(), in(t,j));
      size_t k;
      if (bin != bin_boundaries_.end()) 
        k = static_cast<size_t>(bin - bin_boundaries_.begin());
      else {
        k = static_cast<size_t>(NumBins() - 1);
        KALDI_ASSERT(k == bin_boundaries_.end() - bin_boundaries_.begin());
      }
      
      KALDI_ASSERT(k >= 0 && k < NumBins());
      KALDI_ASSERT(j + NumBins() + k < out->NumCols());
      
      bins[j] = std::make_pair(j * NumBins() + k, 1.0);
    }
    out->SetRow(t, SparseVector<BaseFloat>(out->NumCols(), bins));
  }
}

void ExampleSelector::QuantizeFeats (SparseMatrix<BaseFloat> *out) const {
  SparseVector<BaseFloat> *row = out->Data();
  for (size_t t = 0; t < out->NumRows(); t++, ++row) {
    std::pair<int32, BaseFloat> *pairs = row->Data();
    for (size_t j = 0; j < (out->Row(t)).NumElements(); j++, ++pairs) {
      auto bin = std::lower_bound(bin_boundaries_.begin(), 
                                  bin_boundaries_.end(), pairs->second);
      size_t k;
      if (bin != bin_boundaries_.end()) 
        k = static_cast<size_t>(bin - bin_boundaries_.begin());
      else 
        k = static_cast<size_t>(NumBins());
      
      pairs->first = pairs->first * NumBins() + k;
      pairs->second = 1.0;
    }
  }
}

void ExampleSelector::QuantizeGeneralMatrix (const GeneralMatrix &in,  
                                             GeneralMatrix *out) const {
  out->Clear();
  switch (in.Type()) {
    case kCompressedMatrix: {
                              const CompressedMatrix &cmat = in.GetCompressedMatrix();
                              Matrix<BaseFloat> full_mat(cmat);
                              SparseMatrix<BaseFloat> smat;
                              QuantizeFeats(full_mat, &smat);
                              out->SwapSparseMatrix(&smat);
                              return;
                            }
    case kSparseMatrix: {
                          SparseMatrix<BaseFloat> smat(in.GetSparseMatrix());
                          QuantizeFeats(&smat);
                          out->SwapSparseMatrix(&smat);
                          return;
                        }
    case kFullMatrix: {
                        const Matrix<BaseFloat> &full_mat = in.GetFullMatrix();
                        SparseMatrix<BaseFloat> smat;
                        QuantizeFeats(full_mat, &smat);
                        out->SwapSparseMatrix(&smat);
                        return;
                      }
    default:
                      KALDI_ERR << "Invalid general-matrix type.";
  }
}

void ExampleSelector::FilterAndQuantizeGeneralMatrixRows(
                              const GeneralMatrix &in,  
                              const std::vector<bool> &keep_rows, 
                              GeneralMatrix *out) const {
  out->Clear();
  KALDI_ASSERT(keep_rows.size() == static_cast<size_t>(in.NumRows()));
  int32 num_kept_rows = 0;
  std::vector<bool>::const_iterator iter = keep_rows.begin(),
    end = keep_rows.end();
  for (; iter != end; ++iter)
    if (*iter)
      num_kept_rows++;
  if (num_kept_rows == 0)
    KALDI_ERR << "No kept rows";
  switch (in.Type()) {
    case kCompressedMatrix: {
                              const CompressedMatrix &cmat = in.GetCompressedMatrix();
                              Matrix<BaseFloat> full_mat_out;
                              FilterCompressedMatrixRows(cmat, keep_rows, &full_mat_out);
                              SparseMatrix<BaseFloat> smat_out;
                              QuantizeFeats(full_mat_out, &smat_out);
                              out->SwapSparseMatrix(&smat_out);
                              return;
                            }
    case kSparseMatrix: {
                          const SparseMatrix<BaseFloat> &smat = in.GetSparseMatrix();
                          SparseMatrix<BaseFloat> smat_out;
                          FilterSparseMatrixRows(smat, keep_rows, &smat_out);
                          QuantizeFeats(&smat_out);
                          return;
                        }
    case kFullMatrix: {
                        const Matrix<BaseFloat> &full_mat = in.GetFullMatrix();
                        Matrix<BaseFloat> full_mat_out;
                        FilterMatrixRows(full_mat, keep_rows, &full_mat_out);
                        SparseMatrix<BaseFloat> smat_out;
                        QuantizeFeats(full_mat_out, &smat_out);
                        out->SwapSparseMatrix(&smat_out);
                        return;
                      }
    default:
                      KALDI_ERR << "Invalid general-matrix type.";
  }
}

void ExampleSelector::QuantizeExample(const NnetExample &eg,
                                      NnetExample *eg_out) const {
  eg_out->io.clear();
  eg_out->io.resize(eg.io.size());
  for (size_t i = 0; i < eg.io.size(); i++) {
    bool is_input = false;
    const NnetIo &io_in = eg.io[i];
    NnetIo &io_out = eg_out->io[i];
    const std::string &name = io_in.name;
    io_out.name = name;
    if (name == "input") {
      is_input = true;
    } 
    io_out.indexes = io_in.indexes;
    if (!is_input || !quantize_input_) // Just copy everything.
      io_out.features = io_in.features;
    else {
      QuantizeGeneralMatrix(io_in.features, &io_out.features);
    }
  }
}

/**
   This function filters the indexes (and associated feature rows) in a
   NnetExample, removing any index/row in an NnetIo named "input" with t <
   min_input_t or t > max_input_t and any index/row in an NnetIo named "output" with t <
   min_output_t or t > max_output_t.
   Will crash if filtering removes all Indexes of "input" or "output".
 */
void ExampleSelector::FilterExample(const NnetExample &eg,
                                    int32 min_input_t,
                                    int32 max_input_t,
                                    int32 min_output_t,
                                    int32 max_output_t,
                                    NnetExample *eg_out) const {
  eg_out->io.clear();
  eg_out->io.resize(eg.io.size());
  for (size_t i = 0; i < eg.io.size(); i++) {
    bool is_input_or_output = false, is_input = false;
    int32 min_t, max_t;
    const NnetIo &io_in = eg.io[i];
    NnetIo &io_out = eg_out->io[i];
    const std::string &name = io_in.name;
    io_out.name = name;
    if (name == "input") {
      min_t = min_input_t;
      max_t = max_input_t;
      is_input_or_output = true;
      is_input = true;
    } else if (name == "output") {
      min_t = min_output_t;
      max_t = max_output_t;
      is_input_or_output = true;
    } else {
      is_input_or_output = false;
    }
    if (!is_input_or_output) {  // Just copy everything.
      io_out.indexes = io_in.indexes;
      io_out.features = io_in.features;
    } else {
      const std::vector<Index> &indexes_in = io_in.indexes;
      std::vector<Index> &indexes_out = io_out.indexes;
      indexes_out.reserve(indexes_in.size());
      int32 num_indexes = indexes_in.size(), num_kept = 0;
      KALDI_ASSERT(io_in.features.NumRows() == num_indexes);
      std::vector<bool> keep(num_indexes, false);
      std::vector<Index>::const_iterator iter_in = indexes_in.begin(),
                                          end_in = indexes_in.end();
      std::vector<bool>::iterator iter_out = keep.begin();
      for (; iter_in != end_in; ++iter_in,++iter_out) {
        int32 t = iter_in->t;
        bool is_within_range = (t >= min_t && t <= max_t);
        *iter_out = is_within_range;
        if (is_within_range) {
          indexes_out.push_back(*iter_in);
          num_kept++;
        }
      }
      KALDI_ASSERT(iter_out == keep.end());
      if (num_kept == 0)
        KALDI_ERR << "FilterExample removed all indexes for '" << name << "'";

      if (is_input && quantize_input_)
        FilterAndQuantizeGeneralMatrixRows(io_in.features, keep,
                                           &io_out.features);
      else
        FilterGeneralMatrixRows(io_in.features, keep,
                                &io_out.features);

      KALDI_ASSERT(io_out.features.NumRows() == num_kept &&
                   indexes_out.size() == static_cast<size_t>(num_kept));
    }
  }
}


/**
   This function is responsible for possibly selecting one frame from multiple
   supervised frames, and reducing the left and right context as specified.  If
   frame == "" it does not reduce the supervised frames; if frame == "random" it
   selects one random frame; otherwise it expects frame to be an integer, and
   will select only the output with that frame index (or return false if there was
   no such output).

   If left_context != -1 it removes any inputs with t < (smallest output - left_context).
      If left_context != -1 it removes any inputs with t < (smallest output - left_context).
   
   It returns true if it was able to select a frame.  We only anticipate it ever
   returning false in situations where frame is an integer, and the eg came from
   the end of a file and has a smaller than normal number of supervised frames.

*/
bool ExampleSelector::SelectFromExample(const NnetExample &eg,
                                        NnetExample *eg_out) const {
  int32 min_input_t, max_input_t,
      min_output_t, max_output_t;
  if (!ContainsSingleExample(eg, &min_input_t, &max_input_t,
                             &min_output_t, &max_output_t))
    KALDI_ERR << "Too late to perform frame selection/context reduction on "
              << "these examples (already merged?)";
  if (frame_str_ != "") {
    // select one frame.
    if (frame_str_ == "random") {
      min_output_t = max_output_t = RandInt(min_output_t,
                                                          max_output_t);
    } else {
      int32 frame;
      if (!ConvertStringToInteger(frame_str_, &frame))
        KALDI_ERR << "Invalid option --frame='" << frame_str_ << "'";
      if (frame < min_output_t || frame > max_output_t) {
        // Frame is out of range.  Should happen only rarely.  Calling code
        // makes sure of this.
        return false;
      }
      min_output_t = max_output_t = frame;
    }
  }
  // There may come a time when we want to remove or make it possible to disable
  // the error messages below.  The std::max and std::min expressions may seem
  // unnecessary but are intended to make life easier if and when we do that.
  if (left_context_ != -1) {
    if (min_input_t > min_output_t - left_context_)
      KALDI_ERR << "You requested --left-context=" << left_context_
                << ", but example only has left-context of "
                <<  (min_output_t - min_input_t);
    min_input_t = std::max(min_input_t, min_output_t - left_context_);
  }
  if (right_context_ != -1) {
    if (max_input_t < max_output_t + right_context_)
      KALDI_ERR << "You requested --right-context=" << right_context_
                << ", but example only has right-context of "
                <<  (max_input_t - max_output_t);
    max_input_t = std::min(max_input_t, max_output_t + right_context_);
  }

  FilterExample(eg,
                min_input_t, max_input_t,
                min_output_t, max_output_t,
                eg_out);

  return true;
}


} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples (single frames or fixed-size groups of frames) for neural\n"
        "network training, possibly changing the binary mode.  Supports multiple wspecifiers, in\n"
        "which case it will write the examples round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet3-copy-egs [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet3-copy-egs ark:train.egs ark,t:text.egs\n"
        "or:\n"
        "nnet3-copy-egs ark:train.egs ark:1.egs ark:2.egs\n";
        
    bool random = false;
    int32 srand_seed = 0;
    BaseFloat keep_proportion = 1.0;

    // The following config variables, if set, can be used to extract a single
    // frame of labels from a multi-frame example, and/or to reduce the amount
    // of context.
    int32 left_context = -1, right_context = -1;

    // you can set frame to a number to select a single frame with a particular
    // offset, or to 'random' to select a random single frame.
    std::string frame_str;
 
    bool quantize_input = false;

    QuantizationOptions quantization_opts;

    ParseOptions po(usage);
    quantization_opts.Register(&po);

    po.Register("random", &random, "If true, will write frames to output "
                "archives randomly, not round-robin.");
    po.Register("keep-proportion", &keep_proportion, "If <1.0, this program will "
                "randomly keep this proportion of the input samples.  If >1.0, it will "
                "in expectation copy a sample this many times.  It will copy it a number "
                "of times equal to floor(keep-proportion) or ceil(keep-proportion).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --random=true or --keep-proportion != 1.0)");
    po.Register("frame", &frame_str, "This option can be used to select a single "
                "frame from each multi-frame example.  Set to a number 0, 1, etc. "
                "to select a frame with a given index, or 'random' to select a "
                "random frame.");
    po.Register("left-context", &left_context, "Can be used to truncate the "
                "feature left-context that we output.");
    po.Register("right-context", &right_context, "Can be used to truncate the "
                "feature right-context that we output.");
    po.Register("quantize-input", &quantize_input, "If true, quantize input");
    
    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetExampleReader example_reader(examples_rspecifier);

    int32 num_outputs = po.NumArgs() - 1;
    std::vector<NnetExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetExampleWriter(po.GetArg(i+2));

    ExampleSelector selector(frame_str, left_context, right_context, 
                             quantization_opts, quantize_input);

    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      // count is normally 1; could be 0, or possibly >1.
      int32 count = GetCount(keep_proportion);  
      std::string key = example_reader.Key();
      const NnetExample &eg = example_reader.Value();
      for (int32 c = 0; c < count; c++) {
        int32 index = (random ? Rand() : num_written) % num_outputs;
        if (frame_str.empty() && left_context == -1 && right_context == -1
            && !quantize_input) {
          example_writers[index]->Write(key, eg);
          num_written++;
        } else { // the --frame option or context options were set.
          NnetExample eg_modified;
          if (! ( frame_str.empty() && left_context == -1 && right_context == -1 ) ) {
            if (selector.SelectFromExample(eg, &eg_modified)) {
              // this branch of the if statement will almost always be taken (should only
              // not be taken for shorter-than-normal egs from the end of a file.
              example_writers[index]->Write(key, eg_modified);
              num_written++;
            }
          } else {
            selector.QuantizeExample(eg, &eg_modified);
            example_writers[index]->Write(key, eg_modified);
            num_written++;
          }
        }
      }
    }
    
    for (int32 i = 0; i < num_outputs; i++)
      delete example_writers[i];
    KALDI_LOG << "Read " << num_read << " neural-network training examples, wrote "
              << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


