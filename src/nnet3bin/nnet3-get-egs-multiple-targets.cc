// nnet3bin/nnet3-get-egs-multiple-targets.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//           2014-2016  Vimal Manohar

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

bool ToBool(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if ((str.compare("true") == 0) || (str.compare("t") == 0)
      || (str.compare("1") == 0)) 
    return true;
  if ((str.compare("false") == 0) || (str.compare("f") == 0)
      || (str.compare("0") == 0)) 
    return false;
  KALDI_ERR << "Invalid format for boolean argument [expected true or false]: "
            << str;
  return false;  // never reached
}

static void ProcessFile(
    const MatrixBase<BaseFloat> &feats,
    const MatrixBase<BaseFloat> *ivector_feats,
    const std::vector<std::string> &output_names,
    const std::vector<int32> &output_dims,
    const std::vector<const MatrixBase<BaseFloat>* > &dense_target_matrices,
    const std::vector<const Posterior*> &posteriors,
    const std::vector<const VectorBase<BaseFloat>* > &deriv_weights,
    const std::string &utt_id,
    bool compress_input,
    int32 input_compress_format,
    const std::vector<bool> &compress_targets,
    const std::vector<int32> &targets_compress_formats,
    int32 left_context,
    int32 right_context,
    int32 frames_per_eg,
    std::vector<int64> *num_frames_written,
    std::vector<int64> *num_egs_written,
    NnetExampleWriter *example_writer) {
  
  KALDI_ASSERT(output_names.size() > 0);

  for (int32 t = 0; t < feats.NumRows(); t += frames_per_eg) {

    int32 tot_frames = left_context + frames_per_eg + right_context;

    Matrix<BaseFloat> input_frames(tot_frames, feats.NumCols(), kUndefined);
    
    // Set up "input_frames".
    for (int32 j = -left_context; j < frames_per_eg + right_context; j++) {
      int32 t2 = j + t;
      if (t2 < 0) t2 = 0;
      if (t2 >= feats.NumRows()) t2 = feats.NumRows() - 1;
      SubVector<BaseFloat> src(feats, t2),
          dest(input_frames, j + left_context);
      dest.CopyFromVec(src);
    }

    NnetExample eg;
    
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", - left_context,
                           input_frames));

    if (compress_input)
      eg.io.back().Compress(input_compress_format);
    
    // if applicable, add the iVector feature.
    if (ivector_feats) {
      int32 actual_frames_per_eg = std::min(frames_per_eg,
                                            feats.NumRows() - t);
      // try to get closest frame to middle of window to get
      // a representative iVector.
      int32 closest_frame = t + (actual_frames_per_eg / 2);
      KALDI_ASSERT(ivector_feats->NumRows() > 0);
      if (closest_frame >= ivector_feats->NumRows())
        closest_frame = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(closest_frame));
      eg.io.push_back(NnetIo("ivector", 0, ivector));
    }

    int32 num_outputs_added = 0;

    for (int32 n = 0; n < output_names.size(); n++) {
      Vector<BaseFloat> this_deriv_weights(0);
      if (deriv_weights[n]) {
        // actual_frames_per_eg is the number of frames with actual targets.
        // At the end of the file, we pad with the last frame repeated
        // so that all examples have the same structure (prevents the need
        // for recompilations).
        int32 actual_frames_per_eg = std::min(
            std::min(frames_per_eg, feats.NumRows() - t),
            deriv_weights[n]->Dim() - t);

        this_deriv_weights.Resize(frames_per_eg);
        int32 frames_to_copy = std::min(t + actual_frames_per_eg, 
                                        deriv_weights[n]->Dim()) - t; 
        this_deriv_weights.Range(0, frames_to_copy).CopyFromVec(
            deriv_weights[n]->Range(t, frames_to_copy));
      }

      if (dense_target_matrices[n]) {
        const MatrixBase<BaseFloat> &targets = *dense_target_matrices[n];
        Matrix<BaseFloat> targets_dest(frames_per_eg, targets.NumCols());
        
        // actual_frames_per_eg is the number of frames with actual targets.
        // At the end of the file, we pad with the last frame repeated
        // so that all examples have the same structure (prevents the need
        // for recompilations).
        int32 actual_frames_per_eg = std::min(
            std::min(frames_per_eg, feats.NumRows() - t),
            targets.NumRows() - t);

        for (int32 i = 0; i < actual_frames_per_eg; i++) {
          // Copy the i^th row of the target matrix from the (t+i)^th row of the
          // input targets matrix
          SubVector<BaseFloat> this_target_dest(targets_dest, i);
          SubVector<BaseFloat> this_target_src(targets, t+i);
          this_target_dest.CopyFromVec(this_target_src);
        }
        
        // Copy the last frame's target to the padded frames
        for (int32 i = actual_frames_per_eg; i < frames_per_eg; i++) {
          // Copy the i^th row of the target matrix from the last row of the 
          // input targets matrix
          KALDI_ASSERT(t + actual_frames_per_eg - 1 == targets.NumRows() - 1); 
          SubVector<BaseFloat> this_target_dest(targets_dest, i);
          SubVector<BaseFloat> this_target_src(targets, 
                                               t + actual_frames_per_eg - 1);
          this_target_dest.CopyFromVec(this_target_src);
        }

        if (deriv_weights[n]) {
          eg.io.push_back(NnetIo(output_names[n], this_deriv_weights, 
                                 0, targets_dest));
        } else {
          eg.io.push_back(NnetIo(output_names[n], 0, targets_dest));
        }
      } else if (posteriors[n]) {
        const Posterior &pdf_post = *(posteriors[n]);

        // actual_frames_per_eg is the number of frames with actual targets.
        // At the end of the file, we pad with the last frame repeated
        // so that all examples have the same structure (prevents the need
        // for recompilations).
        int32 actual_frames_per_eg = std::min(
            std::min(frames_per_eg, feats.NumRows() - t),
            static_cast<int32>(pdf_post.size()) - t);

        Posterior labels(frames_per_eg);
        for (int32 i = 0; i < actual_frames_per_eg; i++)
          labels[i] = pdf_post[t + i];
        // remaining posteriors for frames are empty.

        if (deriv_weights[n]) {
          eg.io.push_back(NnetIo(output_names[n], this_deriv_weights,
                                 output_dims[n], 0, labels));
        } else {
          eg.io.push_back(NnetIo(output_names[n], output_dims[n], 0, labels));
        }
      } else 
        continue;
      if (compress_targets[n])
        eg.io.back().Compress(targets_compress_formats[n]);

      num_outputs_added++;
      // Actually actual_frames_per_eg, but that depends on the different
      // output. For simplification, frames_per_eg is used.
      (*num_frames_written)[n] += frames_per_eg;   
      (*num_egs_written)[n] += 1;
    }

    if (num_outputs_added != output_names.size()) continue;
      
    std::ostringstream os;
    os << utt_id << "-" << t;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    KALDI_ASSERT(NumOutputs(eg) == num_outputs_added);

    example_writer->Write(key, eg);
  }
}


} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3 neural network training.\n"
        "This program is similar to nnet3-get-egs, but the targets here are "
        "dense matrices instead of posteriors (sparse matrices).\n"
        "This is useful when you want the targets to be continuous real-valued "
        "with the neural network possibly trained with a quadratic objective\n"
        "\n"
        "Usage:  nnet3-get-egs-multiple-targets [options] "
        "<features-rspecifier> <output1-name>:<targets-rspecifier1>:<num-targets1>[:<deriv-weights-rspecifier1>] "
        "[ <output2-name>:<targets-rspecifier2>:<num-targets2> ... <targets-rspecifierN> ] <egs-out>\n"
        "\n"
        "Here <outputN-name> is any random string for output node name, \n"
        "<targets_rspecifierN> is the rspecifier for either dense targets in matrix format or sparse targets in posterior format,\n"
        "and <num-targetsN> is the target dimension of output node for sparse targets or -1 for dense targets\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet-get-egs-multiple-targets --left-context=12 \\\n"
        "--right-context=9 --num-frames=8 \"$feats\" \\\n"
        "output-snr:\"ark:copy-matrix ark:exp/snrs/snr.1.ark ark:- |\":-1 \n"
        "   ark:- \n";
        

    bool compress_input = true;
    int32 input_compress_format = 0; 
    int32 left_context = 0, right_context = 0,
          num_frames = 1, length_tolerance = 2;
        
    std::string ivector_rspecifier, 
                targets_compress_formats_str,
                compress_targets_str;
    std::string output_dims_str;
    std::string output_names_str;

    ParseOptions po(usage);
    po.Register("compress-input", &compress_input, "If true, write egs in "
                "compressed format.");
    po.Register("input-compress-format", &input_compress_format, "Format for "
                "compressing input feats e.g. Use 2 for compressing wave");
    po.Register("compress-targets", &compress_targets_str, "CSL of whether "
                "targets must be compressed for each of the outputs");
    po.Register("targets-compress-formats", &targets_compress_formats_str,
                "Format for compressing all feats in general");
    po.Register("left-context", &left_context, "Number of frames of left "
                "context the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right "
                "context the neural net requires.");
    po.Register("num-frames", &num_frames, "Number of frames with labels "
                "that each example contains.");
    po.Register("ivectors", &ivector_rspecifier, "Rspecifier of ivector "
                "features, as matrix.");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("output-dims", &output_dims_str, "CSL of output node dims");
    po.Register("output-names", &output_names_str, "CSL of output node names");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
               examples_wspecifier = po.GetArg(po.NumArgs());

    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessBaseFloatMatrixReader ivector_reader(ivector_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    int32 num_outputs = (po.NumArgs() - 2) / 2;
    KALDI_ASSERT(num_outputs > 0);
    
    std::vector<RandomAccessBaseFloatVectorReader*> deriv_weights_readers(
        num_outputs, static_cast<RandomAccessBaseFloatVectorReader*>(NULL));
    std::vector<RandomAccessBaseFloatMatrixReader*> dense_targets_readers(
        num_outputs, static_cast<RandomAccessBaseFloatMatrixReader*>(NULL));
    std::vector<RandomAccessPosteriorReader*> sparse_targets_readers(
        num_outputs, static_cast<RandomAccessPosteriorReader*>(NULL));

    std::vector<bool> compress_targets(1, true);
    std::vector<std::string> compress_targets_vector;

    if (!compress_targets_str.empty()) {
      SplitStringToVector(compress_targets_str, ":,",
                          true, &compress_targets_vector);
    }

    if (compress_targets_vector.size() == 1 && num_outputs != 1) {
      KALDI_WARN << "compress-targets is of size 1. "
                 << "Extending it to size num-outputs=" << num_outputs;
      compress_targets[0] = ToBool(compress_targets_vector[0]);
      compress_targets.resize(num_outputs, ToBool(compress_targets_vector[0]));
    } else {
      if (compress_targets_vector.size() != num_outputs) {
        KALDI_ERR << "Mismatch in length of compress-targets and num-outputs; "
                  << compress_targets_vector.size() << " vs " << num_outputs;
      }
      for (int32 n = 0; n < num_outputs; n++) {
        compress_targets[n] = ToBool(compress_targets_vector[n]);
      }
    }

    std::vector<int32> targets_compress_formats(1, 1);
    if (!targets_compress_formats_str.empty()) {
      SplitStringToIntegers(targets_compress_formats_str, ":,", 
                            true, &targets_compress_formats);
    }

    if (targets_compress_formats.size() == 1 && num_outputs != 1) {
      KALDI_WARN << "targets-compress-formats is of size 1. "
                 << "Extending it to size num-outputs=" << num_outputs;
      targets_compress_formats.resize(num_outputs, targets_compress_formats[0]);
    }

    if (targets_compress_formats.size() != num_outputs) {
      KALDI_ERR << "Mismatch in length of targets-compress-formats "
                << " and num-outputs; "
                << targets_compress_formats.size() << " vs " << num_outputs;
    }
    
    std::vector<int32> output_dims(num_outputs);
    SplitStringToIntegers(output_dims_str, ":,", 
                            true, &output_dims);

    std::vector<std::string> output_names(num_outputs);
    SplitStringToVector(output_names_str, ":,", true, &output_names);
    
    std::vector<std::string> targets_rspecifiers(num_outputs);
    std::vector<std::string> deriv_weights_rspecifiers(num_outputs);
    
    for (int32 n = 0; n < num_outputs; n++) {
      const std::string &targets_rspecifier = po.GetArg(2*n + 2);
      const std::string &deriv_weights_rspecifier = po.GetArg(2*n + 3);
  
      targets_rspecifiers[n] = targets_rspecifier;
      deriv_weights_rspecifiers[n] = deriv_weights_rspecifier;

      if (output_dims[n] >= 0) {
        sparse_targets_readers[n] = new RandomAccessPosteriorReader(
            targets_rspecifier);
      } else {
        dense_targets_readers[n] = new RandomAccessBaseFloatMatrixReader(
            targets_rspecifier);
      }

      if (!deriv_weights_rspecifier.empty())
        deriv_weights_readers[n] = new RandomAccessBaseFloatVectorReader(
            deriv_weights_rspecifier);

      KALDI_LOG << "output-name=" << output_names[n]
                << " target-dim=" << output_dims[n]
                << " targets-rspecifier=\"" << targets_rspecifiers[n] << "\""
                << " deriv-weights-rspecifier=\"" 
                << deriv_weights_rspecifiers[n] << "\""
                << " compress-target=" 
                << (compress_targets[n] ? "true" : "false")
                << " target-compress-format=" << targets_compress_formats[n];
    }

    int32 num_done = 0, num_err = 0;
    
    std::vector<int64> num_frames_written(num_outputs, 0);
    std::vector<int64> num_egs_written(num_outputs, 0);
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
        
      const Matrix<BaseFloat> *ivector_feats = NULL;
      if (!ivector_rspecifier.empty()) {
        if (!ivector_reader.HasKey(key)) {
          KALDI_WARN << "No iVectors for utterance " << key;
          continue;
        } else {
          // this address will be valid until we call HasKey() or Value()
          // again.
          ivector_feats = &(ivector_reader.Value(key));
        }
      }

      if (ivector_feats && 
          (abs(feats.NumRows() - ivector_feats->NumRows()) > length_tolerance
           || ivector_feats->NumRows() == 0)) {
        KALDI_WARN << "Length difference between feats " << feats.NumRows()
                   << " and iVectors " << ivector_feats->NumRows()
                   << "exceeds tolerance " << length_tolerance;
        num_err++;
        continue;
      }

      std::vector<const MatrixBase<BaseFloat>* > dense_targets(
          num_outputs, static_cast<const Matrix<BaseFloat>* >(NULL));
      std::vector<const Posterior* > sparse_targets(
          num_outputs, static_cast<const Posterior* >(NULL));
      std::vector<const VectorBase<BaseFloat>* > deriv_weights(
          num_outputs, static_cast<const Vector<BaseFloat>* >(NULL));

      int32 num_outputs_found = 0;
      for (int32 n = 0; n < num_outputs; n++) {
        if (dense_targets_readers[n]) {
          if (!dense_targets_readers[n]->HasKey(key)) {
            KALDI_WARN << "No dense targets matrix for key " << key << " in " 
                       << "rspecifier " << targets_rspecifiers[n] 
                       << " for output " << output_names[n];
            break;
          } 
          const MatrixBase<BaseFloat> *target_matrix = &(
              dense_targets_readers[n]->Value(key));
          
          if ((target_matrix->NumRows() - feats.NumRows()) > length_tolerance) {
            KALDI_WARN << "Length difference between feats " << feats.NumRows()
                       << " and target matrix " << target_matrix->NumRows()
                       << "exceeds tolerance " << length_tolerance;
            break;
          }

          dense_targets[n] = target_matrix;
        } else {
          if (!sparse_targets_readers[n]->HasKey(key)) {
            KALDI_WARN << "No sparse target matrix for key " << key << " in " 
                       << "rspecifier " << targets_rspecifiers[n]
                       << " for output " << output_names[n];
            break;
          } 
          const Posterior *posterior = &(sparse_targets_readers[n]->Value(key));

          if (abs(static_cast<int32>(posterior->size()) - feats.NumRows()) 
              > length_tolerance
              || posterior->size() < feats.NumRows()) {
            KALDI_WARN << "Posterior has wrong size " << posterior->size()
                       << " versus " << feats.NumRows();
            break;
          }
        
          sparse_targets[n] = posterior;
        }
        
        if (deriv_weights_readers[n]) {
          if (!deriv_weights_readers[n]->HasKey(key)) {
            KALDI_WARN << "No deriv weights for key " << key << " in " 
                       << "rspecifier " << deriv_weights_rspecifiers[n]
                       << " for output " << output_names[n];
            break;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            deriv_weights[n] = &(deriv_weights_readers[n]->Value(key));
          }
        }
        
        if (deriv_weights[n] 
            && (abs(feats.NumRows() - deriv_weights[n]->Dim())
                > length_tolerance
                || deriv_weights[n]->Dim() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and deriv weights " << deriv_weights[n]->Dim()
                     << " exceeds tolerance " << length_tolerance;
          break;
        }
        
        num_outputs_found++;
      }

      if (num_outputs_found != num_outputs) {
        KALDI_WARN << "Not all outputs found for key " << key;
        num_err++;
        continue;
      }

      ProcessFile(feats, ivector_feats, output_names, output_dims,
                  dense_targets, sparse_targets,
                  deriv_weights, key,
                  compress_input, input_compress_format, 
                  compress_targets, targets_compress_formats,
                  left_context, right_context, num_frames,
                  &num_frames_written, &num_egs_written,
                  &example_writer);
      num_done++;
    }

    int64 max_num_egs_written = 0, max_num_frames_written = 0;
    for (int32 n = 0; n < num_outputs; n++) {
      delete dense_targets_readers[n];
      delete sparse_targets_readers[n];
      delete deriv_weights_readers[n];
      if (num_egs_written[n] == 0) return false;
      if (num_egs_written[n] > max_num_egs_written) {
        max_num_egs_written = num_egs_written[n];
        max_num_frames_written = num_frames_written[n];
      }
    }

    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, wrote at most " << max_num_egs_written 
              << " examples, "
              << " with at most " << max_num_frames_written << " egs in total; "
              << num_err << " files had errors.";

    return (num_err > num_done ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
