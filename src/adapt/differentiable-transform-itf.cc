// adapt/differentiable-transform-itf.cc

// Copyright     2018  Johns Hopkins University (author: Daniel Povey)

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

#include "adapt/differentiable-transform-itf.h"
#include "adapt/generic-transform.h"
#include "adapt/differentiable-transform.h"

namespace kaldi {
namespace differentiable_transform {


// static
DifferentiableTransform* DifferentiableTransform::ReadNew(
    std::istream &is, bool binary) {

  std::string token;
  ReadToken(is, binary, &token); // e.g. "<NoOpTransform>"
  token.erase(0, 1); // erase "<".
  token.erase(token.length()-1); // erase ">".
  DifferentiableTransform *ans = NewTransformOfType(token);
  if (!ans)
    KALDI_ERR << "Unknown DifferentialbeTransform type " << token
              << " (maybe you should recompile?)";
  ans->Read(is, binary);
  return ans;
}

// static
DifferentiableTransform* DifferentiableTransform::NewTransformOfType(
    const std::string &type) {
  if (type.size() > 2 && type[type.size() - 1] == '>') {
    std::string new_type(type);
    if (new_type[0] == '<')
      new_type.erase(0, 1);  // erase "<"
    new_type.erase(new_type.size() - 1);  // erase ">".
    return NewTransformOfType(new_type);
  }

  if (type == "NoOpTransform") {
    return new NoOpTransform();
  } else if (type == "FmllrTransform") {
    return new FmllrTransform();
  } else if (type == "MeanOnlyTransform") {
    return new MeanOnlyTransform();
  } else if (type == "SequenceTransform") {
    return new SequenceTransform();
  } else if (type == "AppendTransform") {
    return new AppendTransform();
  } else {
    // Calling code will throw an error.
    return NULL;
  }
}


void DifferentiableTransform::TestingForwardBatch(
    const CuMatrixBase<BaseFloat> &input,
    int32 num_chunks,
    int32 num_spk,
    const Posterior &posteriors,
    CuMatrixBase<BaseFloat> *output) const {
  int32 dim = input.NumCols(),
      num_frames = input.NumRows(),
      chunks_per_spk = num_chunks / num_spk,
      frames_per_chunk = num_frames / num_chunks;

  // Just copy to CPU for now.
  Matrix<BaseFloat> input_cpu(input);
  Matrix<BaseFloat> output_cpu(num_frames, dim, kUndefined);

  for (int32 s = 0; s < num_spk; s++) {
    SpeakerStatsItf *stats = this->GetEmptySpeakerStats();
    for (int32 chunk = s * chunks_per_spk;
         chunk < (s + 1) * chunks_per_spk; chunk++) {
      SubMatrix<BaseFloat> this_input(input_cpu.RowData(chunk),
                                      frames_per_chunk, dim,
                                      input_cpu.Stride() * num_chunks);
      SubPosterior this_posteriors(posteriors,
                                   chunk, // offset
                                   frames_per_chunk, // num_frames
                                   num_chunks);  // stride
      this->TestingAccumulate(this_input, this_posteriors, stats);
    }
    stats->Estimate();
    for (int32 chunk = s * chunks_per_spk;
         chunk < (s + 1) * chunks_per_spk; chunk++) {
      SubMatrix<BaseFloat> this_input(input_cpu.RowData(chunk),
                                      frames_per_chunk, dim,
                                      input_cpu.Stride() * num_chunks),
          this_output(output_cpu.RowData(chunk),
                      frames_per_chunk, dim,
                      output_cpu.Stride() * num_chunks);
      this->TestingForward(this_input, *stats, &this_output);
    }
    delete stats;
  }
  output->CopyFromMat(output_cpu);
}

// static
DifferentiableTransform* DifferentiableTransform::ReadFromConfig(
    std::istream &is, int32 num_classes) {
  std::vector<std::string> lines;
  ReadConfigLines(is, &lines);
  std::vector<ConfigLine> config_lines;
  ParseConfigLines(lines, &config_lines);
  if (config_lines.empty())
    KALDI_ERR << "Config file is empty.";
  std::string transform_type = config_lines[0].FirstToken();
  DifferentiableTransform *transform = NewTransformOfType(transform_type);
  if (transform == NULL)
    KALDI_ERR << "Parsing config file, could not find transform of type "
              << transform_type;
  int32 pos = transform->InitFromConfig(0, &config_lines);
  if (pos != static_cast<int32>(config_lines.size()))
    KALDI_ERR << "Found junk at end of config file, starting with line "
              << pos << ": " << config_lines[pos].WholeLine();
  KALDI_ASSERT(num_classes > 0);
  transform->SetNumClasses(num_classes);
  return transform;
}

int32 DifferentiableTransformMapped::NumPdfs() const {
  if (pdf_map.empty())
    return transform->NumClasses();
  else
    return static_cast<int32>(pdf_map.size());
}

void DifferentiableTransformMapped::Read(std::istream &is, bool binary) {
  if (transform)
    delete transform;
  transform = DifferentiableTransform::ReadNew(is, binary);
  ReadIntegerVector(is, binary, &pdf_map);
  Check();
}

void DifferentiableTransformMapped::Write(std::ostream &os, bool binary) const {
  Check();
  transform->Write(os, binary);
  WriteIntegerVector(os, binary, pdf_map);
}


void DifferentiableTransformMapped::Check() const {
  KALDI_ASSERT(transform != NULL &&
               (pdf_map.empty() ||
                *std::max_element(pdf_map.begin(), pdf_map.end()) + 1 ==
                transform->NumClasses()));
}

}  // namespace differentiable_transform
}  // namespace kaldi
