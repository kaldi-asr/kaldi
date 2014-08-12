// nnet2/nnet-example-functions-test.cc

// Copyright 2013  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet2/nnet-example-functions.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet2 {

// Note: most of these functions we're testing from the command line,
// this is just to test the function to solve the packing problem.

void UnitTestSolvePackingProblem() {
  size_t size = Rand() % 20;
  std::vector<BaseFloat> item_costs;
  for (size_t i = 0; i < size; i++) {
    item_costs.push_back(0.5 * (Rand() % 15));
  }
  BaseFloat max_cost = 0.66 + Rand() % 5;

  std::vector<std::vector<size_t> > groups;
  SolvePackingProblem(max_cost, item_costs, &groups);
  
  std::vector<size_t> all_indices;
  for (size_t i = 0; i < groups.size(); i++) {
    BaseFloat this_group_cost = 0.0;
    for (size_t j = 0; j < groups[i].size(); j++) {
      size_t index = groups[i][j];
      all_indices.push_back(index);
      this_group_cost += item_costs[index];
    }
    KALDI_ASSERT(!groups[i].empty());
    KALDI_ASSERT(groups[i].size() == 1 || this_group_cost <= max_cost);
  }
  SortAndUniq(&all_indices);
  KALDI_ASSERT(all_indices.size() == size);
  if (!all_indices.empty())
    KALDI_ASSERT(all_indices.back() + 1 == size);
}


} // namespace nnet2
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;
  using kaldi::int32;
  for (int32 i = 0; i < 10; i++)
    UnitTestSolvePackingProblem();
}

