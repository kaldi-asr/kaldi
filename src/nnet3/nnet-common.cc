// nnet3/nnet-common.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-common.h"

namespace kaldi {
namespace nnet3 {

// Don't write with too many markers as we don't want to take up too much space.
void Index::Write(std::ostream &os, bool binary) const {
  // writing this token will make it easier to write back-compatible code later
  // on.
  WriteToken(os, binary, "<I1>");
  WriteBasicType(os, binary, n);
  WriteBasicType(os, binary, t);
  WriteBasicType(os, binary, x);
}


void Index::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<I1>");
  ReadBasicType(is, binary, &n);
  ReadBasicType(is, binary, &t);
  ReadBasicType(is, binary, &x);
}


static void WriteIndexVectorElementBinary(
    std::ostream &os,
    const std::vector<Index> &vec,
    int32 i) {
  bool binary = true;
  Index index = vec[i];
  if (i == 0) {
    if (index.n == 0 && index.x == 0 &&
        std::abs(index.t) < 125) {
      // handle this common case in one character.
      os.put(static_cast<signed char>(index.t));
    } else {  // handle the general case less efficiently.
      os.put(127);
      WriteBasicType(os, binary, index.n);
      WriteBasicType(os, binary, index.t);
      WriteBasicType(os, binary, index.x);
    }
  } else {
    Index last_index = vec[i-1];
    if (index.n == last_index.n && index.x == last_index.x &&
        std::abs(index.t - last_index.t) < 125) {
      signed char c = index.t - last_index.t;
      os.put(c);
    } else {  // handle the general case less efficiently.
      os.put(127);
      WriteBasicType(os, binary, index.n);
      WriteBasicType(os, binary, index.t);
      WriteBasicType(os, binary, index.x);
    }
  }
  if (!os.good())
    KALDI_ERR << "Output stream error detected";
}


static void ReadIndexVectorElementBinary(
    std::istream &is,
    int32 i,
    std::vector<Index> *vec) {
  bool binary = true;
  Index &index = (*vec)[i];
  if (!is.good())
    KALDI_ERR << "End of file while reading vector of Index.";
  signed char c = is.get();
  if (i == 0) {
    if (std::abs(int(c)) < 125) {
      index.n = 0;
      index.t = c;
      index.x = 0;
    } else {
      if (c != 127)
        KALDI_ERR << "Unexpected character " << c
                  << " encountered while reading Index vector.";
      ReadBasicType(is, binary, &(index.n));
      ReadBasicType(is, binary, &(index.t));
      ReadBasicType(is, binary, &(index.x));
    }
  } else {
    Index &last_index = (*vec)[i-1];
    if (std::abs(int(c)) < 125) {
      index.n = last_index.n;
      index.t = last_index.t + c;
      index.x = last_index.x;
    } else {
      if (c != 127)
        KALDI_ERR << "Unexpected character " << c
                  << " encountered while reading Index vector.";
      ReadBasicType(is, binary, &(index.n));
      ReadBasicType(is, binary, &(index.t));
      ReadBasicType(is, binary, &(index.x));
    }
  }
}

void WriteIndexVector(std::ostream &os, bool binary,
                      const std::vector<Index> &vec) {
  // This token will make it easier to write back-compatible code if we later
  // change the format.
  WriteToken(os, binary, "<I1V>");
  int32 size = vec.size();
  WriteBasicType(os, binary, size);
  if (!binary) {  // In text mode we just use the native Write functionality.
    for (int32 i = 0; i < size; i++)
      vec[i].Write(os, binary);
  } else {
    for (int32 i = 0; i < size; i++)
      WriteIndexVectorElementBinary(os, vec, i);
  }
}


void ReadIndexVector(std::istream &is, bool binary,
                     std::vector<Index> *vec) {
  ExpectToken(is, binary, "<I1V>");
  int32 size;
  ReadBasicType(is, binary, &size);
  if (size < 0) {
    KALDI_ERR << "Error reading Index vector: size = "
              << size;
  }
  vec->resize(size);
  if (!binary) {
    for (int32 i = 0; i < size; i++)
      (*vec)[i].Read(is, binary);
  } else {
    for (int32 i = 0; i < size; i++)
      ReadIndexVectorElementBinary(is, i, vec);
  }
}



size_t IndexHasher::operator () (const Index &index) const {
  // The numbers that appear below were chosen arbitrarily from a list of primes
  return index.n +
      1619 * index.t +
      15649 * index.x;
}

size_t CindexHasher::operator () (const Cindex &cindex) const {
  // The numbers that appear below were chosen arbitrarily from a list of primes
  return cindex.first +
       1619 * cindex.second.n +
      15649 * cindex.second.t +
      89809 * cindex.second.x;

}

std::ostream &operator << (std::ostream &ostream, const Index &index) {
  return ostream << '(' << index.n << ' ' << index.t << ' ' << index.x << ')';
}

std::ostream &operator << (std::ostream &ostream, const Cindex &cindex) {
  return ostream << '(' << cindex.first << ' ' << cindex.second << ')';
}

void PrintCindex(std::ostream &os, const Cindex &cindex,
                 const std::vector<std::string> &node_names) {
  KALDI_ASSERT(static_cast<size_t>(cindex.first) < node_names.size());
  os << node_names[cindex.first] << "(" << cindex.second.n << ","
     << cindex.second.t;
  if (cindex.second.x != 0)
    os << "," << cindex.second.x;
  os << ")";
}

void PrintIndexes(std::ostream &os,
                  const std::vector<Index> &indexes) {
  if (indexes.empty()) {
    os << "[ ]";
    return;
  }
  // range_starts will be the starts of ranges (with consecutive t values and
  // the same n value and zero x values) that we compactly print.  we'll append
  // "end" to range_starts for convenience.n
  std::vector<int32> range_starts;
  int32 cur_start = 0, end = indexes.size();
  for (int32 i = cur_start; i < end; i++) {
    const Index &index = indexes[i];
    if (i > cur_start &&
        (index.t != indexes[i-1].t + 1 ||
         index.n != indexes[i-1].n ||
         index.x != indexes[i-1].x)) {
      range_starts.push_back(cur_start);
      cur_start = i;
    }
  }
  range_starts.push_back(cur_start);
  range_starts.push_back(end);
  os << "[";
  int32 num_ranges = range_starts.size() - 1;
  for (int32 r = 0; r < num_ranges; r++) {
    int32 range_start = range_starts[r], range_end = range_starts[r+1];
    KALDI_ASSERT(range_end > range_start);
    os << "(" << indexes[range_start].n << ",";
    if (range_end == range_start + 1)
      os << indexes[range_start].t;
    else
      os << indexes[range_start].t << ":" << indexes[range_end - 1].t;
    if (indexes[range_start].x != 0)
      os << "," << indexes[range_start].x;
    os << ")";
    if (r + 1 < num_ranges)
      os << ", ";
  }
  os << "]";
}

void PrintCindexes(std::ostream &ostream,
                   const std::vector<Cindex> &cindexes,
                   const std::vector<std::string> &node_names) {
  int32 num_cindexes = cindexes.size();
  if (num_cindexes == 0) {
    ostream << "[ ]";
    return;
  }
  int32 cur_offset = 0;
  std::vector<Index> indexes;
  indexes.reserve(cindexes.size());
  while (cur_offset < num_cindexes) {
    int32 cur_node_index = cindexes[cur_offset].first;
    while (cur_offset < num_cindexes &&
           cindexes[cur_offset].first == cur_node_index) {
      indexes.push_back(cindexes[cur_offset].second);
      cur_offset++;
    }
    KALDI_ASSERT(static_cast<size_t>(cur_node_index) < node_names.size());
    const std::string &node_name = node_names[cur_node_index];
    ostream << node_name;
    PrintIndexes(ostream, indexes);
    indexes.clear();
  }
}


void PrintIntegerVector(std::ostream &os,
                        const std::vector<int32> &ints) {
  if (ints.empty()) {
    os << "[ ]";
    return;
  }
  // range_starts will be the starts of ranges (with consecutive or identical
  // values) that we compactly print.  we'll append "end" to range_starts for
  // convenience.
  std::vector<int32> range_starts;
  int32 cur_start = 0, end = ints.size();
  for (int32 i = cur_start; i < end; i++) {
    if (i > cur_start) {
      int32 range_start_val = ints[cur_start],
          range_start_plus_one_val = ints[cur_start+1],
          cur_val = ints[i];
      // if we have reached the end of a range...
      if (!((range_start_plus_one_val == range_start_val &&
             cur_val == range_start_val) ||
            (range_start_plus_one_val == range_start_val + 1 &&
             cur_val == range_start_val + i - cur_start))) {
        range_starts.push_back(cur_start);
        cur_start = i;
      }
    }
  }
  range_starts.push_back(cur_start);
  range_starts.push_back(end);
  os << "[";
  int32 num_ranges = range_starts.size() - 1;
  for (int32 r = 0; r < num_ranges; r++) {
    int32 range_start = range_starts[r], range_end = range_starts[r+1];
    KALDI_ASSERT(range_end > range_start);
    if (range_end == range_start + 1)
      os << ints[range_start];
    else if (range_end == range_start + 2)  // don't print ranges of 2.
      os << ints[range_start] << ", " << ints[range_start+1];
    else if (ints[range_start] == ints[range_start+1])
      os << ints[range_start] << "x" << (range_end - range_start);
    else
      os << ints[range_start] << ":" << ints[range_end - 1];
    if (r + 1 < num_ranges)
      os << ", ";
  }
  os << "]";
}



} // namespace nnet3
} // namespace kaldi
