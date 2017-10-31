// nnet3/nnet-common.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2016  Xiaohui Zhang

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
  const Index &index = vec[i];
  if (i == 0) {
    // we don't use std::abs(index.t) < 125 here because it doesn't have the
    // right (or even well-defined) behavior for
    // index.t == std::numeric_limits<int32>::min().
    if (index.n == 0 && index.x == 0 &&
        index.t > -125 && index.t < 125) {
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
    // we don't do if (std::abs(index.t - last_index.t) < 125)
    // below because this doesn't work right if that difference
    // equals std::numeric_limits<int32>::min().
    if (index.n == last_index.n && index.x == last_index.x &&
        index.t - last_index.t < 125 &&
        index.t - last_index.t > -125) {
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

static void WriteCindexVectorElementBinary(
    std::ostream &os,
    const std::vector<Cindex> &vec,
    int32 i) {
  bool binary = true;
  int32 node_index = vec[i].first;
  const Index &index = vec[i].second;
  if (i == 0 || node_index != vec[i-1].first) {
    // divide using '|' into ranges that each have all the same node name, like:
    // [node_1: index_1 index_2] [node_2: index_3 index_4] Caution: '|' is
    // character 124 so we have to avoid that character in places where it might
    // be confused with this separator.
    os.put('|');
    WriteBasicType(os, binary, node_index);
  }
  if (i == 0) {
    // we don't need to be concerned about reserving space for character 124
    // ('|') here, since (wastefully) '|' is always printed for i == 0.
    //
    // we don't use std::abs(index.t) < 125 here because it doesn't have the
    // right (or even well-defined) behavior for
    // index.t == std::numeric_limits<int32>::min().
    if (index.n == 0 && index.x == 0 &&
        index.t > -125 && index.t < 125) {
      // handle this common case in one character.
      os.put(static_cast<signed char>(index.t));
    } else if (index.t == 0 && index.x == 0 &&
               (index.n == 0 || index.n == 1)) {
      // handle this common case in one character.
      os.put(static_cast<signed char>(index.n + 125));
    } else {  // handle the general case less efficiently.
      os.put(127);
      WriteBasicType(os, binary, index.n);
      WriteBasicType(os, binary, index.t);
      WriteBasicType(os, binary, index.x);
    }
  } else {
    const Index &last_index = vec[i-1].second;
    // we don't do if std::abs(index.t - last_index.t) < 124
    // below because it doesn't work right if the difference
    // equals std::numeric_limits<int32>::min().
    if (index.n == last_index.n && index.x == last_index.x &&
        index.t - last_index.t < 124 &&
        index.t - last_index.t > -124) {
      signed char c = index.t - last_index.t;
      os.put(c);
      // note: we have to reserve character 124 ('|') for when 'n' or 'x'
      // changes.
    } else if (index.t == last_index.t && index.x == last_index.x &&
              (index.n == last_index.n || index.n == last_index.n + 1)) {
      os.put(125 + index.n - last_index.n);
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

static void ReadCindexVectorElementBinary(
    std::istream &is,
    int32 i,
    std::vector<Cindex> *vec) {
  bool binary = true;
  Index &index = (*vec)[i].second;
  if (!is.good())
    KALDI_ERR << "End of file while reading vector of Cindex.";
  if (is.peek() == static_cast<int>('|')) {
    is.get();
    ReadBasicType(is, binary, &((*vec)[i].first));
  } else {
    KALDI_ASSERT(i != 0);
    (*vec)[i].first = (*vec)[i-1].first;
  }
  signed char c = is.get();
  if (i == 0) {
    if (std::abs(int(c)) < 125) {
      index.n = 0;
      index.t = c;
      index.x = 0;
    } else if (c == 125 || c == 126) {
      index.n = c - 125;
      index.t = 0;
      index.x = 0;
    } else {
      if (c != 127)
        KALDI_ERR << "Unexpected character " << c
                  << " encountered while reading Cindex vector.";
      ReadBasicType(is, binary, &(index.n));
      ReadBasicType(is, binary, &(index.t));
      ReadBasicType(is, binary, &(index.x));
    }
  } else {
    Index &last_index = (*vec)[i-1].second;
    if (std::abs(int(c)) < 124) {
      index.n = last_index.n;
      index.t = last_index.t + c;
      index.x = last_index.x;
    } else if (c == 125 || c == 126) {
      index.n = last_index.n + c - 125;
      index.t = last_index.t;
      index.x = last_index.x;
    } else {
      if (c != 127)
        KALDI_ERR << "Unexpected character " << c
                  << " encountered while reading Cindex vector.";
      ReadBasicType(is, binary, &(index.n));
      ReadBasicType(is, binary, &(index.t));
      ReadBasicType(is, binary, &(index.x));
    }
  }
}

// This function writes elements of a Cindex vector in a compact form.
// which is similar as the output of PrintCindexes. The vector is divided
// into ranges that each have all the same node name, like:
// [node_1: index_1 index_2] [node_2: index_3 index_4]
void WriteCindexVector(std::ostream &os, bool binary,
                       const std::vector<Cindex> &vec) {
  // This token will make it easier to write back-compatible code if we later
  // change the format.
  WriteToken(os, binary, "<I1V>");
  int32 size = vec.size();
  WriteBasicType(os, binary, size);
  if (!binary) {  // In text mode we just use the native Write functionality.
    for (int32 i = 0; i < size; i++) {
      int32 node_index = vec[i].first;
      if (i == 0 || node_index != vec[i-1].first) {
        if (i > 0)
          os.put(']');
        os.put('[');
        WriteBasicType(os, binary, node_index);
        os.put(':');
      }
      vec[i].second.Write(os, binary);
      if (i == size - 1)
        os.put(']');
    }
  } else {
    for (int32 i = 0; i < size; i++)
      WriteCindexVectorElementBinary(os, vec, i);
  }
}

void ReadCindexVector(std::istream &is, bool binary,
                      std::vector<Cindex> *vec) {
  ExpectToken(is, binary, "<I1V>");
  int32 size;
  ReadBasicType(is, binary, &size);
  if (size < 0) {
    KALDI_ERR << "Error reading Index vector: size = "
              << size;
  }
  vec->resize(size);
  if (!binary) {
    for (int32 i = 0; i < size; i++) {
      is >> std::ws;
      if (is.peek() == static_cast<int>(']') || i == 0) {
        if (i != 0)
          is.get();
        is >> std::ws;
        if (is.peek() == static_cast<int>('[')) {
          is.get();
        } else {
          KALDI_ERR << "ReadCintegerVector: expected to see [, saw "
                    << is.peek() << ", at file position " << is.tellg();
        }
        ReadBasicType(is, binary, &((*vec)[i].first));
        is >> std::ws;
        if (is.peek() == static_cast<int>(':')) {
          is.get();
        } else {
          KALDI_ERR << "ReadCintegerVector: expected to see :, saw "
                    << is.peek() << ", at file position " << is.tellg();
        }
      } else {
        (*vec)[i].first = (*vec)[i-1].first;
      }
      (*vec)[i].second.Read(is, binary);
      if (i == size - 1) {
        is >> std::ws;
        if (is.peek() == static_cast<int>(']')) {
          is.get();
        } else {
          KALDI_ERR << "ReadCintegerVector: expected to see ], saw "
                    << is.peek() << ", at file position " << is.tellg();
        }
      }
    }
  } else {
    for (int32 i = 0; i < size; i++)
      ReadCindexVectorElementBinary(is, i, vec);
  }
}

size_t IndexHasher::operator () (const Index &index) const noexcept {
  // The numbers that appear below were chosen arbitrarily from a list of primes
  return index.n +
      1619 * index.t +
      15649 * index.x;
}

size_t CindexHasher::operator () (const Cindex &cindex) const noexcept {
  // The numbers that appear below were chosen arbitrarily from a list of primes
  return cindex.first +
       1619 * cindex.second.n +
      15649 * cindex.second.t +
      89809 * cindex.second.x;

}

size_t CindexVectorHasher::operator () (
    const std::vector<Cindex> &cindex_vector) const noexcept {
  // this is an arbitrarily chosen prime.
  size_t kPrime = 23539, ans = 0;
  std::vector<Cindex>::const_iterator iter = cindex_vector.begin(),
      end = cindex_vector.end();
  CindexHasher cindex_hasher;
  for (; iter != end; ++iter)
    ans = cindex_hasher(*iter) + kPrime * ans;
  return ans;
}

size_t IndexVectorHasher::operator () (
    const std::vector<Index> &index_vector) const noexcept {
  size_t n1 = 15, n2 = 10;  // n1 and n2 are used to extract only a subset of
                            // elements to hash; this makes the hasher faster by
                            // skipping over more elements.  Setting n1 large or
                            // n2 to 1 would make the hasher consider all
                            // elements.
  // all long-ish numbers appearing below are randomly chosen primes.
  size_t ans = 1433 + 34949  * index_vector.size();
  std::vector<Index>::const_iterator iter = index_vector.begin(),
      end = index_vector.end(), med = end;
  if (med > iter + n1)
    med = iter + n1;

  for (; iter != med; ++iter) {
    ans += iter->n * 1619;
    ans += iter->t * 15649;
    ans += iter->x * 89809;
  }
  // after the first n1 values, look only at every n2'th value.  this makes the
  // hashing much faster, and in the kinds of structures that we actually deal
  // with, we shouldn't get unnecessary hash collisions as a result of this
  // optimization.
  for (; iter < end; iter += n2) {
    ans += iter->n * 1619;
    ans += iter->t * 15649;
    ans += iter->x * 89809;
  }
  return ans;
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

// this will be the most negative number representable as int32.
const int kNoTime = std::numeric_limits<int32>::min();

} // namespace nnet3
} // namespace kaldi
