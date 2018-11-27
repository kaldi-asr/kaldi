// nnet3/nnet-parse.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {
namespace nnet3 {


bool ConfigLine::ParseLine(const std::string &line) {
  data_.clear();
  whole_line_ = line;
  if (line.size() == 0) return false;   // Empty line
  size_t pos = 0, size = line.size();
  while (isspace(line[pos]) && pos < size) pos++;
  if (pos == size)
    return false;  // whitespace-only line
  size_t first_token_start_pos = pos;
  // first get first_token_.
  while (!isspace(line[pos]) && pos < size) {
    if (line[pos] == '=') {
      // If the first block of non-whitespace looks like "foo-bar=...",
      // then we ignore it: there is no initial token, and FirstToken()
      // is empty.
      pos = first_token_start_pos;
      break;
    }
    pos++;
  }
  first_token_ = std::string(line, first_token_start_pos, pos - first_token_start_pos);
  // first_token_ is expected to be either empty or something like
  // "component-node", which actually is a slightly more restrictive set of
  // strings than IsValidName() checks for this is a convenient way to check it.
  if (!first_token_.empty() && !IsValidName(first_token_))
    return false;

  while (pos < size) {
    if (isspace(line[pos])) {
      pos++;
      continue;
    }

    // OK, at this point we know that we are pointing at nonspace.
    size_t next_equals_sign = line.find_first_of("=", pos);
    if (next_equals_sign == pos || next_equals_sign == std::string::npos) {
      // we're looking for something like 'key=value'.  If there is no equals sign,
      // or it's not preceded by something, it's a parsing failure.
      return false;
    }
    std::string key(line, pos, next_equals_sign - pos);
    if (!IsValidName(key)) return false;

    // handle any quotes.  we support key='blah blah' or key="foo bar".
    // no escaping is supported.
    if (line[next_equals_sign+1] == '\'' || line[next_equals_sign+1] == '"') {
      char my_quote = line[next_equals_sign+1];
      size_t next_quote = line.find_first_of(my_quote, next_equals_sign + 2);
      if (next_quote == std::string::npos) {  // no matching quote was found.
        KALDI_WARN << "No matching quote for " << my_quote << " in config line '"
                   << line << "'";
        return false;
      } else {
        std::string value(line, next_equals_sign + 2,
                          next_quote - next_equals_sign - 2);
        data_.insert(std::make_pair(key, std::make_pair(value, false)));
        pos = next_quote + 1;
        continue;
      }
    } else {
      // we want to be able to parse something like "... input=Offset(a, -1) foo=bar":
      // in general, config values with spaces in them, even without quoting.

      size_t next_next_equals_sign = line.find_first_of("=", next_equals_sign + 1),
          terminating_space = size;

      if (next_next_equals_sign != std::string::npos) {  // found a later equals sign.
        size_t preceding_space = line.find_last_of(" \t", next_next_equals_sign);
        if (preceding_space != std::string::npos &&
            preceding_space > next_equals_sign)
          terminating_space = preceding_space;
      }
      while (isspace(line[terminating_space - 1]) && terminating_space > 0)
        terminating_space--;

      std::string value(line, next_equals_sign + 1,
                        terminating_space - (next_equals_sign + 1));
      data_.insert(std::make_pair(key, std::make_pair(value, false)));
      pos = terminating_space;
    }
  }
  return true;
}

bool ConfigLine::GetValue(const std::string &key, std::string *value) {
  KALDI_ASSERT(value != NULL);
  std::map<std::string, std::pair<std::string, bool> >::iterator it = data_.begin();
  for (; it != data_.end(); ++it) {
    if (it->first == key) {
      *value = (it->second).first;
      (it->second).second = true;
      return true;
    }
  }
  return false;
}

bool ConfigLine::GetValue(const std::string &key, BaseFloat *value) {
  KALDI_ASSERT(value != NULL);
  std::map<std::string, std::pair<std::string, bool> >::iterator it = data_.begin();
  for (; it != data_.end(); ++it) {
    if (it->first == key) {
      if (!ConvertStringToReal((it->second).first, value))
        return false;
      (it->second).second = true;
      return true;
    }
  }
  return false;
}

bool ConfigLine::GetValue(const std::string &key, int32 *value) {
  KALDI_ASSERT(value != NULL);
  std::map<std::string, std::pair<std::string, bool> >::iterator it = data_.begin();
  for (; it != data_.end(); ++it) {
    if (it->first == key) {
      if (!ConvertStringToInteger((it->second).first, value))
        return false;
      (it->second).second = true;
      return true;
    }
  }
  return false;
}

bool ConfigLine::GetValue(const std::string &key, std::vector<int32> *value) {
  KALDI_ASSERT(value != NULL);
  value->clear();
  std::map<std::string, std::pair<std::string, bool> >::iterator it = data_.begin();
  for (; it != data_.end(); ++it) {
    if (it->first == key) {
      if (!SplitStringToIntegers((it->second).first, ":,", true, value)) {
        // KALDI_WARN << "Bad option " << (it->second).first;
        return false;
      }
      (it->second).second = true;
      return true;
    }
  }
  return false;
}

bool ConfigLine::GetValue(const std::string &key, bool *value) {
  KALDI_ASSERT(value != NULL);
  std::map<std::string, std::pair<std::string, bool> >::iterator it = data_.begin();
  for (; it != data_.end(); ++it) {
    if (it->first == key) {
      if ((it->second).first.size() == 0) return false;
      switch (((it->second).first)[0]) {
        case 'F':
        case 'f':
          *value = false;
          break;
        case 'T':
        case 't':
          *value = true;
          break;
        default:
          return false;
      }
      (it->second).second = true;
      return true;
    }
  }
  return false;
}

bool ConfigLine::HasUnusedValues() const {
  std::map<std::string, std::pair<std::string, bool> >::const_iterator it = data_.begin();
  for (; it != data_.end(); ++it) {
    if (!(it->second).second) return true;
  }
  return false;
}

std::string ConfigLine::UnusedValues() const {
  std::string unused_str;
  std::map<std::string, std::pair<std::string, bool> >::const_iterator it = data_.begin();
  for (; it != data_.end(); ++it) {
    if (!(it->second).second) {
      if (unused_str == "")
        unused_str = it->first + "=" + (it->second).first;
      else
        unused_str += " " + it->first + "=" + (it->second).first;
    }
  }
  return unused_str;
}

// This is like ExpectToken but for two tokens, and it
// will either accept token1 and then token2, or just token2.
// This is useful in Read functions where the first token
// may already have been consumed.
void ExpectOneOrTwoTokens(std::istream &is, bool binary,
                          const std::string &token1,
                          const std::string &token2) {
  KALDI_ASSERT(token1 != token2);
  std::string temp;
  ReadToken(is, binary, &temp);
  if (temp == token1) {
    ExpectToken(is, binary, token2);
  } else {
    if (temp != token2) {
      KALDI_ERR << "Expecting token " << token1 << " or " << token2
                << " but got " << temp;
    }
  }
}

// static
bool ParseFromString(const std::string &name, std::string *string,
                     int32 *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();

  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!ConvertStringToInteger(split_string[i].substr(len), param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     bool *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();

  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      std::string b = split_string[i].substr(len);
      if (b.empty())
        KALDI_ERR << "Bad option " << split_string[i];
      if (b[0] == 'f' || b[0] == 'F') *param = false;
      else if (b[0] == 't' || b[0] == 'T') *param = true;
      else
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     BaseFloat *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();

  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!ConvertStringToReal(split_string[i].substr(len), param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     std::string *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();

  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      *param = split_string[i].substr(len);

      // Set "string" to all the pieces but the one we used.
      *string = "";
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     std::vector<int32> *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();

  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!SplitStringToIntegers(split_string[i].substr(len), ":,",
                                 false, param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool DescriptorTokenize(const std::string &input,
                        std::vector<std::string> *tokens) {
  KALDI_ASSERT(tokens != NULL);
  size_t start = input.find_first_not_of(" \t"), size = input.size();
  tokens->clear();
  while (start < size) {
    KALDI_ASSERT(!isspace(input[start]));
    if (input[start] == '(' || input[start] == ')' || input[start] == ',') {
      tokens->push_back(std::string(input, start, 1));
      start = input.find_first_not_of(" \t", start + 1);
    } else {
      size_t found = input.find_first_of(" \t(),", start);
      KALDI_ASSERT(found != start);
      if (found == std::string::npos) {
        std::string str(input, start, input.size() - start);
        BaseFloat tmp;
        if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
          KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
          return false;
        }
        tokens->push_back(str);
        break;
      } else {
        if (input[found] == '(' || input[found] == ')' || input[found] == ',') {
          std::string str(input, start, found - start);
          BaseFloat tmp;
          if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
            KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
            return false;
          }
          tokens->push_back(str);
          start = found;
        } else {
          std::string str(input, start, found - start);
          BaseFloat tmp;
          if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
            KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
            return false;
          }
          tokens->push_back(str);
          start = input.find_first_not_of(" \t", found);
        }
      }
    }
  }
  return true;
}

bool IsValidName(const std::string &name) {
  if (name.size() == 0) return false;
  for (size_t i = 0; i < name.size(); i++) {
    if (i == 0 && !isalpha(name[i]) && name[i] != '_')
      return false;
    if (!isalnum(name[i]) && name[i] != '_' && name[i] != '-' && name[i] != '.')
      return false;
  }
  return true;
}

void ReadConfigLines(std::istream &is,
                    std::vector<std::string> *lines) {
  KALDI_ASSERT(lines != NULL);
  std::string line;
  while (std::getline(is, line)) {
    if (line.size() == 0) continue;
    size_t start = line.find_first_not_of(" \t");
    size_t end = line.find_first_of('#');
    if (start == std::string::npos || start == end) continue;
    end = line.find_last_not_of(" \t", end - 1);
    KALDI_ASSERT(end >= start);
    lines->push_back(line.substr(start, end - start + 1));
  }
}

std::string ErrorContext(std::istream &is) {
  if (!is.good()) return "end of line";
  char buf[21];
  is.read(buf, 21);
  if (is) {
    return (std::string(buf, 20) + "...");
  }
  return std::string(buf, is.gcount());
}

std::string ErrorContext(const std::string &str) {
  if (str.size() == 0) return "end of line";
  if (str.size() <= 20) return str;
  return std::string(str, 0, 20) + "...";
}

static void PrintFloatSuccinctly(std::ostream &os, BaseFloat f) {
  if (fabs(f) < 10000.0 && fabs(f) >= 10.0) {
    os  << std::fixed << std::setprecision(0) << f;
  } else if (fabs(f) >= 0.995) {
    os  << std::fixed << std::setprecision(1) << f;
  } else if (fabs(f) >= 0.01) {
    os  << std::fixed << std::setprecision(2) << f;
  } else {
    os << std::setprecision(1) << f;
  }
  os.unsetf(std::ios_base::floatfield);
  os << std::setprecision(6);  // Restore the default.
}


// Returns a string that summarizes a vector fairly succintly, for
// printing stats in info lines.
std::string SummarizeVector(const VectorBase<float> &vec) {
  std::ostringstream os;
  if (vec.Dim() < 10) {
    os << "[ ";
    for (int32 i = 0; i < vec.Dim(); i++) {
      PrintFloatSuccinctly(os, vec(i));
      os << ' ';
    }
    os << "]";
  } else {
    // print out mean and standard deviation, and some selected values.
    BaseFloat mean = vec.Sum() / vec.Dim(),
        stddev = sqrt(VecVec(vec, vec) / vec.Dim() - mean * mean);

    std::string percentiles_str = "0,1,2,5 10,20,50,80,90 95,98,99,100";
    std::vector<int32> percentiles;
    bool ans = SplitStringToIntegers(percentiles_str, ", ", false,
                                     &percentiles);
    KALDI_ASSERT(ans);
    os << "[percentiles(" << percentiles_str << ")=(";
    Vector<BaseFloat> vec_sorted(vec);
    std::sort(vec_sorted.Data(), vec_sorted.Data() + vec_sorted.Dim());
    int32 n = vec.Dim() - 1;
    for (size_t i = 0; i < percentiles.size(); i++) {
      int32 percentile = percentiles[i];
      BaseFloat value = vec_sorted((n * percentile) / 100);
      PrintFloatSuccinctly(os, value);
      if (i + 1 < percentiles.size())
        os << (i == 3 || i == 8 ? ' ' : ',');
    }
    os << std::setprecision(3);
    os << "), mean=" << mean << ", stddev=" << stddev << "]";
  }
  return os.str();
}

std::string SummarizeVector(const VectorBase<double> &vec) {
  Vector<float> vec_copy(vec);
  return SummarizeVector(vec_copy);
}

std::string SummarizeVector(const CuVectorBase<BaseFloat> &cu_vec) {
  Vector<float> vec(cu_vec);
  return SummarizeVector(vec);
}

void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const CuVectorBase<BaseFloat> &params,
                         bool include_mean) {
  os << std::setprecision(4);
  os << ", " << name << '-';
  if (include_mean) {
    BaseFloat mean = params.Sum() / params.Dim(),
        stddev = std::sqrt(VecVec(params, params) / params.Dim() - mean * mean);
    os << "{mean,stddev}=" << mean << ',' << stddev;
  } else {
    BaseFloat rms = std::sqrt(VecVec(params, params) / params.Dim());
    os << "rms=" << rms;
  }
  os << std::setprecision(6);  // restore the default precision.
}

void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const CuMatrix<BaseFloat> &params,
                         bool include_mean,
                         bool include_row_norms,
                         bool include_column_norms,
                         bool include_singular_values) {
  os << std::setprecision(4);
  os << ", " << name << '-';
  int32 dim = params.NumRows() * params.NumCols();
  if (include_mean) {
    BaseFloat mean = params.Sum() / dim,
        stddev = std::sqrt(TraceMatMat(params, params, kTrans) / dim -
                           mean * mean);
    os << "{mean,stddev}=" << mean << ',' << stddev;
  } else {
    BaseFloat rms = std::sqrt(TraceMatMat(params, params, kTrans) / dim);
    os << "rms=" << rms;
  }
  os << std::setprecision(6);  // restore the default precision.

  if (include_row_norms) {
    CuVector<BaseFloat> row_norms(params.NumRows());
    row_norms.AddDiagMat2(1.0, params, kNoTrans, 0.0);
    row_norms.ApplyPow(0.5);
    Vector<BaseFloat> row_norms_cpu;
    row_norms.Swap(&row_norms_cpu);
    os << ", " << name << "-row-norms="
       << SummarizeVector(row_norms_cpu);
  }
  if (include_column_norms) {
    CuVector<BaseFloat> col_norms(params.NumCols());
    col_norms.AddDiagMat2(1.0, params, kTrans, 0.0);
    col_norms.ApplyPow(0.5);
    Vector<BaseFloat> col_norms_cpu;
    col_norms.Swap(&col_norms_cpu);
    os << ", " << name << "-col-norms="
       << SummarizeVector(col_norms_cpu);
  }
  if (include_singular_values) {
    Matrix<BaseFloat> params_cpu(params);
    Vector<BaseFloat> s(std::min(params.NumRows(), params.NumCols()));
    params_cpu.Svd(&s);
    std::string singular_values_str = SummarizeVector(s);
    os << ", " << name << "-singular-values=" << singular_values_str;
    std::ostringstream name_os;
  }
}


void ParseConfigLines(const std::vector<std::string> &lines,
                      std::vector<ConfigLine> *config_lines) {
  config_lines->resize(lines.size());
  for (size_t i = 0; i < lines.size(); i++) {
    bool ret = (*config_lines)[i].ParseLine(lines[i]);
    if (!ret) {
      KALDI_ERR << "Error parsing config line: " << lines[i];
    }
  }
}

bool NameMatchesPattern(const char *name, const char *pattern) {
  if (*pattern == '*') {
    return NameMatchesPattern(name, pattern + 1) ||
        (*name != '\0' && NameMatchesPattern(name + 1, pattern));
  } else if (*name == *pattern) {
    return (*name == '\0' || NameMatchesPattern(name + 1, pattern + 1));
  } else {
    return false;
  }
}



} // namespace nnet3
} // namespace kaldi
