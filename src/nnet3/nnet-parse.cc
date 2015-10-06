// nnet3/nnet-parse.cc

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

namespace kaldi {
namespace nnet3 {

bool ConfigLine::ParseLine(const std::string &line) {
  if (line.size() == 0) return false;   // Empty line

  // Line ends or begins with space -> remove it and recurse.
  if (isspace(line[line.size()-1]) || isspace(line[0])) {
    size_t initial_pos = line.find_first_not_of(" \t\r\n"),
        final_pos = line.find_last_not_of(" \t\r\n");
    if (initial_pos == std::string::npos || final_pos <= initial_pos)
      return false;
    std::string processed_line(line, initial_pos, final_pos - initial_pos + 1);
    return ParseLine(processed_line);
  }

  size_t pos = 0;
  size_t found_eq = line.find_first_of("=", pos + 1);
  if (found_eq == std::string::npos) return false; // Could not find '='


  while (found_eq < line.size()) {
    std::string key(line, pos, found_eq - pos);
    if (!IsValidName(key)) return false;
    if (found_eq == std::string::npos) return false; // Could not find '='
    if (found_eq == line.size() - 1 || line[found_eq+1] == ' ' || line[found_eq+1] == '\t') {
      // Empty value for key
      data_.insert(std::make_pair(key, std::make_pair("", false)));
      pos = line.find_first_not_of(" \t", found_eq + 1);
      if (pos == std::string::npos)
        break; // Done reading
      found_eq = line.find_first_of("=", pos + 1);
      continue;
    }

    // See if there is next key
    size_t found = line.find_first_of("=", found_eq + 1);
    size_t value_end = std::string::npos;

    if (found != std::string::npos) {
      size_t found_ws = line.find_last_of(" \t", found);
      if (found_ws < found_eq + 1) found_ws = found;

      value_end = line.find_last_not_of(" \t", found_ws);
      pos = line.find_first_not_of(" \t", found_ws + 1);
    } else {
      value_end = line.find_last_not_of(" \t", found);
    }

    KALDI_ASSERT(value_end > found_eq);

    std::string value(line, found_eq + 1, value_end - found_eq);

    if (value[0] == ' ' || value[0] == '\t') return false;
    data_.insert(std::make_pair(key, std::make_pair(value, false)));

    found_eq = found;
  }
  whole_line_ = line;
  return true;
}

bool ConfigLine::GetValue(const std::string &key, std::string *value) {
  KALDI_ASSERT(value != NULL);
  value->clear();
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
        int32 tmp;
        if (!IsValidName(str) && !ConvertStringToInteger(str, &tmp)) {
          KALDI_WARN << "Could not parse line " << ErrorContext(std::string(input, start));
          return false;
        }
        tokens->push_back(str);
        break;
      } else {
        if (input[found] == '(' || input[found] == ')' || input[found] == ',') {
          std::string str(input, start, found - start);
          int32 tmp;
          if (!IsValidName(str) && !ConvertStringToInteger(str, &tmp)) {
            KALDI_WARN << "Could not parse line " << ErrorContext(std::string(input, start));
            return false;
          }
          tokens->push_back(str);
          start = found;
        } else {
          std::string str(input, start, found - start);
          int32 tmp;
          if (!IsValidName(str) && !ConvertStringToInteger(str, &tmp)) {
            KALDI_WARN << "Could not parse line " << ErrorContext(std::string(input, start));
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
    if (!isalnum(name[i]) && name[i] != '_' && name[i] != '-')
      return false;
  }
  return true;
}

void ReadConfigFile(std::istream &is,
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
  } else if (fabs(f) >= 1.0) {
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
std::string SummarizeVector(const Vector<BaseFloat> &vec) {
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
    os << "[ " << std::setprecision(2);
    for (int32 i = 0; i < 8; i++) {
      PrintFloatSuccinctly(os, vec(i));
      os << ' ';
    }
    os << "... ";
    os << std::setprecision(3);
    os << "(mean=" << mean << ", stddev=" << stddev << ")]";
  }
  return os.str();
}



} // namespace nnet3
} // namespace kaldi
