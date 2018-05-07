#include <iostream>
#include <fstream>
#include <sstream>

#include "gpufst/numberizer.h"

namespace gpufst{

numberizer read_numberizer(const std::string &filename) {
  std::ifstream file(filename);
  std::string line;
  numberizer nr;
  while (getline(file, line)) {
    std::istringstream iss(line);
    std::string word;
    int num;
    iss >> word >> num;
    nr.add(word, num);
  }
  return nr;
}

std::vector<int> numberizer::split(const std::string &line) const {
  std::istringstream iss(line);
  std::string word;
  std::vector<int> nums;
  while (iss >> word)
    nums.push_back(word_to_num(word));
  return nums;
}

std::string numberizer::join(const std::vector<int> &nums) const {
  std::ostringstream oss;
  for (int i=0; i < (int)nums.size(); i++) {
    if (i > 0)
      oss << " ";
    oss << num_to_word(nums[i]);
  }
  return oss.str();
}

}