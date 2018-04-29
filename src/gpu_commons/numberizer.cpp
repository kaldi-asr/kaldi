#include <iostream>
#include <fstream>
#include <sstream>

#include "numberizer.hpp"

using namespace std;

numberizer read_numberizer(const string &filename) {
  ifstream file(filename);
  string line;
  numberizer nr;
  while (getline(file, line)) {
    istringstream iss(line);
    string word;
    int num;
    iss >> word >> num;
    nr.add(word, num);
  }
  return nr;
}

std::vector<int> numberizer::split(const std::string &line) const {
  istringstream iss(line);
  string word;
  std::vector<int> nums;
  while (iss >> word)
    nums.push_back(word_to_num(word));
  return nums;
}

std::string numberizer::join(const std::vector<int> &nums) const {
  ostringstream oss;
  for (int i=0; i<nums.size(); i++) {
    if (i > 0)
      oss << " ";
    oss << num_to_word(nums[i]);
  }
  return oss.str();
}
