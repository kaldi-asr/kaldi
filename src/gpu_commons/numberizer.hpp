#ifndef NUMBERIZER_HPP
#define NUMBERIZER_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

class numberizer {
public:
  numberizer() : _size(0) { }
  int word_to_num (const std::string &w) const { return _word_to_num.at(w); }
  const std::string &num_to_word (int i) const { return _num_to_word.at(i); }
  int size () const { return _size; }
  void add (const std::string &w, int i) {
    _num_to_word[i] = w;
    _word_to_num[w] = i;
    _size = std::max(_size, i+1);
  }
  std::vector<int> split (const std::string &line) const;
  std::string join (const std::vector<int> &nums) const;
private:
  std::unordered_map<int, std::string> _num_to_word;
  std::unordered_map<std::string, int> _word_to_num;
  int _size;
};

numberizer read_numberizer(const std::string &filename);

#endif

