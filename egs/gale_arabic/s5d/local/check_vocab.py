#!/usr/bin/env python3

import sys

def get_vocab_set(ref_file):
    vocab_set = set()
    with open(ref_file, 'r') as f:
        for line in f.readlines():
            word = line.split()[0]
            vocab_set.add(word)
    return vocab_set
            

def compare(vocab_set, wordlist):
    with open(wordlist, 'r') as f:
        for line in f.readlines():
            word = line.split()[0]
            if word not in vocab_set:
                print(word)

def main():
    ref_file = sys.argv[1]
    wordlist = sys.argv[2]
    vocab_set = get_vocab_set(ref_file)
    compare(vocab_set, wordlist)

if __name__ == "__main__":
    main()
