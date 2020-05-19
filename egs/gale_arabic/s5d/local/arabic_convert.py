#!/usr/bin/env python3

import sys

def hex_to_decimal(utf8_string):
    assert(len(utf8_string) == 3)
    hex_dict = {}
    char_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"]
    value_list = [0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    for key, value in zip (char_list, value_list):
        hex_dict[key] = value

    result = 0
    length = len(utf8_string)
    for i in range(length):
        digit = utf8_string[length - 1 - i]
        result += hex_dict[digit] * (16 ** i)

    return result

def get_unicode_dict():
    unicode_dict = {}
    utf8_list = [("621", "'"), ("622", "|"),("623", ">"),
                 ("624", "&"), ("625", "<"),("626", "}"),
                 ("627", "A"), ("628", "b"),("629", "p"),
                 ("62A", "t"), ("62B", "v"),("62C", "j"),
                 ("62D", "H"), ("62E", "x"),("62F", "d"),
                 ("630", "*"), ("631", "r"),("632", "z"),
                 ("633", "s"), ("634", "$"),("635", "S"),
                 ("636", "D"), ("637", "T"),("638", "Z"),
                 ("639", "E"), ("63A", "g"),("640", "_"),
                 ("641", "f"), ("642", "q"),("643", "k"),
                 ("644", "l"), ("645", "m"),("646", "n"),
                 ("647", "h"), ("648", "w"),("649", "Y"),
                 ("64A", "y"), ("64B", "F"),("64C", "N"),
                 ("64D", "K"), ("64E", "a"),("64F", "u"),
                 ("650", "i"), ("651", "~"),("652", "o"),
                 ("670", "`"), ("671", "{"),("67E", "P"),
                 ("686", "J"), ("6A4", "V"),("6AF", "G")]

    for word_pair in utf8_list:
        utf8 = word_pair[0]
        char = word_pair[1]
        unicode_dict[hex_to_decimal(utf8)] = char

    return unicode_dict
    

def convert(word, unicode_dict):
    word_list = []
    for char in word:
        c_unicode = ord(char)
        if c_unicode in unicode_dict:
            word_list.append(unicode_dict[c_unicode])

    return "".join(word_list)

def process_arabic_text(arabic_text, unicode_dict):
    with open(arabic_text, 'r') as file:
        sentence_list = []
        is_sentence = False
        for line in file.readlines():
#print(line.split()[0], is_sentence, line.split()[0] == "</P>")
            if len(line.split()) > 0:
                if line.split()[0] == "<P>":
                    is_sentence = True

                elif (is_sentence and line.split()[0] != "</P>"):
                    for word in line.split():
                        if word == '.':
                            # when meet period ".", sentence_list should not be empty (do find sentence ending with two period)
                            if (len(sentence_list) > 0):                
                                sentence = " ".join(sentence_list)
                                print(sentence)
                            sentence_list = []
                        elif word[-1] == ".":
                            word = word[:-1]
                            sentence_list.append(word)
                            sentence = " ".join(sentence_list)
                            print(sentence)
                            sentence_list = []
                        else:
                            word = word
                            if word != '':
                                sentence_list.append(word)
    
                if line.split()[0] == "</P>":
                    is_sentence = False
                    if (len(sentence_list) > 0):
                        print(" ".join(sentence_list)) 
                        sentence_list = []
                
                

def main():
    arabic_text = sys.argv[1]
    unicode_dict = get_unicode_dict()
    process_arabic_text(arabic_text, unicode_dict)

if __name__ == "__main__":
    main()
