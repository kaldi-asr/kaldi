#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# Copyright 2014 Author: Andreas Kirkedal

# Licensed under the Apache License, Version 2.0 (the "License");                                                    
# you may not use this file except in compliance with the License.                                                  
# You may obtain a copy of the License at                                                                          
#                                                                                                                 
#  http://www.apache.org/licenses/LICENSE-2.0                                                                    
#                                                                                                               
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY                                 
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED                                   
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,                                       
# MERCHANTABLITY OR NON-INFRINGEMENT.                                                                                 
# See the Apache 2 License for the specific language governing permissions and                                       
# limitations under the License.


Writes out numbers between 0 and 100 (ordinals and cardinals) and CPR numbers. 
Currently only has a table for Danish

Changed to write output to file to prevent problems with shell ascii codec.
'''
from __future__ import print_function

import sys
import os
import codecs
from re import split as charsplit


def list2string(wordlist, lim=" ", newline=False):
    '''Converts a list to a string with a delimiter $lim and the possible
    addition of newline characters.'''
    strout = ""
    for w in wordlist:
        strout += w + lim
    if newline:
        return strout.strip(lim) + "\n"
    else:
        return strout.strip(lim)


def loadNumTable(filename):
    '''Loads a table of numbers into a dictionary.'''
    tabSepTable = codecs.open(filename, "r", "utf8")
    numdict = {}
    for line in tabSepTable:
        num, txt = line.split("\t", 1)
        numdict[num] = txt.strip()
    
    return numdict


def get_birth_date(number):
        """Split the date parts from the number and return the birth date."""
        day = int(number[0:2])
        month = int(number[2:4])
        year = int(number[4:6])
        if number[6] in '5678' and year >= 58:
            year += 1800
        elif number[6] in '0123' or (number[6] in '49' and year >= 37):
            year += 1900
        else:
            year += 2000
        try:
            return datetime.date(year, month, day)
        except ValueError:
            return False


def leadingZero(string):
    '''Returns true if the first character is 0'''
    return string[0] == '0'


def onlydigits(string):
    '''Returns only the numbers in the string'''
    return list2string([x for x in string if x.isdigit()], lim="")


def isDKCPR(string):
    '''Checks whether a numeric string is a DKCPR. Only checks length'''
    number = onlydigits(string)
    if len(number) == 10:
        return True
    else:
        return False


def writeZeroDigit(s, k, t):
    '''Converts numeric strings that start with "0" to their spoken form'''
    numbers = [t[x] for x in s if x in k]
    if len(s.strip(".")) == len(numbers):
        return list2string(numbers)
    else:
        return s


def writeOutCPR(s, k, t):
    '''Writes out a DKCPR number to it's spoken form'''
    digits = onlydigits(s)
    parts = [digits[0:2], digits[2:4], digits[4:6], digits[6:8], digits[8:]]
    numbers = []
    for n in parts:
        if n == '': 
            sys.exit('n: ' +n+ '\n' +s)
        if leadingZero(n):
            numbers.append(writeZeroDigit(n, k, t))
        else:
            numbers.append(t[n])
    return list2string(numbers)


def hundreds(string):
    '''Checks if a number is in the hundreds'''
    if len(string) == 3 and string.isnumeric():
        return True
    return False


def writeHundreds(s, t):
    '''Converts Danish numbers in the hundreds to their spoken form'''
    huns = s[0]
    tens = s[1:3]
    if tens == '00':
        return list2string([t[huns], t['100']])
    elif leadingZero(tens):
        numbers = [t[huns], t['100'], "OG", t[tens[1]]]
    else:
        numbers = [t[huns], t['100'], "OG", t[tens]]
    return list2string(numbers)
    

def thousands(string):
    '''Checks if a number is in the thousands'''
    if len(string) == 4 and string.isnumeric():
        return True
    return False


def writeThousands(s, t):
    '''Converts Danish numbers in the thousands to their spoken form'''
    thous = s[0]
    huns = s[1:4]
    numbers = [t[thous], t['1000']]
    if huns == '000':
        return list2string(numbers)
    elif huns[0] != '0':
        numbers.append(writeHundreds(huns, t))
    else:
        numbers.append("OG")
        if leadingZero(huns[1:3]):
            numbers.append(t[huns[2:3]])
        else:
            numbers.append(t[huns[1:3]])
    return list2string(numbers)


def writeNumber(tok, table, keys):
    '''Converts many numbers to their Danish spoken form'''
    try:
        if tok in keys:
            return table[tok]
        elif isDKCPR(tok):
            return writeOutCPR(tok, keys, table)
        elif leadingZero(tok):
            return writeZeroDigit(tok, keys, table)
        elif hundreds(tok):
            return writeHundreds(tok, table)
        elif thousands(tok):
            return writeThousands(tok, table)        
        else:
            return tok
    except KeyError:
        return tok


def splitNumeric(s, splitchar="([-,/])"):
    '''Splits a date, decimal or other token containind numbers or returns False'''
    parts = charsplit(splitchar, s)
    if len(parts) > 2 and parts[0].isnumeric():
        return parts
    return False    


def writeOutSplits(s):
    '''Writes common separators to their spoken form. Context-sensitive ''' 
    d = {"-": 'TIL',
         "/": 'SKRÅSTREG',
         "-": 'STREG'
         }
    splitchar = d.keys()
    
    if len(s) == 3:
        if s[2].isnumeric():
            if s[1] == "-":
                s1 = d[s[1]]
            elif s[1] == "/":
                s[1] == ""
        elif s[0].isalpha() and s[2].isnumeric():
            s[1] == ""
    else:
        for num, dig in enumerate(s):
            if dig in splitchar:
                s[num] = d[dig]
    return s
    
def rmPvAnnotation(string):
    if string[0] == "_" and string[-1] == "_":
#        print(string+ ": " +string.strip("_"))
        return string.strip("_")
    else:
        return string

def normNumber(line, table):
    tokens = line.split()
    keys = list(table.keys())
    for num, tok in enumerate(tokens):
        newtoks = splitNumeric(tok)
        if newtoks != False:
            newtoks = writeOutSplits(newtoks)
            written = [writeNumber(x, table, keys) for x in newtoks if x.isnumeric()]
            newstring = list2string(written)
        else:
            newstring = writeNumber(tok, table, keys)
        tokens[num] = newstring
    return list2string(tokens, newline=True)


def writeOutNumbers(infile, outfile, tablefile="numbers.tbl"):
    '''Uses a table of numbers-text to write out numbers in the infile.'''
    text = codecs.open(infile, "r", "utf8")
    fout = codecs.open(outfile, "w", "utf8")
    table = loadNumTable(tablefile)
#    keys = table.keys()
    for line in text:
        cleanline = normNumber(line, table)
        #print(cleanline)
        fout.write(cleanline)
     
    text.close()
    fout.close()


if __name__ == '__main__':
    
    try:
        tablefile = sys.argv[1]
        infile = sys.argv[2]
        outfile = sys.argv[3]

    except IndexError:
        print("python3 writenumbers.py <tablefile> <infile> <outfile>")
        sys.exit("Terminate")



    writeOutNumbers(infile, outfile, tablefile)
