#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys

charSet = {}
for line in sys.stdin:
    for char in line.strip():
        charSet[char] = True
for char in charSet:
    sys.stdout.write(char + "\n")
