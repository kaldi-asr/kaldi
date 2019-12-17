#!/usr/bin/env python3
# Copyright   2018 Chun-Chieh Chang

# This script reads in text and outputs the normalized version

import io
import re
import sys
import unicodedata

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")
for line in sys.stdin:
    line = line.strip()
    line = unicodedata.normalize('NFC', line)
    line = re.sub(r'\s', ' ', line)
    sys.stdout.write(line + '\n')
